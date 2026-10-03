from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException
from kubernetes import client, config

from app.core.interfaces import Action, PolicyConfig, Prediction
from app.utils.parsers import (
    format_cpu_milli,
    format_mem_gi_from_mib,
    parse_cpu_milli,
    parse_mem_mib,
)
from config.settings import CONFIG
from app.core.state import JSONStateStore
from app.core.actuation import KubernetesActuator
from app.utils.parsers import parse_iso8601


class SafetyPolicy:
    """
    Deterministic constraint-aware decision layer.

    Responsibilities
    ----------------
    1. Multi-objective candidate evaluation:
       J =
         lambda_slo       * J_slo
         + lambda_cost    * J_cost
         + lambda_stability * J_stability
         + lambda_risk    * J_risk

    2. Replica/resource bounds.

    3. Replica rate limiting.

    4. CPU/memory step limiting.

    5. Hysteresis.

    6. Cooldown.

    7. Exponential smoothing.

    8. Kubernetes-compatible actuation.
    """

    def __init__(self, cfg: PolicyConfig, state_store=None, actuator=None):
        self.cfg = cfg
        self.memory_path = os.environ.get("TGAT_STATE_PATH", "./tgat_state.json")
        self.store = (
            state_store if state_store is not None else JSONStateStore(self.memory_path)
        )
        self.actuator = actuator if actuator is not None else KubernetesActuator(cfg)
        self.state = self.store.load()

    def _load_state(self):
        return self.store.load()

    def _save_state(self):
        self.store.save(self.state)

    def _service_cfg(self, service_id: str) -> Dict[str, Any]:
        return CONFIG.get("services", {}).get(service_id, {})

    def _replica_bounds(self, service_id: str) -> tuple[int, int]:
        svc_cfg = self._service_cfg(service_id)

        r_min = int(svc_cfg.get("min_replicas", self.cfg.r_min))

        r_max = int(svc_cfg.get("max_replicas", self.cfg.r_max))

        if r_max < r_min:
            r_max = r_min

        return (r_min, r_max)

    def _resource_bounds(self, service_id: str) -> tuple[float, float, float, float]:
        svc_cfg = self._service_cfg(service_id)

        min_cpu = parse_cpu_milli(svc_cfg.get("min_cpu", "100m"))

        max_cpu = parse_cpu_milli(svc_cfg.get("max_cpu", "4000m"))

        min_mem = parse_mem_mib(svc_cfg.get("min_memory", "256Mi"))

        max_mem = parse_mem_mib(svc_cfg.get("max_memory", "8192Mi"))

        return (min_cpu, max_cpu, min_mem, max_mem)

    # =========================================================================
    # MULTI-OBJECTIVE DECISION FUNCTION
    # =========================================================================

    def candidate_actions(
        self, prediction: Prediction, proposed: Action
    ) -> List[Action]:
        """
        Construct a finite local admissible action set.

        The neural model proposes a resource target.
        The deterministic layer evaluates nearby horizontal
        scaling alternatives around that target.
        """

        r_min, r_max = self._replica_bounds(proposed.id)

        rate_limit = max(1, int(self.cfg.rate_limit_replicas))

        replica_candidates = {
            int(np.clip(proposed.replicas, r_min, r_max)),
            int(np.clip(proposed.replicas - rate_limit, r_min, r_max)),
            int(np.clip(proposed.replicas + rate_limit, r_min, r_max)),
        }

        previous = self.state.get("last_actions", {}).get(proposed.id)

        if previous is not None:
            replica_candidates.add(
                int(np.clip(int(previous["replicas"]), r_min, r_max))
            )

        candidates = []

        for replicas in sorted(replica_candidates):
            candidates.append(
                Action(
                    id=proposed.id,
                    replicas=replicas,
                    cpu=proposed.cpu,
                    mem=proposed.mem,
                )
            )

        return candidates

    def objective(
        self,
        prediction: Prediction,
        candidate: Action,
        previous: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Multi-objective deterministic criterion.

        The individual terms are normalized to approximately [0, 1].
        """

        r_min, r_max = self._replica_bounds(candidate.id)

        min_cpu, max_cpu, min_mem, max_mem = self._resource_bounds(candidate.id)

        replicas = float(candidate.replicas)

        cpu = parse_cpu_milli(candidate.cpu) if candidate.cpu else min_cpu

        mem = parse_mem_mib(candidate.mem) if candidate.mem else min_mem

        # ---------------------------------------------------------------------
        # J_SLO
        #
        # Primary penalty is predicted SLO risk.
        # Tail latency contributes when it approaches configured thresholds.
        # ---------------------------------------------------------------------
        slo_cfg = CONFIG.get("slo", {})

        p95_limit = max(1.0, float(slo_cfg.get("p95_ms", 400.0)))

        p99_limit = max(1.0, float(slo_cfg.get("p99_ms", 600.0)))

        p95_ratio = float(prediction.p95_ms / p95_limit)

        p99_ratio = float(prediction.p99_ms / p99_limit)

        latency_penalty = float(np.clip(max(p95_ratio, p99_ratio), 0.0, 2.0) / 2.0)

        j_slo = float(
            np.clip(0.65 * prediction.slo_risk + 0.35 * latency_penalty, 0.0, 1.0)
        )

        # ---------------------------------------------------------------------
        # J_COST
        # ---------------------------------------------------------------------
        replica_norm = (replicas - r_min) / max(1.0, float(r_max - r_min))

        cpu_norm = (cpu - min_cpu) / max(1.0, max_cpu - min_cpu)

        mem_norm = (mem - min_mem) / max(1.0, max_mem - min_mem)

        resource_cfg = CONFIG.get("resource_cost", {})

        w_r = float(resource_cfg.get("replica_weight", 1.0 / 3.0))

        w_c = float(resource_cfg.get("cpu_weight", 1.0 / 3.0))

        w_m = float(resource_cfg.get("memory_weight", 1.0 / 3.0))

        weight_sum = w_r + w_c + w_m

        if weight_sum <= 0.0:
            w_r = w_c = w_m = 1.0 / 3.0
        else:
            w_r /= weight_sum
            w_c /= weight_sum
            w_m /= weight_sum

        j_cost = float(
            np.clip(w_r * replica_norm + w_c * cpu_norm + w_m * mem_norm, 0.0, 1.0)
        )

        # ---------------------------------------------------------------------
        # J_STABILITY
        # ---------------------------------------------------------------------
        if previous is None:
            j_stability = 0.0

        else:
            prev_replicas = float(previous.get("replicas", candidate.replicas))

            prev_cpu = parse_cpu_milli(previous.get("cpu") or candidate.cpu or "100m")

            prev_mem = parse_mem_mib(previous.get("mem") or candidate.mem or "256Mi")

            replica_change = abs(replicas - prev_replicas) / max(
                1.0, float(self.cfg.rate_limit_replicas)
            )

            cpu_change = abs(cpu - prev_cpu) / max(1.0, prev_cpu)

            mem_change = abs(mem - prev_mem) / max(1.0, prev_mem)

            j_stability = float(
                np.clip((replica_change + cpu_change + mem_change) / 3.0, 0.0, 1.0)
            )

        # ---------------------------------------------------------------------
        # J_RISK
        #
        # Penalizes actions that allocate less than predicted demand
        # under elevated SLO risk.
        # ---------------------------------------------------------------------
        predicted_replicas = max(1.0, float(prediction.replicas))

        predicted_cpu = max(1.0, float(prediction.cpu_milli))

        predicted_mem = max(1.0, float(prediction.mem_mib))

        replica_shortfall = max(
            0.0, (predicted_replicas - replicas) / predicted_replicas
        )

        cpu_shortfall = max(0.0, (predicted_cpu - cpu) / predicted_cpu)

        mem_shortfall = max(0.0, (predicted_mem - mem) / predicted_mem)

        under_provisioning = float(
            np.clip((replica_shortfall + cpu_shortfall + mem_shortfall) / 3.0, 0.0, 1.0)
        )

        # This is a deterministic capacity proxy, not a learned causal response model.
        # Evaluate demand against total CPU and per-replica memory requirements.
        cpu_capacity = max(1.0, replicas * cpu)
        cpu_demand = predicted_replicas * predicted_cpu
        demand_ratio = max(
            predicted_replicas / max(1.0, replicas),
            cpu_demand / cpu_capacity,
            predicted_mem / max(1.0, mem),
        )
        j_slo = float(np.clip(j_slo * demand_ratio, 0.0, 1.0))
        j_risk = float(
            np.clip(under_provisioning * (0.5 + 0.5 * prediction.slo_risk), 0.0, 1.0)
        )

        # ---------------------------------------------------------------------
        # Weighted scalarization
        # ---------------------------------------------------------------------
        value = (
            self.cfg.lambda_slo * j_slo
            + self.cfg.lambda_cost * j_cost
            + self.cfg.lambda_stability * j_stability
            + self.cfg.lambda_risk * j_risk
        )

        return float(value)

    def select_action(self, prediction: Prediction, proposed: Action) -> Action:
        """
        Select u* = argmin J(u)
        from a finite admissible candidate set.
        """

        previous = self.state.get("last_actions", {}).get(proposed.id)

        candidates = self.candidate_actions(prediction, proposed)

        if not candidates:
            return proposed

        return min(
            candidates,
            key=lambda candidate: self.objective(prediction, candidate, previous),
        )

    def select_actions(
        self, predictions: List[Prediction], proposed_actions: List[Action]
    ) -> List[Action]:
        pred_map = {p.id: p for p in predictions}

        result = []

        for action in proposed_actions:
            prediction = pred_map.get(action.id)

            if prediction is None:
                result.append(action)
                continue

            result.append(self.select_action(prediction, action))

        return result

    # =========================================================================
    # COOLDOWN
    # =========================================================================

    def _cooldown_active(
        self, service_id: str, *, critical: bool = False, now_ts: Optional[float] = None
    ) -> bool:
        if critical:
            return False

        if self.cfg.cooldown_sec <= 0:
            return False

        timestamps = self.state.get("last_action_ts", {})

        last_ts = timestamps.get(service_id)

        if last_ts is None:
            return False

        try:
            elapsed = (time.time() if now_ts is None else now_ts) - float(last_ts)
        except Exception:
            return False

        return elapsed < self.cfg.cooldown_sec

    # =========================================================================
    # RESOURCE CONSTRAINTS
    # =========================================================================

    def _clip_resources(self, action: Action) -> Action:
        min_cpu, max_cpu, min_mem, max_mem = self._resource_bounds(action.id)

        r_min, r_max = self._replica_bounds(action.id)

        replicas = int(np.clip(action.replicas, r_min, r_max))

        cpu = action.cpu
        mem = action.mem

        if cpu:
            cpu_value = parse_cpu_milli(cpu)

            cpu_value = float(np.clip(cpu_value, min_cpu, max_cpu))

            cpu = format_cpu_milli(cpu_value)

        if mem:
            mem_value = parse_mem_mib(mem)

            mem_value = float(np.clip(mem_value, min_mem, max_mem))

            mem = format_mem_gi_from_mib(mem_value)

        return Action(id=action.id, replicas=replicas, cpu=cpu, mem=mem)

    def _rate_limit(self, action: Action, previous: Dict[str, Any]) -> Action:
        """
        Limit horizontal and vertical action magnitude.
        """

        prev_replicas = int(previous.get("replicas", action.replicas))

        desired_replicas = int(action.replicas)

        replica_delta = desired_replicas - prev_replicas

        max_replica_step = max(0, int(self.cfg.rate_limit_replicas))

        if max_replica_step > 0 and abs(replica_delta) > max_replica_step:
            desired_replicas = prev_replicas + (
                max_replica_step if replica_delta > 0 else -max_replica_step
            )

        desired_cpu = action.cpu
        desired_mem = action.mem

        prev_cpu_s = previous.get("cpu")

        prev_mem_s = previous.get("mem")

        if desired_cpu and prev_cpu_s and self.cfg.cpu_step_pct > 0.0:
            prev_cpu = parse_cpu_milli(prev_cpu_s)

            cpu = parse_cpu_milli(desired_cpu)

            max_change = abs(prev_cpu) * self.cfg.cpu_step_pct

            cpu = float(np.clip(cpu, prev_cpu - max_change, prev_cpu + max_change))

            desired_cpu = format_cpu_milli(cpu)

        if desired_mem and prev_mem_s and self.cfg.mem_step_pct > 0.0:
            prev_mem = parse_mem_mib(prev_mem_s)

            mem = parse_mem_mib(desired_mem)

            max_change = abs(prev_mem) * self.cfg.mem_step_pct

            mem = float(np.clip(mem, prev_mem - max_change, prev_mem + max_change))

            desired_mem = format_mem_gi_from_mib(mem)

        return Action(
            id=action.id, replicas=desired_replicas, cpu=desired_cpu, mem=desired_mem
        )

    def _smooth_action(
        self, action: Action, previous: Optional[Dict[str, Any]]
    ) -> Action:
        if previous is None:
            return action

        alpha = float(np.clip(self.cfg.smoothing_alpha, 0.0, 1.0))

        if alpha >= 1.0:
            return action

        cpu = action.cpu
        mem = action.mem

        prev_cpu_s = previous.get("cpu")

        prev_mem_s = previous.get("mem")

        if cpu and prev_cpu_s:
            new_cpu = parse_cpu_milli(cpu)

            old_cpu = parse_cpu_milli(prev_cpu_s)

            smoothed_cpu = alpha * new_cpu + (1.0 - alpha) * old_cpu

            cpu = format_cpu_milli(smoothed_cpu)

        if mem and prev_mem_s:
            new_mem = parse_mem_mib(mem)

            old_mem = parse_mem_mib(prev_mem_s)

            smoothed_mem = alpha * new_mem + (1.0 - alpha) * old_mem

            mem = format_mem_gi_from_mib(smoothed_mem)

        return Action(id=action.id, replicas=action.replicas, cpu=cpu, mem=mem)

    # =========================================================================
    # HYSTERESIS
    # =========================================================================

    @staticmethod
    def _action_changed(current: Action, previous: Optional[Dict[str, Any]]) -> bool:
        if previous is None:
            return True

        return (
            int(previous.get("replicas", current.replicas)) != int(current.replicas)
            or previous.get("cpu") != current.cpu
            or previous.get("mem") != current.mem
        )

    # =========================================================================
    # FINAL SAFETY FILTER
    # =========================================================================

    def filter(
        self,
        actions: List[Action],
        tstamp: str,
        predictions: Optional[List[Prediction]] = None,
        current_actions: Optional[List[Action]] = None,
        persist: bool = True,
    ) -> List[Action]:
        """
        Apply deterministic operational constraints.

        predictions are optional and are used only to detect critical
        SLO risk that may bypass ordinary cooldown.
        """

        pred_map = {p.id: p for p in (predictions or [])}

        prev_actions: Dict[str, Dict[str, Any]] = self.state.get("last_actions", {})

        counters: Dict[str, int] = self.state.get("hysteresis", {})

        last_action_ts: Dict[str, float] = self.state.get("last_action_ts", {})

        final_actions: List[Action] = []

        now_ts = parse_iso8601(tstamp).timestamp()
        previous_window = self.state.get("last_window")
        if previous_window and now_ts < parse_iso8601(previous_window).timestamp():
            raise ValueError("Safety windows must be chronological")
        if previous_window and now_ts == parse_iso8601(previous_window).timestamp():
            # Replaying the same control window must not advance hysteresis/cooldown.
            return [
                Action(id=a.id, **prev_actions[a.id])
                for a in actions
                if a.id in prev_actions
            ]
        for current in current_actions or []:
            prev_actions[current.id] = current.model_dump(exclude={"id"})
        signatures = self.state.setdefault("hysteresis_signatures", {})

        for original_action in actions:
            action = self._clip_resources(original_action)

            previous = prev_actions.get(action.id)

            prediction = pred_map.get(action.id)

            critical = bool(
                prediction is not None
                and prediction.slo_risk >= self.cfg.critical_slo_risk
            )

            # -----------------------------------------------------------------
            # Exponential smoothing
            # -----------------------------------------------------------------
            action = self._smooth_action(action, previous)

            # -----------------------------------------------------------------
            # Per-step action limits
            # -----------------------------------------------------------------
            if previous is not None:
                action = self._rate_limit(action, previous)

            action = self._clip_resources(action)

            changed = self._action_changed(action, previous)

            # -----------------------------------------------------------------
            # Cooldown
            # -----------------------------------------------------------------
            if (
                changed
                and previous is not None
                and self._cooldown_active(action.id, critical=critical, now_ts=now_ts)
            ):
                action = Action(
                    id=action.id,
                    replicas=int(previous["replicas"]),
                    cpu=previous.get("cpu"),
                    mem=previous.get("mem"),
                )

                changed = False

            # -----------------------------------------------------------------
            # Hysteresis
            # -----------------------------------------------------------------
            if (
                changed
                and previous is not None
                and not critical
                and self.cfg.hysteresis_windows > 1
            ):
                # Consecutive observations must request the same direction.
                def direction(new, old):
                    return int(new > old) - int(new < old)

                signature = [
                    direction(action.replicas, previous["replicas"]),
                    (
                        direction(
                            parse_cpu_milli(action.cpu),
                            parse_cpu_milli(previous["cpu"]),
                        )
                        if action.cpu and previous.get("cpu")
                        else 0
                    ),
                    (
                        direction(
                            parse_mem_mib(action.mem), parse_mem_mib(previous["mem"])
                        )
                        if action.mem and previous.get("mem")
                        else 0
                    ),
                ]
                counter = (
                    counters.get(action.id, 0) + 1
                    if signatures.get(action.id) == signature
                    else 1
                )
                signatures[action.id] = signature

                counters[action.id] = counter

                if counter < self.cfg.hysteresis_windows:
                    action = Action(
                        id=action.id,
                        replicas=int(previous["replicas"]),
                        cpu=previous.get("cpu"),
                        mem=previous.get("mem"),
                    )

                    changed = False

                else:
                    counters[action.id] = 0

            else:
                if not changed:
                    counters[action.id] = 0

            action = self._clip_resources(action)

            final_actions.append(action)

            if self._action_changed(action, previous):
                last_action_ts[action.id] = now_ts

        self.state["last_window"] = tstamp

        self.state["last_actions"] = {
            **prev_actions,
            **{
                action.id: {
                    "replicas": action.replicas,
                    "cpu": action.cpu,
                    "mem": action.mem,
                }
                for action in final_actions
            },
        }

        self.state["hysteresis"] = counters

        self.state["last_action_ts"] = last_action_ts

        if persist:
            self._save_state()

        return final_actions

    # =========================================================================
    # KUBERNETES ACTUATION
    # =========================================================================

    def apply_to_k8s(self, actions: List[Action]) -> Dict[str, Any]:
        return self.actuator.apply(actions)
