from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import HTTPException

from app.core.graph import PrometheusGraphProvider
from app.core.interfaces import PolicyConfig, GraphWindow
from app.core.model import TGATAutoscalerModel
from app.core.safety import SafetyPolicy
from app.dto.train_csv_request_dto import TrainCSVRequest
from config.settings import CONFIG, DEFAULT_MODEL_CFG, MODEL_OUTPUT_DIM, MODEL_PATH


class TGATService:
    def __init__(self, model=None, graph_provider=None, policy_factory=None):
        self.safety_enabled = True
        self.model = (
            model if model is not None else TGATAutoscalerModel(DEFAULT_MODEL_CFG)
        )
        self.graph_provider = graph_provider or PrometheusGraphProvider(CONFIG)
        self.policy_factory = policy_factory or (
            lambda: SafetyPolicy(self._build_policy_config())
        )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    @staticmethod
    def _build_policy_config() -> PolicyConfig:
        safety = CONFIG.get("safety", {})

        objective = CONFIG.get("objective", {})

        return PolicyConfig(
            hysteresis_windows=int(safety.get("hysteresis_windows", 2)),
            rate_limit_replicas=int(safety.get("rate_limit_replicas", 2)),
            cpu_step_pct=float(safety.get("cpu_step_pct", 0.20)),
            mem_step_pct=float(safety.get("mem_step_pct", 0.20)),
            cooldown_sec=int(safety.get("cooldown_sec", 600)),
            smoothing_alpha=float(safety.get("smoothing_alpha", 0.60)),
            r_min=int(safety.get("r_min", 1)),
            r_max=int(safety.get("r_max", 50)),
            lambda_slo=float(objective.get("lambda_slo", 0.35)),
            lambda_cost=float(objective.get("lambda_cost", 0.25)),
            lambda_stability=float(objective.get("lambda_stability", 0.20)),
            lambda_risk=float(objective.get("lambda_risk", 0.20)),
            critical_slo_risk=float(safety.get("critical_slo_risk", 0.90)),
            dry_run=bool(safety.get("dry_run", True)),
        )

    # =========================================================================
    # PREDICTION
    # =========================================================================

    async def predict(
        self, graph_window: Optional[GraphWindow] = None
    ) -> Dict[str, Any]:
        """
        Forecast only.

        No safety filtering and no Kubernetes mutation are performed.
        """

        started = time.perf_counter()

        graph_window = (
            graph_window if graph_window is not None else self.graph_provider()
        )

        graph_started = time.perf_counter()

        x, edge_index, edge_attr = self.model.build_graph_from_payload(graph_window)

        graph_ms = (time.perf_counter() - graph_started) * 1000.0

        inference_started = time.perf_counter()

        predictions = self.model.predict_targets(x, edge_index, edge_attr)

        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        proposed_actions = self.model.predictions_to_actions(predictions)

        total_ms = (time.perf_counter() - started) * 1000.0

        return {
            "window": graph_window.window,
            "predictions": [prediction.model_dump() for prediction in predictions],
            "proposed_actions": [action.model_dump() for action in proposed_actions],
            "hidden_edges": self.model.get_last_hidden_edges(),
            "ablation": self.model.ablation_state(),
            "profiling": {
                "graph_build_ms": round(graph_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "total_ms": round(total_ms, 3),
            },
        }

    # =========================================================================
    # APPLY
    # =========================================================================

    async def apply(self, graph_window: Optional[GraphWindow] = None) -> Dict[str, Any]:
        """
        Complete telemetry -> prediction -> decision -> safety -> K8s loop.
        """

        total_started = time.perf_counter()

        # ---------------------------------------------------------------------
        # Telemetry / graph
        # ---------------------------------------------------------------------
        collection_started = time.perf_counter()

        graph_window = (
            graph_window if graph_window is not None else self.graph_provider()
        )

        collection_ms = (time.perf_counter() - collection_started) * 1000.0

        graph_started = time.perf_counter()

        x, edge_index, edge_attr = self.model.build_graph_from_payload(graph_window)

        graph_ms = (time.perf_counter() - graph_started) * 1000.0

        # ---------------------------------------------------------------------
        # Neural forecasting
        # ---------------------------------------------------------------------
        inference_started = time.perf_counter()

        predictions = self.model.predict_targets(x, edge_index, edge_attr)

        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        proposed_actions = self.model.predictions_to_actions(predictions)

        # ---------------------------------------------------------------------
        # Multi-objective decision + operational safety layer
        # ---------------------------------------------------------------------
        decision_started = time.perf_counter()

        policy = self.policy_factory()
        import copy

        before_state = copy.deepcopy(policy.state)
        from app.core.interfaces import Action
        from app.utils.parsers import format_cpu_milli, format_mem_gi_from_mib

        for node in graph_window.nodes:
            meta = node.meta or {}
            previous = policy.state.get("last_actions", {}).get(node.id, {})
            # Horizontal constraints begin from observed replicas on the first cycle.
            # Usage telemetry must never be mistaken for allocated per-pod resources.
            allocation = Action(
                id=node.id,
                replicas=max(1, int(round(node.x[-1]))),
                cpu=(
                    format_cpu_milli(meta["cpu_request_m"])
                    if "cpu_request_m" in meta
                    else previous.get("cpu")
                ),
                mem=(
                    format_mem_gi_from_mib(meta["mem_request_mib"])
                    if "mem_request_mib" in meta
                    else previous.get("mem")
                ),
            )
            policy.state.setdefault("last_actions", {})[node.id] = (
                allocation.model_dump(exclude={"id"})
            )

        if self.safety_enabled:
            selected_actions = policy.select_actions(predictions, proposed_actions)
            safe_actions = policy.filter(
                selected_actions,
                graph_window.window,
                predictions=predictions,
                persist=False,
            )
        else:
            # Ablation removes operational stabilization; service bounds remain enforced.
            selected_actions = safe_actions = proposed_actions
            policy.state["last_window"] = graph_window.window
            policy.state["last_actions"] = {
                a.id: a.model_dump(exclude={"id"}) for a in safe_actions
            }

        decision_ms = (time.perf_counter() - decision_started) * 1000.0

        # ---------------------------------------------------------------------
        # Kubernetes actuation
        # ---------------------------------------------------------------------
        actuation_started = time.perf_counter()

        try:
            report = policy.apply_to_k8s(safe_actions)
        except Exception:
            policy.state = before_state
            raise
        if not report.get("dry_run", False):
            successful = set(report.get("patched", []))
            # Failed or partially applied patches must not be recorded as completed.
            for action in safe_actions:
                if action.id not in successful:
                    for key in (
                        "last_actions",
                        "last_action_ts",
                        "hysteresis",
                        "hysteresis_signatures",
                    ):
                        prior = before_state.get(key, {})
                        if action.id in prior:
                            policy.state.setdefault(key, {})[action.id] = prior[
                                action.id
                            ]
                        else:
                            policy.state.get(key, {}).pop(action.id, None)
        if not report.get("dry_run", False) and not report.get("patched"):
            # A wholly failed window remains retryable and cannot advance the control clock.
            policy.state = before_state
        policy._save_state()

        actuation_ms = (time.perf_counter() - actuation_started) * 1000.0

        total_ms = (time.perf_counter() - total_started) * 1000.0

        control_period_ms = float(CONFIG.get("metrics_interval", 300)) * 1000.0

        occupancy_pct = (
            (total_ms / control_period_ms) * 100.0 if control_period_ms > 0.0 else 0.0
        )

        return {
            "window": graph_window.window,
            "predictions": [prediction.model_dump() for prediction in predictions],
            "proposed_actions": [action.model_dump() for action in proposed_actions],
            "selected_actions": [action.model_dump() for action in selected_actions],
            "applied": [
                action.model_dump()
                for action in safe_actions
                if report.get("dry_run") or action.id in report.get("patched", [])
            ],
            "hidden_edges": self.model.get_last_hidden_edges(),
            "report": report,
            "profiling": {
                "telemetry_ms": round(collection_ms, 3),
                "graph_build_ms": round(graph_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "decision_ms": round(decision_ms, 3),
                "actuation_ms": round(actuation_ms, 3),
                "total_ms": round(total_ms, 3),
                "control_interval_occupancy_pct": round(occupancy_pct, 6),
            },
            "ablation": self.model.ablation_state(),
        }

    # =========================================================================
    # ABLATIONS
    # =========================================================================

    def ablate(
        self,
        *,
        time_encoding: Optional[bool] = None,
        hidden_edges: Optional[bool] = None,
        graph_enabled: Optional[bool] = None,
        dropedge_prob: Optional[float] = None,
        history_minutes: Optional[int] = None,
        safety_enabled: Optional[bool] = None,
        hysteresis_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if dropedge_prob is not None and not 0 <= dropedge_prob <= 1:
            raise ValueError("dropedge_prob must be in [0, 1]")
        if history_minutes is not None:
            if history_minutes <= 0:
                raise ValueError("history_minutes " "must be > 0")

            CONFIG["observation_window_minutes"] = int(history_minutes)

            CONFIG["edge_feature_window_sec"] = int(history_minutes * 60)

        state = self.model.configure_ablation(
            time_encoding=(time_encoding),
            hidden_edges=(hidden_edges),
            graph_enabled=(graph_enabled),
            dropedge_prob=(dropedge_prob),
        )

        if safety_enabled is not None:
            self.safety_enabled = safety_enabled
        if hysteresis_enabled is not None:
            CONFIG["safety"]["hysteresis_windows"] = 2 if hysteresis_enabled else 0
        state["safety_enabled"] = self.safety_enabled
        state["history_minutes"] = CONFIG.get("observation_window_minutes", 60)

        return state

    # =========================================================================
    # TRAINING
    # =========================================================================

    async def train_from_csv(self, req: TrainCSVRequest) -> Dict[str, Any]:
        from app.training.trainer import CSVTrainer

        try:
            return CSVTrainer(self.model).train(req)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
