"""Reproducible dataset generation, independent of command-line parsing."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from app.synthetic.workload import SyntheticWorkload

SERVICES = ["orders-service", "products-service", "payments-service"]
TRUE_EDGES = [
    ("orders-service", "products-service"),
    ("orders-service", "payments-service"),
]
BASELINE = {"orders-service": 15.0, "products-service": 6.0, "payments-service": 4.0}
EDGE_COLUMNS = [
    "window_utc",
    "logical_day",
    "scenario",
    "src",
    "dst",
    "tau",
    "edge_weight",
    "edge_rps",
    "edge_p95_ms",
    "edge_errors",
    "observed",
]
TRUTH_COLUMNS = EDGE_COLUMNS + ["hidden", "missing_edge_rate"]


@dataclass(frozen=True)
class GeneratorConfig:
    start: str
    end: str  # Exclusive boundary: two months need not equal 60 days.
    output_dir: str = "data/synthetic_2months"
    seed: int = 42
    step_minutes: int = 5
    missing_edge_rate: float = 0.30
    overwrite: bool = False

    def timestamps(self):
        start, end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        start = (
            start.tz_localize("UTC")
            if start.tzinfo is None
            else start.tz_convert("UTC")
        )
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end <= start or self.step_minutes <= 0 or 1440 % self.step_minutes:
            raise ValueError("Require start < end and a step dividing one day")
        if not 0 <= self.missing_edge_rate <= 1:
            raise ValueError("missing_edge_rate must be in [0, 1]")
        step = pd.Timedelta(minutes=self.step_minutes)
        if (end - start) % step:
            raise ValueError("Range must contain complete control intervals")
        return pd.date_range(start, end, freq=step, inclusive="left")


class SyntheticDatasetGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def generate(self):
        config = self.config
        timestamps = config.timestamps()
        output = Path(config.output_dir)
        outputs = [
            "nodes.csv",
            "edges.csv",
            "edges_ground_truth.csv",
            "scenarios.csv",
            "manifest.json",
        ]
        if not config.overwrite and any((output / n).exists() for n in outputs):
            raise FileExistsError(
                "Dataset already exists; choose another directory or use --overwrite"
            )
        # Reset the RNG for every generation, including repeated calls on one instance.
        model = SyntheticWorkload(config.seed)
        nodes, observed, truth, scenarios = [], [], [], []
        mask_rng = np.random.default_rng(config.seed + 1)
        mask_blocks = {}
        for t in timestamps:
            day = (t.date() - timestamps[0].date()).days + 1
            cycle_day = (day - 1) % 30 + 1
            scenario = model.scenario_for_day(cycle_day)
            flash = model.is_flash_event(t.to_pydatetime(), cycle_day)
            multiplier = model.scenario_multiplier(
                t.to_pydatetime(), cycle_day, scenario
            )
            orders = max(
                0.1,
                BASELINE["orders-service"] * multiplier * model.rng.uniform(0.92, 1.08),
            )
            outgoing = 0.85 * orders * model.rng.uniform(0.97, 1.03)
            shares = model.outgoing_shares()
            loads = {"orders-service": orders}
            for service, share in zip(SERVICES[1:], shares):
                loads[service] = max(
                    0.1,
                    BASELINE[service] * multiplier * model.rng.uniform(0.92, 1.08)
                    + outgoing * share,
                )
            iso = t.isoformat().replace("+00:00", "Z")
            partial = scenario == "partial_observability"
            service_data = {}
            for service in SERVICES:
                load = loads[service]
                p95 = model.p95_from_rps(load, service=service, flash=flash)
                p99 = model.p99_from_p95(p95, flash=flash)
                errors = model.error_rate_from_rps(load, flash=flash)
                replicas, cpu, mem = model.targets_from_load(load, p95, p99, errors)
                current = max(1, replicas + int(model.rng.choice([-1, 0, 0, 0, 1])))
                # Kubernetes resource requests are per replica; CPU usage is service total.
                target_cpu = max(
                    400.0,
                    min(
                        2000.0 if service == "orders-service" else 1200.0,
                        cpu / replicas,
                    ),
                )
                target_mem = max(
                    512.0, min(3072.0 if service == "orders-service" else 2048.0, mem)
                )
                feats = {
                    "cpu_mcores": model.cpu_from_rps(load),
                    "mem_mib": model.memory_from_rps(load),
                    "rps_in": load,
                    "rps_out": (
                        outgoing
                        if service == SERVICES[0]
                        else model.rng.uniform(0, 0.2)
                    ),
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "error_rate": errors,
                    "replicas": current,
                }
                service_data[service] = feats
                nodes.append(
                    {
                        "window_utc": iso,
                        "logical_day": day,
                        "scenario": scenario,
                        "flash_event": int(flash),
                        "partial_observability": int(partial),
                        "service": service,
                        **{k: round(v, 6) for k, v in feats.items()},
                        "cpu_request_m": round(target_cpu, 6),
                        "mem_request_mib": round(target_mem, 6),
                        "slo_risk": model.slo_risk_label(p95, p99, errors),
                        "target_replicas": replicas,
                        "target_cpu_m": round(target_cpu, 6),
                        "target_mem_mib": round(target_mem, 6),
                    }
                )
            for (src, dst), share in zip(TRUE_EDGES, shares):
                metrics = model.edge_metrics(src, dst, share, service_data)
                block = int((t - timestamps[0]).total_seconds() // (2 * 3600))
                mask_key = (block, src, dst)
                if mask_key not in mask_blocks:
                    mask_blocks[mask_key] = mask_rng.random()
                hidden = partial and mask_blocks[mask_key] < config.missing_edge_rate
                record = {
                    "window_utc": iso,
                    "logical_day": day,
                    "scenario": scenario,
                    "src": src,
                    "dst": dst,
                    "tau": (t - pd.Timedelta(minutes=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    **{k: round(v, 6) for k, v in metrics.items()},
                    "observed": int(not hidden),
                    "hidden": int(hidden),
                    "missing_edge_rate": config.missing_edge_rate if partial else 0.0,
                }
                truth.append(record)
                if not hidden:
                    observed.append({k: record[k] for k in EDGE_COLUMNS})
            scenarios.append(
                {
                    "window_utc": iso,
                    "logical_day": day,
                    "scenario": scenario,
                    "flash_event": int(flash),
                    "missing_edge_rate": config.missing_edge_rate if partial else 0.0,
                }
            )
        output.mkdir(parents=True, exist_ok=True)
        frames = {
            "nodes.csv": pd.DataFrame(nodes),
            "edges.csv": pd.DataFrame(observed, columns=EDGE_COLUMNS),
            "edges_ground_truth.csv": pd.DataFrame(truth, columns=TRUTH_COLUMNS),
            "scenarios.csv": pd.DataFrame(scenarios),
        }
        for name, frame in frames.items():
            frame.to_csv(output / name, index=False, lineterminator="\n")
        partial_truth = frames["edges_ground_truth.csv"].query(
            "scenario == 'partial_observability'"
        )
        manifest = {
            "schema_version": 2,
            "provenance": "synthetic_telemetry",
            "start_inclusive": timestamps[0].isoformat(),
            "end_exclusive": (
                timestamps[-1] + pd.Timedelta(minutes=config.step_minutes)
            ).isoformat(),
            "seed": config.seed,
            "step_minutes": config.step_minutes,
            "windows": len(timestamps),
            "services": SERVICES,
            "scenario_protocol": "repeat original 30-day blocks; final block may be truncated",
            "missing_edge_rate": config.missing_edge_rate,
            "mask_protocol": "independent 2-hour blocks; independent RNG from telemetry",
            "actual_partial_missing_rate": (
                float(partial_truth["hidden"].mean()) if len(partial_truth) else None
            ),
            "rows": {k: len(v) for k, v in frames.items()},
            "sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in frames
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest
