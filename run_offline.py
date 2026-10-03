"""Train and replay CSV telemetry without contacting Prometheus or Kubernetes."""

from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from app.core.model import TGATAutoscalerModel
from app.core.safety import SafetyPolicy
from app.core.state import MemoryStateStore
from app.data.csv_dataset import CSVGraphDataset
from app.dto.train_csv_request_dto import TrainCSVRequest
from app.service.tgat_service import TGATService
from app.training.trainer import CSVTrainer
from config.settings import DEFAULT_MODEL_CFG, TARGET_ORDER


def replay(dataset, checkpoint_path, output_path, start=None, truth_path=None):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = TGATAutoscalerModel(
        {**DEFAULT_MODEL_CFG, "model_path": str(checkpoint_path)}
    )
    if not model.load_checkpoint(str(checkpoint_path)):
        raise ValueError(
            getattr(model, "checkpoint_error", "Checkpoint is unavailable")
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Default evaluation uses only validation period; the report states this explicitly.
    start = pd.Timestamp(start or ckpt["validation_start"])
    policy_cfg = TGATService._build_policy_config()
    policy_cfg.dry_run = True
    policy = SafetyPolicy(policy_cfg, state_store=MemoryStateStore())
    error_abs = np.zeros(6)
    error_sq = np.zeros(6)
    risk_sq = 0.0
    count = 0
    windows = 0
    changes = 0
    baseline_abs = np.zeros(6)
    baseline_sq = np.zeros(6)
    edge_counts = {"tp": 0, "fp": 0, "fn": 0}
    risk_counts = {"tp": 0, "fp": 0, "fn": 0, "positive": 0, "negative": 0}
    truth_by_window = {}
    if truth_path is not None:
        truth_frame = pd.read_csv(truth_path)
        truth_frame["window_utc"] = pd.to_datetime(truth_frame["window_utc"], utc=True)
        truth_by_window = {
            t: set(zip(g.src, g.dst)) for t, g in truth_frame.groupby("window_utc")
        }
    action_rows = []
    forecast_rows = []
    edge_rows = []
    runtime = []
    target_names = TARGET_ORDER[:6]
    for sample in dataset.iter_samples():
        if pd.Timestamp(sample.graph.window) < start:
            continue
        started = time.perf_counter()
        x, ei, ea = model.build_graph_from_payload(sample.graph)
        predictions = model.predict_targets(x, ei, ea)
        proposed = model.predictions_to_actions(predictions)
        selected = policy.select_actions(predictions, proposed)
        before = policy.state.get("last_actions", {})
        safe = policy.filter(selected, sample.graph.window, predictions, persist=False)
        elapsed = (time.perf_counter() - started) * 1000
        runtime.append(elapsed)
        values = np.array(
            [
                [
                    p.rps,
                    p.replicas,
                    p.cpu_milli,
                    p.mem_mib,
                    p.p95_ms,
                    p.p99_ms,
                    p.slo_risk,
                ]
                for p in predictions
            ]
        )
        target = sample.targets.numpy()
        errors = values[:, :6] - target[:, :6]
        current = dataset.groups[pd.Timestamp(sample.graph.window)].loc[
            dataset.services
        ]
        baseline = current[
            [
                "rps_in",
                "target_replicas",
                "target_cpu_m",
                "target_mem_mib",
                "p95_ms",
                "p99_ms",
            ]
        ].to_numpy()
        baseline_error = baseline - target[:, :6]
        baseline_abs += np.abs(baseline_error).sum(axis=0)
        baseline_sq += (baseline_error**2).sum(axis=0)
        positive = target[:, 6] >= 0.5
        predicted_positive = values[:, 6] >= 0.5
        risk_counts["tp"] += int((positive & predicted_positive).sum())
        risk_counts["fp"] += int((~positive & predicted_positive).sum())
        risk_counts["fn"] += int((positive & ~predicted_positive).sum())
        risk_counts["positive"] += int(positive.sum())
        risk_counts["negative"] += int((~positive).sum())
        if truth_by_window:
            from app.evaluation.metrics import precision_recall_f1

            true_pairs = truth_by_window.get(pd.Timestamp(sample.graph.window), set())
            observed_pairs = {(e.src, e.dst) for e in sample.graph.events}
            reconstructed = {
                (e["src"], e["dst"]) for e in model.get_last_hidden_edges()
            }
            metrics = precision_recall_f1(true_pairs, reconstructed, observed_pairs)
            for key in edge_counts:
                edge_counts[key] += metrics[key]
        error_abs += np.abs(errors).sum(axis=0)
        error_sq += (errors**2).sum(axis=0)
        risk_sq += ((values[:, 6] - target[:, 6]) ** 2).sum()
        count += len(predictions)
        windows += 1
        for p, y in zip(predictions, target):
            forecast_rows.append(
                {
                    "window_utc": sample.graph.window,
                    **p.model_dump(),
                    **{f"true_{k}": float(v) for k, v in zip(TARGET_ORDER, y)},
                }
            )
        for a in safe:
            record = a.model_dump(exclude={"id"})
            changes += int(before.get(a.id) != record)
            action_rows.append({"window_utc": sample.graph.window, **a.model_dump()})
        for edge in model.get_last_hidden_edges():
            edge_rows.append({"window_utc": sample.graph.window, **edge})
        if windows % 1000 == 0:
            print(f"replayed {windows} windows", flush=True)
    if not count:
        raise ValueError("No evaluation windows")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecast_rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(action_rows).to_csv(output / "actions.csv", index=False)
    pd.DataFrame(
        edge_rows,
        columns=[
            "window_utc",
            "src",
            "dst",
            "probability",
            "rps_similarity",
            "latency_similarity",
            "architectural_prior",
        ],
    ).to_csv(output / "hidden_edges.csv", index=False)

    def rates(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }

    report = {
        "mode": "open_loop_csv_replay",
        "provenance": "synthetic_telemetry",
        "kubernetes_mutations": 0,
        "evaluation_start": start.isoformat(),
        "windows": windows,
        "forecast_rows": count,
        "regression": {
            k: {"mae": float(a / count), "rmse": float(np.sqrt(b / count))}
            for k, a, b in zip(target_names, error_abs, error_sq)
        },
        "risk_brier": float(risk_sq / count),
        "risk_classification": rates(risk_counts),
        "hidden_edge_reconstruction": (
            rates(edge_counts) if truth_path is not None else None
        ),
        "persistence_baseline": {
            k: {"mae": float(a / count), "rmse": float(np.sqrt(b / count))}
            for k, a, b in zip(target_names, baseline_abs, baseline_sq)
        },
        "includes_training_period": start < pd.Timestamp(ckpt["validation_start"]),
        "proposed_safe_changes": changes,
        "controller_ms": {
            "mean": float(np.mean(runtime)),
            "p95": float(np.quantile(runtime, 0.95)),
        },
        "limitations": [
            "Actions do not change recorded telemetry; this measures forecasting and control decisions.",
            "Validation data select checkpoints and are not an independent final test set.",
            "Measured SLO improvement, recovery and billing require a closed-loop application experiment.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "replay"])
    parser.add_argument("--data-dir", default="data/synthetic_2months")
    parser.add_argument("--model", default="artifacts/synthetic_2months.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", default="out/synthetic_2months")
    parser.add_argument(
        "--start", help="Replay from this UTC timestamp; defaults to validation period"
    )
    args = parser.parse_args()
    root = Path(args.data_dir)
    if args.command == "train":
        model = TGATAutoscalerModel(DEFAULT_MODEL_CFG)
        report = CSVTrainer(model).train(
            TrainCSVRequest(
                nodes_csv_path=str(root / "nodes.csv"),
                edges_csv_path=str(root / "edges.csv"),
                epochs=args.epochs,
                model_path=args.model,
                device="cpu",
            )
        )
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "training.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    else:
        report = replay(
            CSVGraphDataset(root / "nodes.csv", root / "edges.csv"),
            args.model,
            args.output_dir,
            args.start,
            root / "edges_ground_truth.csv",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
