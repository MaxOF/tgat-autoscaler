"""Causal CSV dataset adapter: observed history and exact future labels."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from app.core.interfaces import EdgeEvent, GraphWindow, NodeFeatures
from config.settings import CONFIG, FEATURE_ORDER, TARGET_ORDER


def _to_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> None:
    if column not in df.columns:
        df[column] = default

    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)


def _normalize_nodes_dataframe(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize current and legacy nodes.csv files to the revised
    eight-feature representation.
    """

    for column in [
        "cpu_mcores",
        "mem_mib",
        "rps_in",
        "rps_out",
        "p95_ms",
        "error_rate",
    ]:
        _to_numeric_column(nodes_df, column, 0.0)

    # p99 was not present in the original dataset.
    if "p99_ms" not in (nodes_df.columns):
        nodes_df["p99_ms"] = nodes_df["p95_ms"] * 1.35

    else:
        _to_numeric_column(nodes_df, "p99_ms", 0.0)

        missing_p99 = nodes_df["p99_ms"] <= 0.0

        nodes_df.loc[missing_p99, "p99_ms"] = nodes_df.loc[missing_p99, "p95_ms"] * 1.35

    # The original synthetic CSV contains target_replicas but not
    # current replicas. Use target_replicas as a compatibility proxy
    # only when the current replica count is unavailable.
    if "replicas" not in (nodes_df.columns):
        if "target_replicas" in nodes_df.columns:
            nodes_df["replicas"] = pd.to_numeric(
                nodes_df["target_replicas"], errors="coerce"
            ).fillna(1.0)

        else:
            nodes_df["replicas"] = 1.0

    else:
        _to_numeric_column(nodes_df, "replicas", 1.0)

    if "target_replicas" not in nodes_df.columns:
        nodes_df["target_replicas"] = nodes_df["replicas"]

    _to_numeric_column(nodes_df, "target_replicas", 1.0)

    if "target_cpu_m" not in nodes_df.columns:
        nodes_df["target_cpu_m"] = nodes_df["cpu_mcores"]

    _to_numeric_column(nodes_df, "target_cpu_m", 0.0)

    if "target_mem_mib" not in nodes_df.columns:
        nodes_df["target_mem_mib"] = nodes_df["mem_mib"]

    _to_numeric_column(nodes_df, "target_mem_mib", 0.0)

    return nodes_df


def _slo_risk_label(p95_ms: float, p99_ms: float, error_rate: float) -> float:
    slo = CONFIG.get("slo", {})

    p95_limit = float(slo.get("p95_ms", 400.0))

    p99_limit = float(slo.get("p99_ms", 600.0))

    error_limit = float(slo.get("error_rate", 0.01))

    violation = p95_ms > p95_limit or p99_ms > p99_limit or error_rate > error_limit

    return 1.0 if violation else 0.0


@dataclass
class GraphSample:
    graph: GraphWindow
    targets: torch.Tensor


class CSVGraphDataset:
    """Index windows once and retain only observed events in each history."""

    def __init__(
        self,
        nodes_path,
        edges_path=None,
        time_col="window_utc",
        history_minutes=None,
        horizon_minutes=None,
    ):
        self.time_col = time_col
        self.history_minutes = history_minutes or CONFIG["observation_window_minutes"]
        self.horizon_minutes = horizon_minutes or CONFIG["forecast_horizon_minutes"]
        nodes = pd.read_csv(nodes_path)
        if not {time_col, "service"}.issubset(nodes.columns):
            raise ValueError("nodes CSV requires time and service columns")
        nodes[time_col] = pd.to_datetime(nodes[time_col], utc=True, errors="raise")
        if nodes.empty or nodes[time_col].isna().any() or nodes["service"].isna().any():
            raise ValueError("nodes CSV contains empty identities or timestamps")
        if nodes.duplicated([time_col, "service"]).any():
            raise ValueError("Duplicate service/window rows")
        self.nodes = _normalize_nodes_dataframe(nodes).sort_values(
            [time_col, "service"]
        )
        numeric = self.nodes[
            FEATURE_ORDER + ["target_replicas", "target_cpu_m", "target_mem_mib"]
        ]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite telemetry or targets")
        self.services = sorted(self.nodes["service"].unique())
        self.groups = {
            t: g.set_index("service") for t, g in self.nodes.groupby(time_col)
        }
        self.timestamps = sorted(self.groups)
        self.edges = {}
        if edges_path is not None:
            # An explicitly supplied file must exist; an empty graph is meaningful.
            edges = pd.read_csv(edges_path)
            if not {time_col, "src", "dst"}.issubset(edges.columns):
                raise ValueError("edges CSV requires time, src and dst columns")
            edges[time_col] = pd.to_datetime(edges[time_col], utc=True, errors="raise")
            for row in edges.to_dict("records"):
                if row["src"] not in self.services or row["dst"] not in self.services:
                    raise ValueError("Edge endpoint is absent from node data")
                if not bool(row.get("observed", 1)):
                    continue
                t = row[time_col]
                tau = pd.Timestamp(row.get("tau", t - pd.Timedelta(minutes=1)))
                if tau.tzinfo is None:
                    raise ValueError("Edge timestamp must include timezone")
                # At a CSV window's right boundary, events must already have occurred.
                if tau >= t:
                    raise ValueError("Edge event must precede its observation window")
                values = [
                    float(row.get(k, 0))
                    for k in ("edge_weight", "edge_p95_ms", "edge_errors")
                ]
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite edge features")
                self.edges.setdefault(t, []).append(
                    EdgeEvent(
                        src=row["src"],
                        dst=row["dst"],
                        tau=tau.isoformat(),
                        e=values,
                        confidence=float(row.get("confidence", 1)),
                    )
                )
        # Topology priors remain available to reconstruction through CONFIG;
        # they are never inserted here as measured edges.

    def feature_scaler(self, timestamps):
        values = np.concatenate(
            [
                self.groups[t][FEATURE_ORDER].to_numpy(dtype=np.float32)
                for t in timestamps
            ]
        )
        return values.mean(axis=0), np.maximum(values.std(axis=0), 1.0)

    def iter_samples(self):
        history = deque()
        horizon = pd.Timedelta(minutes=self.horizon_minutes)
        window = pd.Timedelta(minutes=self.history_minutes)
        for t in self.timestamps:
            for event in self.edges.get(t, []):
                history.append(event)
            lower = t - window
            # Compare event timestamps rather than their aggregation bucket.
            history = deque(e for e in history if pd.Timestamp(e.tau) >= lower)
            future = t + horizon
            if future not in self.groups:
                continue
            current, following = self.groups[t], self.groups[future]
            # Missing targets are not replaced with current labels.
            if not set(self.services).issubset(current.index) or not set(
                self.services
            ).issubset(following.index):
                continue
            nodes, targets = [], []
            for service in self.services:
                row, nxt = current.loc[service], following.loc[service]
                feats = {k: float(row[k]) for k in FEATURE_ORDER}
                meta = {"features": feats}
                for key in ("cpu_request_m", "mem_request_mib"):
                    if key in row:
                        meta[key] = float(row[key])
                nodes.append(
                    NodeFeatures(
                        id=service, x=[feats[k] for k in FEATURE_ORDER], meta=meta
                    )
                )
                targets.append(
                    [
                        float(nxt["rps_in"]),
                        float(nxt["target_replicas"]),
                        float(nxt["target_cpu_m"]),
                        float(nxt["target_mem_mib"]),
                        float(nxt["p95_ms"]),
                        float(nxt["p99_ms"]),
                        _slo_risk_label(
                            nxt["p95_ms"], nxt["p99_ms"], nxt["error_rate"]
                        ),
                    ]
                )
            yield GraphSample(
                GraphWindow(window=t.isoformat(), nodes=nodes, events=list(history)),
                torch.tensor(targets, dtype=torch.float32),
            )


def _graphs_from_csv(model, nodes_csv_path, edges_csv_path=None, time_col="window_utc"):
    """Compatibility wrapper for existing callers; training uses CSVGraphDataset."""
    dataset = CSVGraphDataset(nodes_csv_path, edges_csv_path, time_col)
    if model.feature_mu is None:
        model.feature_mu, model.feature_sd = dataset.feature_scaler(
            dataset.timestamps[:-1]
        )
    return [
        (*model.build_graph_from_payload(s.graph), s.targets)
        for s in dataset.iter_samples()
    ]
