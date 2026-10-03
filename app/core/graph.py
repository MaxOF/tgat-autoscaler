from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from app.core.interfaces import EdgeEvent, GraphWindow, NodeFeatures
from app.core.prometheus import PromClient
from app.utils.parsers import now_utc_iso, parse_cpu_milli, parse_mem_mib
from config.settings import CONFIG, FEATURE_ORDER, TARGET_ORDER

# =============================================================================
# PROMETHEUS GRAPH CONSTRUCTION
# =============================================================================


def build_graph_from_prometheus(config: Dict[str, Any]) -> GraphWindow:
    """
    Build one current temporal graph from live Prometheus telemetry.

    Node feature order follows FEATURE_ORDER:
        cpu_mcores
        mem_mib
        rps_in
        rps_out
        p95_ms
        p99_ms
        error_rate
        replicas
    """

    prom = PromClient(config["prometheus_url"])

    services = list(config["services"].keys())

    nodes: List[NodeFeatures] = []

    for service in services:
        features: Dict[str, float] = {name: 0.0 for name in FEATURE_ORDER}

        cpu = _query_service_metric(prom, "cpu_mcores", service)
        memory = _query_service_metric(prom, "mem_mib", service)
        incoming = _query_service_metric(prom, "rps_in", service)
        if cpu is None or memory is None or incoming is None:
            raise ValueError(f"Required telemetry is unavailable for {service}")
        features["cpu_mcores"] = cpu

        features["mem_mib"] = memory

        features["rps_in"] = incoming

        features["rps_out"] = _try_service_queries(prom, "rps_out", service)

        features["p95_ms"] = _try_service_queries(prom, "p95_ms", service)

        features["p99_ms"] = _try_p99_query(prom, service)

        error_rate = _try_service_queries(prom, "error_rate", service)

        if not np.isfinite(error_rate):
            error_rate = 0.0

        features["error_rate"] = float(np.clip(error_rate, 0.0, 1.0))

        features["replicas"] = _try_replicas_query(prom, service)

        # Conservative fallbacks for missing Prometheus metrics.
        if features["p99_ms"] <= 0.0 and features["p95_ms"] > 0.0:
            features["p99_ms"] = features["p95_ms"] * 1.35

        if features["replicas"] <= 0.0:
            features["replicas"] = float(
                config["services"][service].get("min_replicas", 1)
            )

        x = [float(features[name]) for name in FEATURE_ORDER]

        meta = {"features": features}
        cpu_request = _try_service_queries(prom, "cpu_request_m", service)
        memory_request = _try_service_queries(prom, "mem_request_mib", service)
        if cpu_request > 0 and memory_request > 0:
            meta.update(cpu_request_m=cpu_request, mem_request_mib=memory_request)
        nodes.append(NodeFeatures(id=service, x=x, meta=meta))

    window_iso = now_utc_iso()

    # Events are given a timestamp slightly before the decision instant.
    tau = datetime.now(timezone.utc) - timedelta(
        seconds=min(60, int(config.get("metrics_interval", 300)))
    )

    tau_iso = tau.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    node_stats = {
        node.id: (node.meta.get("features", {}) if node.meta else {}) for node in nodes
    }

    events: List[EdgeEvent] = []

    for src, service_cfg in config["services"].items():
        for dst in service_cfg.get("dependencies", []):
            measured = _try_edge_query_optional(prom, "edge_rps", src, dst)
            if measured is None:
                continue
            edge_weight, edge_p95, edge_errors = _collect_edge_features(
                prom=prom, src=src, dst=dst, node_stats=node_stats, config=config
            )

            events.append(
                EdgeEvent(
                    src=src,
                    dst=dst,
                    tau=tau_iso,
                    e=[float(edge_weight), float(edge_p95), float(edge_errors)],
                    observed=True,
                    confidence=1.0,
                    meta={
                        "edge_weight": float(edge_weight),
                        "edge_p95_ms": float(edge_p95),
                        "edge_errors": float(edge_errors),
                    },
                )
            )

    return GraphWindow(window=window_iso, nodes=nodes, events=events, horizon=1)


# =============================================================================
# PROMETHEUS HELPERS
# =============================================================================


def _first_value_or_zero(result_json: Dict[str, Any]) -> float:
    try:
        data = result_json.get("data", {}).get("result", [])

        if not data:
            return 0.0

        value = float(data[0]["value"][1])

        if math.isnan(value) or math.isinf(value):
            return 0.0

        return value

    except Exception:
        return 0.0


def _query_service_metric(prom: PromClient, key: str, service: str) -> Optional[float]:
    for query in prom.candidate_queries.get(key, []):
        for label in prom.serivce_label_keys:
            prepared = query.replace("$SVC", service).replace("$LBL", label)
            try:
                rows = prom.query(prepared).get("data", {}).get("result", [])
                if rows:
                    value = float(rows[0]["value"][1])
                    if np.isfinite(value):
                        return value
            except Exception:
                continue
    return None


def _try_service_queries(prom: PromClient, key: str, service: str) -> float:
    value = _query_service_metric(prom, key, service)
    return value if value is not None else 0.0


def _try_p99_query(prom: PromClient, service: str) -> float:
    """
    prometheus.py currently contains only the p95 histogram query.

    Reuse the same metric expression with quantile 0.99.
    """

    p95_queries = prom.candidate_queries.get("p95_ms", [])

    for query in p95_queries:
        p99_query = query.replace("histogram_quantile(0.95", "histogram_quantile(0.99")

        for label in prom.serivce_label_keys:
            prepared = p99_query.replace("$SVC", service).replace("$LBL", label)

            try:
                result = prom.query(prepared)

                value = _first_value_or_zero(result)

                if value != 0.0:
                    return value

            except Exception:
                continue

    return 0.0


def _try_replicas_query(prom: PromClient, service: str) -> float:
    candidate_queries = [
        f'kube_deployment_spec_replicas{{deployment="{service}"}}',
        ("kube_deployment_status_" "replicas_available" f'{{deployment="{service}"}}'),
        ("kube_deployment_status_" "replicas" f'{{deployment="{service}"}}'),
    ]

    for query in candidate_queries:
        try:
            value = _first_value_or_zero(prom.query(query))

            if value > 0.0:
                return value

        except Exception:
            continue

    return 0.0


def _try_edge_query_optional(prom, key, src, dst):
    for query in prom.candidate_queries.get(key, []):
        prepared = query.replace("$SRC", src).replace("$DST", dst)
        try:
            records = prom.query(prepared).get("data", {}).get("result", [])
            if records:
                value = float(records[0]["value"][1])
                if np.isfinite(value):
                    return value
        except Exception:
            continue
    return None


def _try_edge_query(prom: PromClient, key: str, src: str, dst: str) -> float:
    queries = prom.candidate_queries.get(key, [])

    for query in queries:
        prepared = query.replace("$SRC", src).replace("$DST", dst)

        try:
            result = prom.query(prepared)

            value = _first_value_or_zero(result)

            if value != 0.0:
                return value

        except Exception:
            continue

    return 0.0


def _collect_edge_features(
    prom: PromClient,
    src: str,
    dst: str,
    node_stats: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Tuple[float, float, float]:
    edge_rps = _try_edge_query(prom, "edge_rps", src, dst)

    edge_p95 = _try_edge_query(prom, "edge_p95_ms", src, dst)

    edge_errors = _try_edge_query(prom, "edge_errors", src, dst)

    if edge_rps > 0.0:
        rps_ref = max(1.0, float(config.get("rps_norm_src", 120.0)))

        edge_weight = float(np.clip(edge_rps / rps_ref, 0.0, 1.0))

    else:
        edge_weight = edge_weight_from_nodes(src, dst, node_stats, config)

    if edge_p95 <= 0.0:
        edge_p95 = edge_weight * float(node_stats.get(dst, {}).get("p95_ms", 0.0))

    if edge_errors <= 0.0:
        edge_errors = edge_weight * float(
            node_stats.get(dst, {}).get("error_rate", 0.0)
        )

    return (float(edge_weight), float(edge_p95), float(edge_errors))


# =============================================================================
# EDGE WEIGHT
# =============================================================================


def edge_weight_from_nodes(
    src: str, dst: str, node_stats: Dict[str, Dict[str, float]], cfg: Dict[str, Any]
) -> float:
    dependencies = CONFIG.get("services", {}).get(src, {}).get("dependencies", [])

    out_degree = len(dependencies)

    edge_defaults = cfg.get("edge_defaults", {})

    base_weight = (
        edge_defaults.get(src, {}).get(dst)
        if isinstance(edge_defaults.get(src, {}), dict)
        else None
    )

    if base_weight is None:
        base_weight = (
            1.0 / out_degree
            if out_degree > 0
            else float(cfg.get("default_edge_weight", 0.5))
        )

    src_features = node_stats.get(src, {})

    dst_features = node_stats.get(dst, {})

    rps_norm_src = max(1e-6, float(cfg.get("rps_norm_src", 120.0)))

    rps_norm_dst = max(1e-6, float(cfg.get("rps_norm_dst", 120.0)))

    p95_ref_ms = max(1e-6, float(cfg.get("p95_ref_ms", 250.0)))

    source_pressure = float(
        np.clip(float(src_features.get("rps_out", 0.0)) / rps_norm_src, 0.0, 1.0)
    )

    destination_flow = float(
        np.clip(float(dst_features.get("rps_in", 0.0)) / rps_norm_dst, 0.0, 1.0)
    )

    p95 = float(dst_features.get("p95_ms", 0.0))

    latency_factor = p95 / (p95 + p95_ref_ms) if p95 > 0.0 else 0.0

    reliability = 1.0 - float(np.clip(dst_features.get("error_rate", 0.0), 0.0, 1.0))

    dst_cfg = cfg.get("services", {}).get(dst, {})

    max_cpu = max(1.0, parse_cpu_milli(dst_cfg.get("max_cpu", "2000m")))

    max_mem = max(1.0, parse_mem_mib(dst_cfg.get("max_memory", "2048Mi")))

    cpu_util = float(
        np.clip(float(dst_features.get("cpu_mcores", 0.0)) / max_cpu, 0.0, 1.0)
    )

    memory_util = float(
        np.clip(float(dst_features.get("mem_mib", 0.0)) / max_mem, 0.0, 1.0)
    )

    destination_load = max(cpu_util, memory_util)

    alpha = float(cfg.get("ew_alpha", 0.4))

    beta = float(cfg.get("ew_beta", 0.2))

    gamma = float(cfg.get("ew_gamma", 0.2))

    delta = float(cfg.get("ew_delta", 0.2))

    weight = (
        alpha * float(base_weight)
        + beta * source_pressure
        + gamma * max(destination_flow, destination_load)
        + delta * latency_factor
    )

    weight = float(np.clip(weight, 0.0, 1.0))

    weight *= reliability

    return float(np.clip(weight, 0.0, 1.0))


# Compatibility alias used by older imports.
_edge_weight_from_nodes = edge_weight_from_nodes


# =============================================================================
# CSV TRAINING DATASET
# =============================================================================


from app.data.csv_dataset import _graphs_from_csv  # Backward-compatible import.


class PrometheusGraphProvider:
    """Retain causal live events across control cycles; restart requires warm-up."""

    def __init__(self, config):
        self.config = config
        self.events = []

    def __call__(self):
        from app.utils.parsers import parse_iso8601

        graph = build_graph_from_prometheus(self.config)
        now = parse_iso8601(graph.window)
        lower = now - timedelta(
            minutes=self.config.get("observation_window_minutes", 60)
        )
        self.events = [e for e in self.events if parse_iso8601(e.tau) >= lower]
        self.events.extend(graph.events)
        graph.events = list(self.events)
        return graph
