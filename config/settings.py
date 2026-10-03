from __future__ import annotations

from typing import Any, Dict, List

# =============================================================================
# GLOBAL APPLICATION CONFIGURATION
# =============================================================================

CONFIG: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Monitoring / control loop
    # -------------------------------------------------------------------------
    "prometheus_url": "http://localhost:9090",
    # Autoscaling decision interval.
    # The manuscript uses a 5-minute control interval.
    "metrics_interval": 5 * 60,
    # Offline retraining interval.
    "training_interval": 24 * 60 * 60,
    # Legacy path retained for backward compatibility.
    "data_file": "./out/nodes.csv",
    # -------------------------------------------------------------------------
    # Artifacts / datasets
    # -------------------------------------------------------------------------
    "model_file": "./artifacts/tgat_model.pt",
    "nodes_csv_path": "./data/nodes.csv",
    "edges_csv_path": "./data/edges.csv",
    # -------------------------------------------------------------------------
    # Temporal graph configuration
    # -------------------------------------------------------------------------
    # Default observation/history window W.
    "observation_window_minutes": 60,
    # Time range used for edge telemetry aggregation.
    "edge_feature_window_sec": 60 * 60,
    # Maximum allowed lag of an interaction event.
    "edge_lag_max_minutes": 60,
    # Maximum number of temporal neighbors sampled per node.
    "max_temporal_neighbors": 32,
    # Forecasting horizon Delta.
    "forecast_horizon_minutes": 5,
    # -------------------------------------------------------------------------
    # Edge construction
    # -------------------------------------------------------------------------
    "default_edge_weight": 1.0,
    "edge_defaults": {
        # Optional explicit topology priors.
        "orders-service": {
            "payments-service": 0.70,
            "products-service": 0.60,
            "orders-service": 0.80,
        },
        # Used by heuristic normalization in graph.py.
        "default_dst_rps": 1.0,
        "default_dst_p95_ms": 250.0,
        "edge_norm_mode": "node",
        "rps_per_core": 30.0,
    },
    # Parameters retained for edge_weight_from_nodes() in graph.py.
    "rps_norm_src": 120.0,
    "rps_norm_dst": 120.0,
    "p95_ref_ms": 250.0,
    "ew_alpha": 0.40,
    "ew_beta": 0.20,
    "ew_gamma": 0.20,
    "ew_delta": 0.20,
    # -------------------------------------------------------------------------
    # Hidden-edge synthesis
    # -------------------------------------------------------------------------
    "hidden_edge": {
        "enabled": True,
        # Maximum reconstructed incoming edges for one target node.
        "top_k": 3,
        # p_ji >= threshold -> candidate dependency is accepted.
        "threshold": 0.70,
        # Candidate pre-filtering.
        "candidate_similarity_threshold": 0.25,
        # Components of auxiliary similarity vector psi_ji.
        "gamma_rps": 0.45,
        "gamma_latency": 0.35,
        "gamma_arch": 0.20,
        # Conservative upper bound for scalability.
        "max_candidates_total": 128,
    },
    # -------------------------------------------------------------------------
    # Multi-objective autoscaling criterion
    #
    # J =
    # lambda_slo       * J_slo
    # + lambda_cost    * J_cost
    # + lambda_stability * J_stability
    # + lambda_risk    * J_risk
    # -------------------------------------------------------------------------
    "objective": {
        "lambda_slo": 0.35,
        "lambda_cost": 0.25,
        "lambda_stability": 0.20,
        "lambda_risk": 0.20,
    },
    # -------------------------------------------------------------------------
    # Resource-cost normalization
    # -------------------------------------------------------------------------
    "resource_cost": {
        "replica_weight": 1.0 / 3.0,
        "cpu_weight": 1.0 / 3.0,
        "memory_weight": 1.0 / 3.0,
    },
    # -------------------------------------------------------------------------
    # Deterministic safety layer
    # -------------------------------------------------------------------------
    "safety": {
        # Require this number of consistent windows before a small action
        # is allowed to pass.
        "hysteresis_windows": 2,
        # Maximum horizontal scaling step per decision.
        "rate_limit_replicas": 2,
        # Maximum vertical resource change per control step.
        "cpu_step_pct": 0.20,
        "mem_step_pct": 0.20,
        # Minimum interval between ordinary scaling actions.
        "cooldown_sec": 10 * 60,
        # Exponential smoothing:
        #
        # smoothed = alpha * predicted + (1-alpha) * previous
        "smoothing_alpha": 0.60,
        # Global replica limits. Service-specific limits may override them.
        "r_min": 1,
        "r_max": 50,
        # Risk above this value may bypass ordinary cooldown handling.
        "critical_slo_risk": 0.90,
        "dry_run": True,
    },
    # -------------------------------------------------------------------------
    # SLO configuration
    # -------------------------------------------------------------------------
    "slo": {"p95_ms": 400.0, "p99_ms": 600.0, "error_rate": 0.01},
    # -------------------------------------------------------------------------
    # Services
    #
    # This is the current runnable three-service demo configuration.
    # The experimental 12-service topology should later be placed here
    # or loaded from an external experiment configuration.
    # -------------------------------------------------------------------------
    "services": {
        "orders-service": {
            "dependencies": ["products-service", "payments-service"],
            "min_replicas": 1,
            "max_replicas": 20,
            "min_cpu": "400m",
            "max_cpu": "2000m",
            "min_memory": "512Mi",
            "max_memory": "3072Mi",
        },
        "payments-service": {
            "dependencies": [],
            "min_replicas": 1,
            "max_replicas": 20,
            "min_cpu": "400m",
            "max_cpu": "1200m",
            "min_memory": "512Mi",
            "max_memory": "2048Mi",
        },
        "products-service": {
            "dependencies": [],
            "min_replicas": 1,
            "max_replicas": 20,
            "min_cpu": "400m",
            "max_cpu": "1200m",
            "min_memory": "512Mi",
            "max_memory": "2560Mi",
        },
    },
}


# =============================================================================
# PYTORCH GEOMETRIC
# =============================================================================

# Kept for compatibility with the existing project. model.py also performs
# its own guarded PyG import.
PYG_AVAILABLE = True


# =============================================================================
# PATHS
# =============================================================================

MODEL_PATH = CONFIG.get("model_file", "./artifacts/tgat_model.pt")

NODES_CSV_PATH = CONFIG.get("nodes_csv_path", "./data/nodes.csv")

EDGES_CSV_PATH = CONFIG.get("edges_csv_path", "./data/edges.csv")


# =============================================================================
# NODE FEATURES
#
# x_i(t)
#
# The first six fields preserve compatibility with the current graph.py.
# p99_ms and replicas extend the representation used in the manuscript.
# graph.py will be updated next to collect/populate both values explicitly.
# =============================================================================

FEATURE_ORDER: List[str] = [
    "cpu_mcores",
    "mem_mib",
    "rps_in",
    "rps_out",
    "p95_ms",
    "p99_ms",
    "error_rate",
    "replicas",
]


# =============================================================================
# PREDICTION TARGETS
#
# y_hat_i(t + Delta) =
# [
#   lambda_hat,
#   r_hat,
#   c_hat,
#   m_hat,
#   q95_hat,
#   q99_hat,
#   rho_hat
# ]
# =============================================================================

TARGET_ORDER: List[str] = [
    "rps_next",
    "replicas_next",
    "cpu_mcores_next",
    "mem_mib_next",
    "p95_ms_next",
    "p99_ms_next",
    "slo_risk_next",
]

MODEL_OUTPUT_DIM = len(TARGET_ORDER)


# =============================================================================
# RAW EDGE FEATURES
#
# Fourier temporal encoding is appended dynamically in model.py.
# =============================================================================

EDGE_FEATURE_ORDER: List[str] = [
    "edge_weight",
    "edge_p95_ms",
    "edge_errors",
    "edge_confidence",
]

RAW_EDGE_FEATURE_DIM = len(EDGE_FEATURE_ORDER)


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

EPOCHS = 100

DEFAULT_LEARNING_RATE = 1e-3

DEFAULT_WEIGHT_DECAY = 1e-4


# =============================================================================
# TGAT MODEL CONFIGURATION
# =============================================================================

DEFAULT_MODEL_CFG: Dict[str, Any] = {
    # Fourier periods in minutes:
    # 1 h, 6 h, 1 day, 1 week.
    "fourier_periods_min": [60, 6 * 60, 24 * 60, 7 * 24 * 60],
    "time_encoding": True,
    # Edge dropout used in experimental ablations.
    "dropedge_prob": 0.0,
    # TGAT architecture.
    "d_model": 128,
    "heads": 4,
    "layers": 2,
    "dropout": 0.10,
    # Reviewer-driven switches.
    "use_graph": True,
    "use_hidden_edges": True,
    # Old 3-output checkpoints must not silently be interpreted
    # as the new seven-target architecture.
    "required_output_dim": MODEL_OUTPUT_DIM,
    # Prevent accidental scientific evaluation using randomly
    # initialized weights when the checkpoint is absent/incompatible.
    "allow_untrained_inference": False,
}
