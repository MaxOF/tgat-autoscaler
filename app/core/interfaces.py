from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# GRAPH DTOs
# =============================================================================


class NodeFeatures(BaseModel):
    """
    Node-level telemetry representation.

    x follows config.settings.FEATURE_ORDER whenever the complete
    experimental representation is used.
    """

    id: str

    x: List[float]

    meta: Optional[Dict[str, Any]] = None


class EdgeEvent(BaseModel):
    """
    Time-stamped directed interaction event.

    e contains the raw operational edge features.
    Temporal Fourier encoding is appended later in model.py.
    """

    src: str

    dst: str

    # ISO-8601 timestamp, e.g. 2026-09-28T12:30:00Z
    tau: str

    e: List[float] = Field(default_factory=list)

    # True for directly measured dependencies.
    # False for reconstructed hidden edges.
    observed: bool = True

    # Confidence assigned to the edge.
    # Observed dependencies normally have confidence 1.
    confidence: float = 1.0

    meta: Optional[Dict[str, Any]] = None


class GraphWindow(BaseModel):
    """
    Temporal graph observed up to `window`.
    """

    # ISO-8601 right boundary of the observation window.
    window: str

    nodes: List[NodeFeatures]

    events: List[EdgeEvent] = Field(default_factory=list)

    # Number of control intervals into the future.
    horizon: int = 1


# =============================================================================
# MODEL OUTPUT
# =============================================================================


class Prediction(BaseModel):
    """
    Seven-target TGAT prediction for one service.

    Corresponds to

    y_hat_i(t + Delta) =
    [
        lambda_hat_i,
        r_hat_i,
        c_hat_i,
        m_hat_i,
        q95_hat_i,
        q99_hat_i,
        rho_hat_i
    ]
    """

    id: str

    # Predicted request intensity.
    rps: float

    # Predicted horizontal demand.
    replicas: float

    # Predicted vertical CPU demand.
    cpu_milli: float

    # Predicted vertical memory demand.
    mem_mib: float

    # Predicted latency quantiles.
    p95_ms: float

    p99_ms: float

    # Probability/risk of SLO violation, [0, 1].
    slo_risk: float


class HiddenEdgePrediction(BaseModel):
    """
    Diagnostic record for one synthesized dependency.
    """

    src: str

    dst: str

    probability: float

    rps_similarity: float = 0.0

    latency_similarity: float = 0.0

    architectural_prior: float = 0.0


# =============================================================================
# AUTOSCALING ACTION
# =============================================================================


class Action(BaseModel):
    id: str

    replicas: int = Field(ge=1)

    # Kubernetes resource strings, e.g. "800m".
    cpu: Optional[str] = None

    # e.g. "1.50Gi".
    mem: Optional[str] = None


# =============================================================================
# SAFETY / DECISION CONFIGURATION
# =============================================================================


class PolicyConfig(BaseModel):
    # -------------------------------------------------------------------------
    # Hysteresis / rate limiting
    # -------------------------------------------------------------------------
    hysteresis_windows: int = Field(2, ge=0)

    rate_limit_replicas: int = Field(2, ge=0)

    cpu_step_pct: float = Field(0.20, ge=0, le=1)

    mem_step_pct: float = Field(0.20, ge=0, le=1)

    # -------------------------------------------------------------------------
    # Cooldown
    # -------------------------------------------------------------------------
    cooldown_sec: int = Field(10 * 60, ge=0)

    # -------------------------------------------------------------------------
    # Exponential smoothing
    # -------------------------------------------------------------------------
    smoothing_alpha: float = Field(0.60, ge=0, le=1)

    # -------------------------------------------------------------------------
    # Replica bounds
    # -------------------------------------------------------------------------
    r_min: int = Field(1, ge=1)

    r_max: int = Field(50, ge=1)

    # -------------------------------------------------------------------------
    # Multi-objective criterion
    # -------------------------------------------------------------------------
    lambda_slo: float = Field(0.35, ge=0, allow_inf_nan=False)

    lambda_cost: float = Field(0.25, ge=0, allow_inf_nan=False)

    lambda_stability: float = Field(0.20, ge=0, allow_inf_nan=False)

    lambda_risk: float = Field(0.20, ge=0, allow_inf_nan=False)

    # -------------------------------------------------------------------------
    # Critical risk may later be used to bypass ordinary cooldown.
    # -------------------------------------------------------------------------
    critical_slo_risk: float = Field(0.90, ge=0, le=1)

    dry_run: bool = True

    @model_validator(mode="after")
    def validate_limits(self):
        if self.r_max < self.r_min:
            raise ValueError("r_max must be at least r_min")
        weights = [
            self.lambda_slo,
            self.lambda_cost,
            self.lambda_stability,
            self.lambda_risk,
        ]
        if sum(weights) <= 0:
            raise ValueError("At least one objective weight must be positive")
        total = sum(weights)
        self.lambda_slo /= total
        self.lambda_cost /= total
        self.lambda_stability /= total
        self.lambda_risk /= total
        return self


# =============================================================================
# TRAINING
# =============================================================================


class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None

    config: Optional[Dict[str, Any]] = None


# =============================================================================
# API REQUESTS
# =============================================================================


class PredictRequest(GraphWindow):
    pass


class ApplyRequest(BaseModel):
    """
    Supports either explicitly supplied actions or a temporal graph
    from which the actions are obtained.
    """

    actions: Optional[List[Action]] = None

    graph: Optional[GraphWindow] = None

    policy: Optional[PolicyConfig] = None


# =============================================================================
# ABLATION
# =============================================================================


class AblateRequest(BaseModel):
    """
    Experimental switches required by the reviewers.

    Examples
    --------
    Full:
        graph_enabled=True
        time_encoding=True
        hidden_edges=True
        safety_enabled=True

    noGraph:
        graph_enabled=False

    w/o temporal encoding:
        time_encoding=False

    w/o hidden-edge synthesis:
        hidden_edges=False
    """

    time_encoding: Optional[bool] = None

    hidden_edges: Optional[bool] = None

    graph_enabled: Optional[bool] = None

    safety_enabled: Optional[bool] = None

    # Retained for the existing implementation / DropEdge experiments.
    dropedge_prob: Optional[float] = None

    # Retained for backward compatibility with the old endpoint.
    hysteresis_enabled: Optional[bool] = None

    # Allows the reduced-history-window experiment.
    history_minutes: Optional[int] = None


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================


class SensitivityRequest(BaseModel):
    """
    Parameter sweep configuration used in reviewer-requested
    sensitivity experiments.
    """

    history_minutes: Optional[int] = None

    hidden_edge_threshold: Optional[float] = None

    beta: Optional[float] = None

    lambda_slo: Optional[float] = None

    lambda_cost: Optional[float] = None

    lambda_stability: Optional[float] = None

    lambda_risk: Optional[float] = None


# =============================================================================
# EVALUATION DTOs
# =============================================================================


class RegressionMetrics(BaseModel):
    mae: float

    rmse: float


class ClassificationMetrics(BaseModel):
    precision: float

    recall: float

    f1: float

    auroc: Optional[float] = None

    brier_score: Optional[float] = None


class HiddenEdgeMetrics(BaseModel):
    true_positive: int

    false_positive: int

    false_negative: int

    precision: float

    recall: float

    f1: float
