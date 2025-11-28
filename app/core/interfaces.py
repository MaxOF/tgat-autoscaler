from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class NodeFeatures(BaseModel):
    id: str
    x: List[float]
    meta: Optional[Dict[str, Any]] = None

class EdgeEvent(BaseModel):
    src: str
    dst: str
    tau: str  # ISO8601, напр. "2025-10-29T11:59:00Z"
    e: List[float] = Field(default_factory=list)

class GraphWindow(BaseModel):
    window: str  # ISO8601 конца окна t
    nodes: List[NodeFeatures]
    events: List[EdgeEvent] = Field(default_factory=list)
    horizon: int = 1  # кол-во окон вперёд

class Action(BaseModel):
    id: str
    replicas: int
    cpu: Optional[str] = None  # например, "800m"
    mem: Optional[str] = None  # например, "1.5Gi"

class PolicyConfig(BaseModel):
    hysteresis_windows: int = 2
    rate_limit_replicas: int = 2
    cpu_step_pct: float = 0.2
    mem_step_pct: float = 0.2
    r_min: int = 1
    r_max: int = 50
    dry_run: bool = True

class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class PredictRequest(GraphWindow):
    pass

class ApplyRequest(BaseModel):
    actions: Optional[List[Action]] = None
    graph: Optional[GraphWindow] = None
    policy: Optional[PolicyConfig] = None

class AblateRequest(BaseModel):
    time_encoding: Optional[bool] = None
    dropedge_prob: Optional[float] = None
    hysteresis_enabled: Optional[bool] = None