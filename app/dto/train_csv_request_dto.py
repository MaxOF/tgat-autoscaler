from typing import Optional
from pydantic import BaseModel, Field


class TrainCSVRequest(BaseModel):
    nodes_csv_path: str = Field(..., description="Path to node telemetry CSV")
    edges_csv_path: Optional[str] = Field(None, description="Observed edges only")
    csv_time_column: str = "window_utc"
    epochs: int = Field(20, ge=1, le=500)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    shuffle: bool = True
    seed: int = 42
    device: Optional[str] = None
    validation_fraction: float = Field(0.2, gt=0, lt=0.5)
    model_path: Optional[str] = None
