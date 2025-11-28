from typing import Optional
from pydantic import BaseModel, Field

class TrainCSVRequest(BaseModel):
    nodes_csv_path: str = Field(..., description="Путь к nodes.csv (обязателен)")
    edges_csv_path: Optional[str] = Field(None, description="Путь к edges.csv (опционально)")
    csv_time_column: str = Field("window_utc", description="Имя колонки времени в CSV")
    epochs: int = Field(20, ge=1, le=500, description="Число эпох обучения")
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    shuffle: bool = True
    seed: int = 42
    device: Optional[str] = Field(None, description='"cuda", "cpu" или None для авто')