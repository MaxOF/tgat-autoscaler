import torch
import math
import torch.nn as nn
from typing import List

class TemporalFourierEncoding(nn.Module):
    def __init__(self, periods_min: List[float]):
        super().__init__()
        self.periods = torch.tensor(periods_min, dtype=torch.float32)
        self.out_dim = 2 * len(periods_min)
    def forward(self, delta_minutes: torch.Tensor) -> torch.Tensor:
        if delta_minutes.ndim == 1:
            delta_minutes = delta_minutes.unsqueeze(1)
        arg = 2 * math.pi * delta_minutes / self.periods.to(delta_minutes.device)
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)