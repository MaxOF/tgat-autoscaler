from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import os

from app.core.temporal_fourier_encoding import TemporalFourierEncoding
from app.core.interfaces import GraphWindow, NodeFeatures, Action
from app.utils.parsers import parse_cpu_milli, parse_iso8601, parse_mem_mib, format_cpu_milli, format_mem_gi_from_mib
from config.settings import CONFIG, MODEL_PATH, PYG_AVAILABLE


class SimpleTGAT(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, d_model: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.layers = layers
        self.expected_edge_dim = int(edge_dim)

        # Всегда держим MLP-фолбэк (на случай отсутствия рёбер)
        self.mlp_fallback = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

        # GAT-ветка — только если PyG доступен
        self.gnn_enabled = PYG_AVAILABLE
        if self.gnn_enabled:
            self.input = nn.Linear(in_dim, d_model)
            self.convs = nn.ModuleList([
                GATv2Conv(d_model, d_model // heads, heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True)
                for _ in range(layers)
            ])
            self.out_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3)  # [replicas_float, cpu_milli, mem_mib]
        )

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor], edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        use_gnn = (
            self.gnn_enabled
            and edge_index is not None
            and edge_attr is not None
            and edge_attr.size(1) == self.expected_edge_dim
            and edge_index.numel() > 0
        )
        if use_gnn:
            h = self.input(x)
            for conv in self.convs:
                h = conv(h, edge_index, edge_attr)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.out_norm(h)
        else:
            h = self.mlp_fallback(x)

        return self.head(h)


class TGATAutoscalerModel:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        periods_min = cfg.get('fourier_periods_min', [60, 6*60, 24*60, 7*24*60])
        self.temporal_encoder = TemporalFourierEncoding(periods_min)
        self.time_encoding_enabled = cfg.get('time_encoding', True)
        self.dropedge_prob = cfg.get('dropedge_prob', 0.0)
        self.model: Optional[SimpleTGAT] = None
        self.scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.id2idx: Dict[str, int] = {}
        self.idx2id: List[str] = []
        self.ckpt_in_dim: Optional[int] = None
        self.ckpt_edge_dim: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_scale = False

    def _fit_scalers(self, nodes: List[NodeFeatures]):
        for n in nodes:
            x = np.asarray(n.x, dtype=np.float32)
            mu = x.mean() if x.size else 0.0
            sd = x.std() + 1e-6
            self.scalers[n.id] = (mu, sd)

    def _apply_scaler(self, node_id: str, x: np.ndarray) -> np.ndarray:
        mu, sd = self.scalers.get(node_id, (0.0, 1.0))
        return (x - mu) / sd

    def build_graph_from_payload(self, gw: GraphWindow) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.id2idx = {n.id: i for i, n in enumerate(gw.nodes)}
        self.idx2id = [n.id for n in gw.nodes]
        self._fit_scalers(gw.nodes)

        X = np.stack([self._apply_scaler(n.id, np.asarray(n.x, dtype=np.float32)) for n in gw.nodes], axis=0)
        if hasattr(self, "global_scaler"):
            mu, sd = self.global_scaler
            X = (X - mu) / (sd + 1e-6)
        x = torch.tensor(X, dtype=torch.float32)
        if len(gw.events) == 0:
            return x, None, None
        t_ref = parse_iso8601(gw.window)
        src_idx, dst_idx, edge_feats, deltas = [], [], [], []
        for ev in gw.events:
            if ev.src not in self.id2idx or ev.dst not in self.id2idx:
                continue
            src_idx.append(self.id2idx[ev.src])
            dst_idx.append(self.id2idx[ev.dst])
            edge_feats.append(np.asarray(ev.e, dtype=np.float32))
            tau = parse_iso8601(ev.tau)
            delta_min = max(0.0, (t_ref - tau).total_seconds() / 60.0)
            deltas.append(delta_min)
        if not src_idx:
            return x, None, None
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_attr = torch.tensor(np.stack(edge_feats, axis=0), dtype=torch.float32)
        delta_t = torch.tensor(deltas, dtype=torch.float32)
        if self.time_encoding_enabled:
            phi = self.temporal_encoder(delta_t)
            edge_attr = torch.cat([edge_attr, phi], dim=1)
        return x, edge_index, edge_attr

    def init_model(self, in_dim: int, edge_dim: int):
        self.model = SimpleTGAT(
            in_dim=in_dim, edge_dim=edge_dim,
            d_model=self.cfg.get('d_model', 128),
            heads=self.cfg.get('heads', 4),
            layers=self.cfg.get('layers', 2),
            dropout=self.cfg.get('dropout', 0.1),
        ).to(self.device)
    
    def _align_edge_attr(self, edge_attr: Optional[torch.Tensor], target_dim: int, num_edges: int) -> Optional[torch.Tensor]:
        # Нет рёбер — возвращаем None (модель уйдёт в MLP-фолбэк)
        if target_dim == 0:
            return None
        if edge_attr is None:
            return torch.zeros((num_edges, target_dim), dtype=torch.float32, device=self.device)
        cur = edge_attr.size(1)
        if cur == target_dim:
            return edge_attr.to(self.device)
        if cur > target_dim:
            return edge_attr[:, :target_dim].to(self.device)
        # cur < target_dim — дополним нулями
        pad = torch.zeros((edge_attr.size(0), target_dim - cur), dtype=edge_attr.dtype, device=edge_attr.device)
        return torch.cat([edge_attr, pad], dim=1).to(self.device)
    
    def _postprocess_actions(self, raw_out: np.ndarray) -> List[Action]:
        actions: List[Action] = []
        for i, node_id in enumerate(self.idx2id):
            r_f, cpu_m, mem_mib = raw_out[i]
          
            replicas = int(max(1, round(r_f)))
            # Клиппинг ресурсов по CONFIG.services
            svc_cfg = CONFIG['services'].get(node_id, {})
            min_cpu = parse_cpu_milli(svc_cfg.get('min_cpu', '100m'))
            max_cpu = parse_cpu_milli(svc_cfg.get('max_cpu', '4000m'))
            min_mem = parse_mem_mib(svc_cfg.get('min_memory', '256Mi'))
            max_mem = parse_mem_mib(svc_cfg.get('max_memory', '8192Mi'))
            cpu_m = float(np.clip(cpu_m, min_cpu, max_cpu))
            mem_mib = float(np.clip(mem_mib, min_mem, max_mem))
            cpu_str = format_cpu_milli(cpu_m)
            mem_str = format_mem_gi_from_mib(mem_mib)
    
            actions.append(Action(id=node_id, replicas=replicas, cpu=cpu_str, mem=mem_str))
        return actions

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location="cpu")
            self.ckpt_in_dim = int(ckpt.get('in_dim', 0))
            self.ckpt_edge_dim = int(ckpt.get('edge_dim', 0))
            # если в чекпойнте нет dim — считаем 0, но лучше переобучить/пересохранить
            if self.ckpt_in_dim is None or self.ckpt_in_dim <= 0:
                return False
            if self.ckpt_edge_dim is None or self.ckpt_edge_dim < 0:
                self.ckpt_edge_dim = 0
            # инициализируем модель под сохранённые размерности
            self.init_model(self.ckpt_in_dim, self.ckpt_edge_dim)
            self.model.load_state_dict(ckpt['model'], strict=False)
            self.model.eval()
            self.global_scale = True
            return True
        except Exception as e:
            print(f"[predict] failed to load checkpoint: {e}")
            return False

    def ensure_ready(self, in_dim_now: int, edge_dim_now: int, model_path: str):
        """
        Гарантируем, что в self.model загружены веса, и размерности согласованы.
        """
        # если уже загружали чекпойнт ранее — модель есть
        if self.model is not None and (self.ckpt_in_dim is not None):
            return

        # попытка загрузить с диска
        if self.load_checkpoint(model_path):
            return

        # чекпойнта нет — инициализируем «с нуля» по текущим размерностям
        self.ckpt_in_dim = in_dim_now
        self.ckpt_edge_dim = edge_dim_now
        self.init_model(in_dim_now, edge_dim_now)
        self.model.eval()

    def predict_graph(self, x: torch.Tensor, edge_index: Optional[torch.Tensor], edge_attr: Optional[torch.Tensor]) -> List[Action]:
        # Текущие размерности графа
        in_dim_now = x.shape[1]
        edge_dim_now = edge_attr.shape[1] if edge_attr is not None else 0

        # Готовим/подгружаем модель
        self.ensure_ready(in_dim_now, edge_dim_now, MODEL_PATH)

        # Приводим edge_attr к ожидаемой чекпойнтом размерности
        tgt_edge_dim = int(self.ckpt_edge_dim or edge_dim_now or 0)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)
        x = x.to(self.device)
        edge_attr = self._align_edge_attr(edge_attr, tgt_edge_dim, edge_index.size(1) if edge_index is not None else 0)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x, edge_index, edge_attr)  # [N,3]
  
        return self._postprocess_actions(out.cpu().numpy())
