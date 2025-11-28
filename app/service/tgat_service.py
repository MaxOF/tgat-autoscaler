from __future__ import annotations
import os
import time
import torch
import torch.nn.functional as F
from fastapi import HTTPException

from app.core.graph import _graphs_from_csv
from app.core.model import TGATAutoscalerModel
from config.settings import DEFAULT_MODEL_CFG, CONFIG, MODEL_PATH
from app.dto.train_csv_request_dto import TrainCSVRequest
from app.core.interfaces import PolicyConfig
from app.core.safety import SafetyPolicy
from app.core.graph import build_graph_from_prometheus


class TGATService:
    def __init__(self):
        self.model = TGATAutoscalerModel(DEFAULT_MODEL_CFG)

    async def apply(self):
    
        gw = build_graph_from_prometheus(CONFIG)

        builded_graph = self.model.build_graph_from_payload(gw)
        actions = self.model.predict_graph(*builded_graph)
        tstamp = gw.window

        policy_cfg = PolicyConfig()
        policy = SafetyPolicy(policy_cfg)
        safe_actions = policy.filter(actions, tstamp)
        report = policy.apply_to_k8s(safe_actions)
  
        return {'window': tstamp, 'applied': [a.model_dump() for a in safe_actions], 'report': report}

    async def predict(self):
        gw = build_graph_from_prometheus(CONFIG)
    
        x, edge_index, edge_attr = self.model.build_graph_from_payload(gw)

        actions = self.model.predict_graph(x, edge_index, edge_attr)

        return {'window': gw.window, 'actions': [a.model_dump() for a in actions]}

    def ablate(self):
        policy_cfg = PolicyConfig()
        hysteresis_windows = policy_cfg.hysteresis_windows
        time_encoding_enabled = self.model.time_encoding_enabled
        dropedge_prob = self.model.dropedge_prob
      
        return {
            'time_encoding': time_encoding_enabled,
            'dropedge_prob': dropedge_prob,
            'hysteresis_windows': hysteresis_windows
        }


    async def train_from_csv(self, req: TrainCSVRequest):
        if MODEL_PATH is None:
            raise HTTPException(status_code=400, detail="MODEL_PATH не задан (None)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = time.time()

        # 1) Собираем датасет графов
        try:
            data = _graphs_from_csv(
                self.model,
                nodes_csv_path=req.nodes_csv_path,
                edges_csv_path=req.edges_csv_path,
                time_col=getattr(req, "time_column", getattr(req, "csv_time_column", "window_utc"))
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка парсинга CSV: {e}")

        if not data:
            raise HTTPException(status_code=400, detail="Датасет пустой после парсинга CSV")

        # 2) Определяем размерности
        try:
            sample_x, sample_edge_index, sample_edge_attr, sample_y = data[0]
            in_dim  = int(sample_x.shape[1])
            edge_dim = int(sample_edge_attr.shape[1]) if sample_edge_attr is not None else 0
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Невозможно определить in_dim/edge_dim: {e}")

        # 3) Инициализируем модель и берём внутреннюю нейросеть
        self.model.init_model(in_dim, edge_dim)
        torch_model = self.model.model.to(device)  # <-- ТУТ главное отличие
        torch_model.train()

        # 4) Оптимизатор по параметрам нейросети
        opt = torch.optim.AdamW(
            torch_model.parameters(),
            lr=req.learning_rate,
            weight_decay=req.weight_decay
        )

        # 5) Красивые логи
        print("=" * 80)
        print("[train_csv] TGAT-Autoscaler — CSV training")
        print(f"[train_csv] nodes_csv: {req.nodes_csv_path}")
        print(f"[train_csv] edges_csv: {req.edges_csv_path or '(none)'}")
        print(f"[train_csv] time_col: {getattr(req, 'time_column', getattr(req, 'csv_time_column', 'window_utc'))} | samples: {len(data)}")
        print(f"[train_csv] dims: in_dim={in_dim}, edge_dim={edge_dim}, out_dim=3")
        print(f"[train_csv] device: {device.type} | epochs: {req.epochs} | lr={req.learning_rate} | wd={req.weight_decay}")
        print("=" * 80)

        # 6) Тренировка
        best_loss = float("inf")
        last_epoch_loss = None

        for ep in range(1, req.epochs + 1):
            running = 0.0
            steps = 0
            t0 = time.time()

            for x, edge_index, edge_attr, y in data:
                x = x.to(device)
                y = y.to(device)
                if edge_index is not None:
                    edge_index = edge_index.to(device)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)

                opt.zero_grad(set_to_none=True)
                pred = torch_model(x, edge_index, edge_attr)  # <-- вызыаем НЕ self.model, а именно torch_model
                loss = F.smooth_l1_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
                opt.step()

                running += float(loss.item())
                steps += 1

            last_epoch_loss = running / max(1, steps)
            elapsed = time.time() - t0
            best_loss = min(best_loss, last_epoch_loss)
            print(f"[epoch {ep:03d}/{req.epochs}] loss={last_epoch_loss:.6f} (best={best_loss:.6f}) | steps={steps} | {elapsed:.2f}s")

        # 7) Сохранение (включая размерности — это важно для predict/ensure_ready)
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        torch.save({
            "model": torch_model.state_dict(),
            "cfg": self.model.cfg,
            "in_dim": in_dim,
            "edge_dim": edge_dim
        }, MODEL_PATH)

        total_time = time.time() - start
        print("-" * 80)
        print(f"[train_csv] DONE | saved to: {MODEL_PATH} | total_time: {total_time:.2f}s")
        print("-" * 80)

        return {
            "status": "ok",
            "model_path": MODEL_PATH,
            "samples": len(data),
            "epochs": req.epochs,
            "last_epoch_loss": last_epoch_loss,
            "best_loss": best_loss,
            "device": device.type,
            "in_dim": in_dim,
            "edge_dim": edge_dim,
            "train_time_sec": round(total_time, 3),
        }
