"""Batched temporal training with training-only normalization and held-out validation."""

from __future__ import annotations
import copy
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from app.data.csv_dataset import CSVGraphDataset
from config.settings import (
    CONFIG,
    FEATURE_ORDER,
    TARGET_ORDER,
    MODEL_OUTPUT_DIM,
    MODEL_PATH,
)


class CSVTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, req):
        started = time.perf_counter()
        random.seed(req.seed)
        np.random.seed(req.seed)
        torch.manual_seed(req.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        device = torch.device(
            req.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type not in ("cpu", "cuda") or (
            device.type == "cuda" and not torch.cuda.is_available()
        ):
            raise ValueError("Requested training device is unavailable")
        dataset = CSVGraphDataset(
            req.nodes_csv_path, req.edges_csv_path, req.csv_time_column
        )
        samples = list(dataset.iter_samples())
        if len(samples) < 3:
            raise ValueError("At least three complete forecast samples are required")
        split = int(len(samples) * (1 - req.validation_fraction))
        boundary = pd.Timestamp(samples[split].graph.window)
        horizon = pd.Timedelta(minutes=dataset.horizon_minutes)
        # Purge samples whose future labels touch the validation partition.
        train = [
            s
            for s in samples[:split]
            if pd.Timestamp(s.graph.window) + horizon < boundary
        ]
        valid = samples[split:]
        if not train or not valid:
            raise ValueError("Temporal split produced an empty partition")
        self.model.model = None
        self.model.feature_mu, self.model.feature_sd = dataset.feature_scaler(
            [pd.Timestamp(s.graph.window) for s in train]
        )
        target_values = torch.cat([s.targets[:, :6] for s in train])
        scale = target_values.square().mean(dim=0).sqrt().clamp_min(1).numpy()
        self.model.target_scale = scale
        self.model.init_model(len(FEATURE_ORDER), self.model.edge_feature_dim)
        net = self.model.model.to(device)
        edge_dim = self.model.edge_feature_dim

        def tensors(sample):
            x, ei, ea = self.model.build_graph_from_payload(sample.graph)
            if ei is None:
                ei = torch.empty((2, 0), dtype=torch.long)
                ea = torch.empty((0, edge_dim), dtype=torch.float32)
            elif ea.shape[1] < edge_dim:
                ea = F.pad(ea, (0, edge_dim - ea.shape[1]))
            return Data(x=x, edge_index=ei, edge_attr=ea, y=sample.targets)

        train_data = [tensors(s) for s in train]
        valid_data = [tensors(s) for s in valid]
        train_pairs = {(e.src, e.dst) for s in train for e in s.graph.events}
        ids = dataset.services
        # This is supervision from TRAINING observed interactions, never ground truth.
        src = []
        dst = []
        labels = []
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i != j:
                    src.append(i)
                    dst.append(j)
                    labels.append(float((a, b) in train_pairs))
        architecture_prior = torch.tensor(
            [
                float(
                    ids[j] in CONFIG["services"].get(ids[i], {}).get("dependencies", [])
                )
                for i, j in zip(src, dst)
            ],
            device=device,
        )
        src = torch.tensor(src, device=device)
        dst = torch.tensor(dst, device=device)
        edge_labels = torch.tensor(labels, device=device)
        scale_tensor = torch.tensor(scale, device=device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=req.learning_rate, weight_decay=req.weight_decay
        )
        best_loss = float("inf")
        best_state = None
        last_loss = None
        last_reg = None
        last_risk = None
        batch_size = 128
        generator = random.Random(req.seed)

        def loss_components(batch, augment=False):
            ei, ea = batch.edge_index, batch.edge_attr
            if augment and ei.shape[1]:
                keep = torch.rand(ei.shape[1], device=device) >= 0.3
                ei, ea = ei[:, keep], ea[keep]
            h = net.encode(batch.x, ei, ea, force_no_graph=not self.model.use_graph)
            pred = net.decode(h)
            regression = F.smooth_l1_loss(pred[:, :6], batch.y[:, :6] / scale_tensor)
            risk = F.binary_cross_entropy(
                pred[:, 6].clamp(1e-6, 1 - 1e-6), batch.y[:, 6]
            )
            edge_loss = torch.zeros((), device=device)
            if (
                augment
                and self.model.use_graph
                and self.model.use_hidden_edges
                and len(src)
            ):
                offset = batch.ptr[:-1, None]
                src_batch = (src[None, :] + offset).flatten()
                dst_batch = (dst[None, :] + offset).flatten()
                # Similarity uses physical telemetry, matching inference.
                mean = torch.as_tensor(self.model.feature_mu, device=device)
                std = torch.as_tensor(self.model.feature_sd, device=device)
                raw = batch.x * std + mean

                def sim(a, b):
                    return (1 - (a - b).abs() / (a.abs() + b.abs() + 1e-6)).clamp(0, 1)

                arch = architecture_prior.repeat(batch.num_graphs)
                psi = torch.stack(
                    [
                        sim(raw[src_batch, 3], raw[dst_batch, 2]),
                        sim(raw[src_batch, 4], raw[dst_batch, 4]),
                        arch,
                    ],
                    dim=1,
                )
                probabilities = net.score_hidden_edges(h[src_batch], h[dst_batch], psi)
                edge_loss = F.binary_cross_entropy(
                    probabilities.clamp(1e-6, 1 - 1e-6),
                    edge_labels.repeat(batch.num_graphs),
                )
            return regression + 0.25 * risk + 0.1 * edge_loss, regression, risk

        for epoch in range(req.epochs):
            net.train()
            order = list(range(len(train_data)))
            if req.shuffle:
                generator.shuffle(order)
            losses = []
            regs = []
            risks = []
            for start in range(0, len(order), batch_size):
                batch = Batch.from_data_list(
                    [train_data[i] for i in order[start : start + batch_size]]
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, reg, risk = loss_components(batch, augment=True)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                regs.append(float(reg.detach()))
                risks.append(float(risk.detach()))
            net.eval()
            validation_sum = 0.0
            validation_nodes = 0
            with torch.no_grad():
                for start in range(0, len(valid_data), batch_size):
                    batch = Batch.from_data_list(
                        valid_data[start : start + batch_size]
                    ).to(device)
                    loss, _, _ = loss_components(batch)
                    validation_sum += float(loss) * batch.num_nodes
                    validation_nodes += batch.num_nodes
            validation = validation_sum / validation_nodes
            last_loss = float(np.mean(losses))
            last_reg = float(np.mean(regs))
            last_risk = float(np.mean(risks))
            if validation < best_loss:
                best_loss = validation
                best_state = copy.deepcopy(net.state_dict())
            print(
                f"epoch {epoch+1}/{req.epochs}: train={last_loss:.6f}, validation={validation:.6f}",
                flush=True,
            )
        net.load_state_dict(best_state)
        path = Path(req.model_path or self.model.cfg.get("model_path", MODEL_PATH))
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "schema_version": 2,
            "model": {k: v.cpu() for k, v in net.state_dict().items()},
            "cfg": self.model.cfg,
            "in_dim": len(FEATURE_ORDER),
            "edge_dim": edge_dim,
            "output_dim": MODEL_OUTPUT_DIM,
            "feature_order": FEATURE_ORDER,
            "target_order": TARGET_ORDER,
            "feature_scaler": {
                "mean": self.model.feature_mu.tolist(),
                "std": self.model.feature_sd.tolist(),
            },
            "target_scale": scale.tolist(),
            "seed": req.seed,
            "validation_start": boundary.isoformat(),
            "training_end": train[-1].graph.window,
            "best_validation_loss": best_loss,
        }
        torch.save(ckpt, path)
        self.model.cfg["model_path"] = str(path)
        self.model.ckpt_in_dim = len(FEATURE_ORDER)
        self.model.ckpt_edge_dim = edge_dim
        self.model.ckpt_output_dim = MODEL_OUTPUT_DIM
        self.model.model = net.to(self.model.device).eval()
        return {
            "status": "ok",
            "model_path": str(path),
            "samples": len(samples),
            "training_samples": len(train),
            "validation_samples": len(valid),
            "validation_start": boundary.isoformat(),
            "epochs": req.epochs,
            "last_epoch_loss": last_loss,
            "last_regression_loss": last_reg,
            "last_risk_loss": last_risk,
            "best_loss": best_loss,
            "device": device.type,
            "in_dim": len(FEATURE_ORDER),
            "edge_dim": edge_dim,
            "output_dim": MODEL_OUTPUT_DIM,
            "seed": req.seed,
            "train_time_sec": round(time.perf_counter() - started, 3),
        }
