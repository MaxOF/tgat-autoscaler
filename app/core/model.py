from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.interfaces import (
    Action,
    EdgeEvent,
    GraphWindow,
    HiddenEdgePrediction,
    NodeFeatures,
    Prediction,
)
from app.core.temporal_fourier_encoding import TemporalFourierEncoding

from app.utils.parsers import (
    format_cpu_milli,
    format_mem_gi_from_mib,
    parse_cpu_milli,
    parse_iso8601,
    parse_mem_mib,
)

from config.settings import (
    CONFIG,
    EDGE_FEATURE_ORDER,
    MODEL_OUTPUT_DIM,
    MODEL_PATH,
    PYG_AVAILABLE,
    FEATURE_ORDER,
    TARGET_ORDER,
)

# =============================================================================
# NEURAL MODEL
# =============================================================================


from app.core.network import SimpleTGAT

# =============================================================================
# TGAT AUTOSCALER WRAPPER
# =============================================================================


class TGATAutoscalerModel:
    """
    High-level model wrapper responsible for:

    - temporal feature construction;
    - checkpoint management;
    - hidden-edge synthesis;
    - noGraph ablation;
    - seven-target prediction;
    - backward-compatible conversion of predictions to Actions.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)

        periods_min = self.cfg.get(
            "fourier_periods_min", [60, 6 * 60, 24 * 60, 7 * 24 * 60]
        )

        self.temporal_encoder = TemporalFourierEncoding(periods_min)

        self.time_encoding_enabled = bool(self.cfg.get("time_encoding", True))

        self.dropedge_prob = float(self.cfg.get("dropedge_prob", 0.0))

        self.use_graph = bool(self.cfg.get("use_graph", True))

        self.use_hidden_edges = bool(self.cfg.get("use_hidden_edges", True))

        self.model: Optional[SimpleTGAT] = None

        # Feature-wise standardization parameters.
        self.feature_mu: Optional[np.ndarray] = None

        self.feature_sd: Optional[np.ndarray] = None

        # Legacy attribute retained because older project code
        # may inspect it.
        self.scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self.id2idx: Dict[str, int] = {}

        self.idx2id: List[str] = []

        self.raw_node_features: Dict[str, np.ndarray] = {}

        self.node_meta: Dict[str, Dict[str, Any]] = {}

        self.ckpt_in_dim: Optional[int] = None

        self.ckpt_edge_dim: Optional[int] = None

        self.ckpt_output_dim: Optional[int] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_scale = False

        # Diagnostic information useful for direct reconstruction
        # evaluation requested by reviewers.
        self.last_hidden_edges: List[HiddenEdgePrediction] = []

    # =========================================================================
    # FEATURE NORMALIZATION
    # =========================================================================

    def _apply_scaler(self, node_id: str, x: np.ndarray) -> np.ndarray:
        if self.feature_mu is None or self.feature_sd is None:
            return x

        if x.shape != self.feature_mu.shape:
            # Graphs created from legacy CSV data can still
            # contain the six-feature representation.
            return x

        return (x - self.feature_mu) / self.feature_sd

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    def build_graph_from_payload(
        self, gw: GraphWindow
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Convert GraphWindow to tensors.

        Hidden-edge synthesis is deliberately NOT performed here.

        It is performed later in predict_targets(), after the first
        observed-graph encoder pass, which makes it possible to use
        latent h_i and h_j in the hidden-edge probability model.
        """

        if any(len(n.x) != len(FEATURE_ORDER) for n in gw.nodes):
            raise ValueError("Node feature dimensions do not match FEATURE_ORDER")
        if len({n.id for n in gw.nodes}) != len(gw.nodes):
            raise ValueError("Duplicate node IDs")
        if any(not np.isfinite(n.x).all() for n in gw.nodes):
            raise ValueError("Node features must be finite")
        if not gw.nodes:
            raise ValueError("GraphWindow contains no nodes")

        self.id2idx = {n.id: i for i, n in enumerate(gw.nodes)}

        self.idx2id = [n.id for n in gw.nodes]

        self.raw_node_features = {
            n.id: np.asarray(n.x, dtype=np.float32) for n in gw.nodes
        }

        self.node_meta = {
            n.id: (n.meta if n.meta is not None else {}) for n in gw.nodes
        }

        if self.feature_mu is None and self.model is None:
            checkpoint_path = self.cfg.get("model_path", MODEL_PATH)
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)

        X = np.stack(
            [
                self._apply_scaler(n.id, np.asarray(n.x, dtype=np.float32))
                for n in gw.nodes
            ],
            axis=0,
        )

        x = torch.tensor(X, dtype=torch.float32)

        if not gw.events:
            return (x, None, None)

        t_ref = parse_iso8601(gw.window)

        src_idx: List[int] = []

        dst_idx: List[int] = []

        edge_feats: List[np.ndarray] = []

        deltas: List[float] = []

        expected_raw_edge_dim = len(EDGE_FEATURE_ORDER)

        # Recent events are sampled independently for each incoming neighborhood.
        t_ref = parse_iso8601(gw.window)
        window_min = float(CONFIG.get("observation_window_minutes", 60))
        limit = int(CONFIG.get("max_temporal_neighbors", 32))
        counts = {}
        eligible = []
        for ev in sorted(gw.events, key=lambda e: parse_iso8601(e.tau), reverse=True):
            lag = (t_ref - parse_iso8601(ev.tau)).total_seconds() / 60
            if not 0 < lag <= window_min:
                continue
            if counts.get(ev.dst, 0) >= limit:
                continue
            counts[ev.dst] = counts.get(ev.dst, 0) + 1
            eligible.append(ev)
        for ev in eligible:
            if ev.src not in self.id2idx or ev.dst not in self.id2idx:
                continue

            src_idx.append(self.id2idx[ev.src])

            dst_idx.append(self.id2idx[ev.dst])

            raw = list(ev.e)

            # Existing graph.py produces three raw values:
            #
            # [edge_weight,
            #  edge_p95_ms,
            #  edge_errors]
            #
            # Add edge-confidence as the fourth value.
            if len(raw) < 3:
                raw.extend([0.0 for _ in range(3 - len(raw))])

            raw = raw[:3]

            raw.append(float(np.clip(ev.confidence, 0.0, 1.0)))

            # Future-proofing if EDGE_FEATURE_ORDER changes.
            if len(raw) < expected_raw_edge_dim:
                raw.extend([0.0 for _ in range(expected_raw_edge_dim - len(raw))])

            raw = raw[:expected_raw_edge_dim]

            if not np.isfinite(raw).all():
                raise ValueError("Edge features must be finite")
            edge_feats.append(np.asarray(raw, dtype=np.float32))

            tau = parse_iso8601(ev.tau)

            delta_min = max(0.0, (t_ref - tau).total_seconds() / 60.0)

            deltas.append(delta_min)

        if not src_idx:
            return (x, None, None)

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        edge_attr = torch.tensor(np.stack(edge_feats, axis=0), dtype=torch.float32)

        delta_t = torch.tensor(deltas, dtype=torch.float32)

        if self.time_encoding_enabled:
            phi = self.temporal_encoder(delta_t)

            edge_attr = torch.cat([edge_attr, phi], dim=1)

        edge_index, edge_attr = self._apply_dropedge(edge_index, edge_attr)

        return (x, edge_index, edge_attr)

    # =========================================================================
    # DROPEDGE
    # =========================================================================

    def _apply_dropedge(
        self, edge_index: Optional[torch.Tensor], edge_attr: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            edge_index is None
            or edge_attr is None
            or self.dropedge_prob <= 0.0
            or not self.model_training_mode()
        ):
            return (edge_index, edge_attr)

        num_edges = int(edge_index.size(1))

        if num_edges == 0:
            return (edge_index, edge_attr)

        keep = torch.rand(num_edges) >= self.dropedge_prob

        if not torch.any(keep):
            # Keep one edge to avoid collapsing the entire
            # graph by chance.
            keep[torch.randint(0, num_edges, (1,))] = True

        return (edge_index[:, keep], edge_attr[keep])

    def model_training_mode(self) -> bool:
        return bool(self.model is not None and self.model.training)

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================

    def init_model(self, in_dim: int, edge_dim: int) -> None:
        self.model = SimpleTGAT(
            in_dim=in_dim,
            edge_dim=edge_dim,
            d_model=int(self.cfg.get("d_model", 128)),
            heads=int(self.cfg.get("heads", 4)),
            layers=int(self.cfg.get("layers", 2)),
            dropout=float(self.cfg.get("dropout", 0.10)),
        ).to(self.device)

    # =========================================================================
    # EDGE ATTRIBUTE ALIGNMENT
    # =========================================================================

    def _align_edge_attr(
        self, edge_attr: Optional[torch.Tensor], target_dim: int, num_edges: int
    ) -> Optional[torch.Tensor]:
        if target_dim <= 0:
            return None

        if edge_attr is None:
            return torch.zeros(
                (num_edges, target_dim), dtype=torch.float32, device=self.device
            )

        edge_attr = edge_attr.to(self.device)

        cur = int(edge_attr.size(1))

        if cur == target_dim:
            return edge_attr

        if cur > target_dim:
            return edge_attr[:, :target_dim]

        pad = torch.zeros(
            (edge_attr.size(0), target_dim - cur),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )

        return torch.cat([edge_attr, pad], dim=1)

    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================

    @staticmethod
    def _infer_checkpoint_output_dim(
        state_dict: Dict[str, torch.Tensor],
    ) -> Optional[int]:
        """
        Detect new seven-target checkpoints and reject legacy
        three-target checkpoints.
        """

        # New architecture:
        #
        # regression_head -> 6
        # risk_head       -> 1
        if "regression_head.weight" in state_dict and "risk_head.weight" in state_dict:
            regression_dim = int(state_dict["regression_head.weight"].shape[0])

            risk_dim = int(state_dict["risk_head.weight"].shape[0])

            return regression_dim + risk_dim

        # Legacy model.py:
        #
        # self.head =
        # Sequential(... Linear(..., 3))
        #
        # Search conservatively for an obvious final 3-output head.
        for key, tensor in state_dict.items():
            if (
                key.endswith(".weight")
                and tensor.ndim == 2
                and tensor.shape[0] == 3
                and "head" in key
            ):
                return 3

        return None

    @property
    def edge_feature_dim(self) -> int:
        return len(EDGE_FEATURE_ORDER) + 2 * len(
            self.cfg.get("fourier_periods_min", [60, 360, 1440, 10080])
        )

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if ckpt.get("schema_version") != 2:
                raise ValueError("Checkpoint schema changed; retraining is required")
            if ckpt.get("target_order") != TARGET_ORDER:
                raise ValueError("Checkpoint target order does not match")
            if ckpt.get("feature_order") != FEATURE_ORDER:
                raise ValueError("Checkpoint feature order does not match")
            mu = np.asarray(ckpt["feature_scaler"]["mean"], dtype=np.float32)
            sd = np.asarray(ckpt["feature_scaler"]["std"], dtype=np.float32)
            target_scale = np.asarray(ckpt["target_scale"], dtype=np.float32)
            if mu.shape != (len(FEATURE_ORDER),) or sd.shape != mu.shape:
                raise ValueError("Invalid feature scaler dimensions")
            if (
                not np.isfinite(mu).all()
                or not np.isfinite(sd).all()
                or (sd <= 0).any()
            ):
                raise ValueError("Invalid feature scaler values")
            if (
                target_scale.shape != (6,)
                or not np.isfinite(target_scale).all()
                or (target_scale <= 0).any()
            ):
                raise ValueError("Invalid target scale")
            cfg = dict(self.cfg)
            for key in ("d_model", "heads", "layers", "dropout", "fourier_periods_min"):
                if key in ckpt["cfg"]:
                    cfg[key] = ckpt["cfg"][key]
            candidate = SimpleTGAT(
                int(ckpt["in_dim"]),
                int(ckpt["edge_dim"]),
                d_model=cfg.get("d_model", 128),
                heads=cfg.get("heads", 4),
                layers=cfg.get("layers", 2),
                dropout=cfg.get("dropout", 0.1),
            ).to(self.device)
            candidate.load_state_dict(ckpt["model"], strict=True)
            candidate.eval()
            self.cfg = cfg
            self.temporal_encoder = TemporalFourierEncoding(
                cfg["fourier_periods_min"]
            ).to(self.device)
            self.model = candidate
            self.feature_mu, self.feature_sd = mu, sd
            self.target_scale = target_scale
            self.ckpt_in_dim = int(ckpt["in_dim"])
            self.ckpt_edge_dim = int(ckpt["edge_dim"])
            self.ckpt_output_dim = MODEL_OUTPUT_DIM
            return True
        except Exception as exc:
            self.checkpoint_error = str(exc)
            return False

    def ensure_ready(self, in_dim_now: int, edge_dim_now: int, model_path: str) -> None:
        """
        Ensure a seven-target model is available.

        Scientifically important behavior:
        an incompatible legacy checkpoint is NOT silently accepted.
        """

        if self.model is not None and self.ckpt_in_dim is not None:
            return

        if self.load_checkpoint(model_path):
            return

        allow_untrained = bool(self.cfg.get("allow_untrained_inference", False))

        if not allow_untrained:
            raise RuntimeError(
                "No compatible seven-target "
                "TGAT checkpoint is available. "
                "The old three-output model "
                "must be retrained after the "
                "training pipeline is migrated "
                "to the seven prediction targets."
            )

        # Useful only for development/tests.
        self.ckpt_in_dim = int(in_dim_now)

        self.ckpt_edge_dim = int(edge_dim_now)

        self.ckpt_output_dim = MODEL_OUTPUT_DIM

        self.init_model(in_dim_now, edge_dim_now)

        assert self.model is not None

        self.model.eval()

    # =========================================================================
    # AUXILIARY SIMILARITY FEATURES
    # =========================================================================

    def _synthesize_hidden_edges(self, h, edge_index, edge_attr):
        from app.core.hidden_edges import HiddenEdgeReconstructor

        reconstructor = HiddenEdgeReconstructor(
            self.model,
            self.idx2id,
            self.raw_node_features,
            self.node_meta,
            self.temporal_encoder,
            self.ckpt_edge_dim,
            use_graph=self.use_graph,
            enabled=self.use_hidden_edges,
        )
        reconstructor.time_encoding_enabled = self.time_encoding_enabled
        result = reconstructor.synthesize(h, edge_index, edge_attr)
        self.last_hidden_edges = reconstructor.last_hidden_edges
        return result

    # =========================================================================
    # SEVEN-TARGET PREDICTION
    # =========================================================================

    def predict_targets(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> List[Prediction]:
        """
        Return neural forecasts only.

        This method intentionally does NOT execute the safety layer.
        It therefore preserves the forecasting/actuation separation
        required by the revised manuscript.
        """

        in_dim_now = int(x.shape[1])

        edge_dim_now = int(edge_attr.shape[1]) if edge_attr is not None else 0

        self.ensure_ready(
            in_dim_now, self.edge_feature_dim, self.cfg.get("model_path", MODEL_PATH)
        )

        assert self.model is not None

        if self.ckpt_in_dim is not None and in_dim_now != self.ckpt_in_dim:
            raise ValueError(
                "Node feature dimension "
                "does not match checkpoint: "
                f"current={in_dim_now}, "
                f"checkpoint="
                f"{self.ckpt_in_dim}"
            )

        x = x.to(self.device)

        if edge_index is not None:
            edge_index = edge_index.to(self.device)

        target_edge_dim = int(self.ckpt_edge_dim or edge_dim_now or 0)

        edge_attr = self._align_edge_attr(
            edge_attr,
            target_edge_dim,
            (edge_index.size(1) if edge_index is not None else 0),
        )

        self.model.eval()

        with torch.no_grad():
            # -------------------------------------------------------------
            # First pass over OBSERVED graph.
            # -------------------------------------------------------------
            h_observed = self.model.encode(
                x, edge_index, edge_attr, force_no_graph=(not self.use_graph)
            )

            # -------------------------------------------------------------
            # Hidden-edge synthesis.
            # -------------------------------------------------------------
            extended_edge_index, extended_edge_attr = self._synthesize_hidden_edges(
                h_observed, edge_index, edge_attr
            )

            # -------------------------------------------------------------
            # Final inference over extended graph.
            # -------------------------------------------------------------
            raw_out = self.model(
                x,
                extended_edge_index,
                extended_edge_attr,
                force_no_graph=(not self.use_graph),
            )

        if hasattr(self, "target_scale"):
            scale = torch.as_tensor(self.target_scale, device=raw_out.device)
            raw_out[:, :6] = raw_out[:, :6] * scale
        return self._postprocess_predictions(raw_out.detach().cpu().numpy())

    def _postprocess_predictions(self, raw_out: np.ndarray) -> List[Prediction]:
        if not np.isfinite(raw_out).all():
            raise ValueError("Predictions must be finite")
        if raw_out.ndim != 2:
            raise ValueError("Expected model output " "with shape [N, 7]")

        if raw_out.shape[1] != MODEL_OUTPUT_DIM:
            raise ValueError(
                "Expected "
                f"{MODEL_OUTPUT_DIM} "
                "prediction targets, got "
                f"{raw_out.shape[1]}"
            )

        if raw_out.shape[0] != len(self.idx2id):
            raise ValueError("Prediction node count " "does not match idx2id")

        predictions: List[Prediction] = []

        for i, node_id in enumerate(self.idx2id):
            rps, replicas, cpu_milli, mem_mib, p95_ms, p99_ms, slo_risk = raw_out[i]

            predictions.append(
                Prediction(
                    id=node_id,
                    rps=max(0.0, float(rps)),
                    replicas=max(1.0, float(replicas)),
                    cpu_milli=max(0.0, float(cpu_milli)),
                    mem_mib=max(0.0, float(mem_mib)),
                    p95_ms=max(0.0, float(p95_ms)),
                    p99_ms=max(0.0, float(p99_ms)),
                    slo_risk=float(np.clip(slo_risk, 0.0, 1.0)),
                )
            )

        return predictions

    # =========================================================================
    # PREDICTION -> PRELIMINARY RESOURCE ACTIONS
    # =========================================================================

    def predictions_to_actions(self, predictions: List[Prediction]) -> List[Action]:
        """
        Convert forecasts into PRELIMINARY resource targets.

        This does not replace SafetyPolicy.
        Cooldown, hysteresis, rate limits, smoothing, and objective
        optimization are applied later by the decision/safety layer.
        """

        actions: List[Action] = []

        global_safety = CONFIG.get("safety", {})

        for prediction in predictions:
            node_id = prediction.id

            svc_cfg = CONFIG.get("services", {}).get(node_id, {})

            min_replicas = int(
                svc_cfg.get("min_replicas", global_safety.get("r_min", 1))
            )

            max_replicas = int(
                svc_cfg.get("max_replicas", global_safety.get("r_max", 50))
            )

            replicas = int(
                np.clip(round(prediction.replicas), min_replicas, max_replicas)
            )

            min_cpu = parse_cpu_milli(svc_cfg.get("min_cpu", "100m"))

            max_cpu = parse_cpu_milli(svc_cfg.get("max_cpu", "4000m"))

            min_mem = parse_mem_mib(svc_cfg.get("min_memory", "256Mi"))

            max_mem = parse_mem_mib(svc_cfg.get("max_memory", "8192Mi"))

            cpu_milli = float(np.clip(prediction.cpu_milli, min_cpu, max_cpu))

            mem_mib = float(np.clip(prediction.mem_mib, min_mem, max_mem))

            actions.append(
                Action(
                    id=node_id,
                    replicas=replicas,
                    cpu=format_cpu_milli(cpu_milli),
                    mem=(format_mem_gi_from_mib(mem_mib)),
                )
            )

        return actions

    # =========================================================================
    # BACKWARD-COMPATIBLE METHOD
    # =========================================================================

    def predict_graph(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> List[Action]:
        """
        Backward-compatible API for the current TGATService.

        Existing code expects predict_graph() -> List[Action].

        New code should use:

            predictions = model.predict_targets(...)
            actions = model.predictions_to_actions(predictions)

        before passing actions/predictions to the deterministic
        SafetyPolicy.
        """

        predictions = self.predict_targets(x, edge_index, edge_attr)

        return self.predictions_to_actions(predictions)

    # =========================================================================
    # ABLATION CONTROL
    # =========================================================================

    def configure_ablation(
        self,
        *,
        time_encoding: Optional[bool] = None,
        hidden_edges: Optional[bool] = None,
        graph_enabled: Optional[bool] = None,
        dropedge_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply experimental ablation settings.

        Supports:
        - w/o temporal encoding
        - w/o hidden-edge synthesis
        - TGAT-noGraph
        - DropEdge sensitivity
        """

        if time_encoding is not None:
            self.time_encoding_enabled = bool(time_encoding)

        if hidden_edges is not None:
            self.use_hidden_edges = bool(hidden_edges)

            self.cfg["use_hidden_edges"] = self.use_hidden_edges

        if graph_enabled is not None:
            self.use_graph = bool(graph_enabled)

            self.cfg["use_graph"] = self.use_graph

        if dropedge_prob is not None:
            probability = float(dropedge_prob)

            if not (0.0 <= probability < 1.0):
                raise ValueError("dropedge_prob must " "be in [0, 1)")

            self.dropedge_prob = probability

            self.cfg["dropedge_prob"] = probability

        return self.ablation_state()

    def ablation_state(self) -> Dict[str, Any]:
        return {
            "time_encoding": self.time_encoding_enabled,
            "hidden_edges": self.use_hidden_edges,
            "graph_enabled": self.use_graph,
            "dropedge_prob": self.dropedge_prob,
        }

    # =========================================================================
    # HIDDEN-EDGE DIAGNOSTICS
    # =========================================================================

    def get_last_hidden_edges(self) -> List[Dict[str, Any]]:
        """
        Return accepted hidden-edge predictions from the last inference
        pass. Useful for precision/recall/F1 evaluation.
        """

        return [edge.model_dump() for edge in self.last_hidden_edges]
