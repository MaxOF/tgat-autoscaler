"""Candidate filtering, learned scoring and bounded graph reconstruction."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from app.core.interfaces import HiddenEdgePrediction
from config.settings import CONFIG, EDGE_FEATURE_ORDER


class HiddenEdgeReconstructor:
    def __init__(
        self,
        network,
        node_ids,
        raw_features,
        node_meta,
        temporal_encoder,
        edge_dim,
        use_graph=True,
        enabled=True,
    ):
        self.model = network
        self.idx2id = node_ids
        self.raw_node_features = raw_features
        self.node_meta = node_meta
        self.temporal_encoder = temporal_encoder
        self.ckpt_edge_dim = edge_dim
        self.device = next(network.parameters()).device
        self.use_graph = use_graph
        self.use_hidden_edges = enabled
        self.time_encoding_enabled = True
        self.last_hidden_edges = []

    @staticmethod
    def _bounded_similarity(a: float, b: float) -> float:
        """
        Scale-free similarity in [0, 1].

        sim = 1 - |a-b|/(|a|+|b|+eps)
        """

        denominator = abs(a) + abs(b) + 1e-6

        score = 1.0 - abs(a - b) / denominator

        return float(np.clip(score, 0.0, 1.0))

    def _node_feature_by_name(
        self, node_id: str, name: str, fallback_index: int
    ) -> float:
        meta = self.node_meta.get(node_id, {})

        features = meta.get("features", {})

        if isinstance(features, dict) and name in features:
            try:
                return float(features[name])
            except Exception:
                pass

        raw = self.raw_node_features.get(node_id)

        if raw is not None and raw.size > fallback_index:
            return float(raw[fallback_index])

        return 0.0

    def _aux_similarity(self, src_id: str, dst_id: str) -> Tuple[float, float, float]:
        """
        psi_ji =
        [
            request-rate similarity,
            latency similarity,
            architecture prior
        ]

        Current GraphWindow contains one aggregated state per node.
        Historical correlation can later replace these instantaneous
        similarities without changing the hidden-edge interface.
        """

        src_rps = self._node_feature_by_name(src_id, "rps_out", 3)

        dst_rps = self._node_feature_by_name(dst_id, "rps_in", 2)

        src_p95 = self._node_feature_by_name(src_id, "p95_ms", 4)

        dst_p95 = self._node_feature_by_name(dst_id, "p95_ms", 4)

        rps_similarity = self._bounded_similarity(src_rps, dst_rps)

        latency_similarity = self._bounded_similarity(src_p95, dst_p95)

        dependencies = (
            CONFIG.get("services", {}).get(src_id, {}).get("dependencies", [])
        )

        architecture_prior = 1.0 if dst_id in dependencies else 0.0

        return (rps_similarity, latency_similarity, architecture_prior)

    # =========================================================================
    # HIDDEN-EDGE SYNTHESIS
    # =========================================================================

    def synthesize(
        self,
        h: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Add high-confidence candidate dependencies.

        A first encoder pass over the observed graph provides h_i.
        Candidate probabilities are then computed and accepted edges are
        appended before the final prediction pass.
        """

        self.last_hidden_edges = []

        hidden_cfg = CONFIG.get("hidden_edge", {})

        if not bool(hidden_cfg.get("enabled", True)):
            return (edge_index, edge_attr)

        if not self.use_hidden_edges:
            return (edge_index, edge_attr)

        if not self.use_graph:
            return (edge_index, edge_attr)

        if self.model is None:
            return (edge_index, edge_attr)

        num_nodes = len(self.idx2id)

        if num_nodes < 2:
            return (edge_index, edge_attr)

        observed_pairs = set()

        if edge_index is not None:
            edge_cpu = edge_index.detach().cpu().numpy()

            for k in range(edge_cpu.shape[1]):
                observed_pairs.add((int(edge_cpu[0, k]), int(edge_cpu[1, k])))

        candidate_src: List[int] = []

        candidate_dst: List[int] = []

        candidate_psi: List[List[float]] = []

        prefilter_threshold = float(
            hidden_cfg.get("candidate_similarity_threshold", 0.25)
        )

        gamma_rps = float(hidden_cfg.get("gamma_rps", 0.45))

        gamma_latency = float(hidden_cfg.get("gamma_latency", 0.35))

        gamma_arch = float(hidden_cfg.get("gamma_arch", 0.20))

        candidate_records = []

        for src_idx in range(num_nodes):
            for dst_idx in range(num_nodes):
                if src_idx == dst_idx:
                    continue

                if (src_idx, dst_idx) in observed_pairs:
                    continue

                src_id = self.idx2id[src_idx]

                dst_id = self.idx2id[dst_idx]

                rps_similarity, latency_similarity, architecture_prior = (
                    self._aux_similarity(src_id, dst_id)
                )

                prefilter_score = (
                    gamma_rps * rps_similarity
                    + gamma_latency * latency_similarity
                    + gamma_arch * architecture_prior
                )

                if prefilter_score < prefilter_threshold:
                    continue

                candidate_records.append(
                    (
                        prefilter_score,
                        src_idx,
                        dst_idx,
                        rps_similarity,
                        latency_similarity,
                        architecture_prior,
                    )
                )

        max_candidates_total = int(hidden_cfg.get("max_candidates_total", 128))

        candidate_records.sort(key=lambda item: item[0], reverse=True)

        candidate_records = candidate_records[:max_candidates_total]

        if not candidate_records:
            return (edge_index, edge_attr)

        for (
            _,
            src_idx,
            dst_idx,
            rps_similarity,
            latency_similarity,
            architecture_prior,
        ) in candidate_records:
            candidate_src.append(src_idx)

            candidate_dst.append(dst_idx)

            candidate_psi.append(
                [rps_similarity, latency_similarity, architecture_prior]
            )

        src_tensor = torch.tensor(candidate_src, dtype=torch.long, device=self.device)

        dst_tensor = torch.tensor(candidate_dst, dtype=torch.long, device=self.device)

        psi_tensor = torch.tensor(
            candidate_psi, dtype=torch.float32, device=self.device
        )

        probabilities = self.model.score_hidden_edges(
            h[src_tensor], h[dst_tensor], psi_tensor
        )

        threshold = float(hidden_cfg.get("threshold", 0.70))

        top_k = max(0, int(hidden_cfg.get("top_k", 3)))

        # Accept at most top-k incoming reconstructed dependencies
        # per target service.
        by_target: Dict[int, List[Tuple[float, int, int, int]]] = {}

        probability_values = probabilities.detach().cpu().tolist()

        for cand_idx, prob in enumerate(probability_values):
            prob = float(prob)

            if prob < threshold:
                continue

            dst_idx = candidate_dst[cand_idx]

            by_target.setdefault(dst_idx, []).append(
                (prob, cand_idx, candidate_src[cand_idx], dst_idx)
            )

        accepted = []

        for _, rows in by_target.items():
            rows.sort(key=lambda item: item[0], reverse=True)

            accepted.extend(rows[:top_k])

        if not accepted:
            return (edge_index, edge_attr)

        new_src: List[int] = []

        new_dst: List[int] = []

        new_attr: List[torch.Tensor] = []

        target_edge_dim = int(
            self.ckpt_edge_dim or (edge_attr.size(1) if edge_attr is not None else 0)
        )

        raw_edge_dim = len(EDGE_FEATURE_ORDER)

        for prob, cand_idx, src_idx, dst_idx in accepted:
            src_id = self.idx2id[src_idx]

            dst_id = self.idx2id[dst_idx]

            psi = candidate_psi[cand_idx]

            self.last_hidden_edges.append(
                HiddenEdgePrediction(
                    src=src_id,
                    dst=dst_id,
                    probability=prob,
                    rps_similarity=float(psi[0]),
                    latency_similarity=float(psi[1]),
                    architectural_prior=float(psi[2]),
                )
            )

            new_src.append(src_idx)

            new_dst.append(dst_idx)

            if target_edge_dim > 0:
                attr = torch.zeros(
                    target_edge_dim, dtype=torch.float32, device=self.device
                )

                # Raw layout:
                #
                # 0 edge_weight
                # 1 edge_p95
                # 2 edge_errors
                # 3 confidence
                attr[0] = prob

                if target_edge_dim >= 4:
                    attr[3] = prob

                # New edge is synthesized at the current decision
                # instant, therefore delta_t = 0.
                if self.time_encoding_enabled and target_edge_dim > raw_edge_dim:
                    zero_delta = torch.zeros(1, dtype=torch.float32, device=self.device)

                    phi = self.temporal_encoder(zero_delta)[0]

                    available = min(phi.numel(), (target_edge_dim - raw_edge_dim))

                    attr[raw_edge_dim : raw_edge_dim + available] = phi[:available]

                new_attr.append(attr)

        synth_edge_index = torch.tensor(
            [new_src, new_dst], dtype=torch.long, device=self.device
        )

        if edge_index is None:
            combined_edge_index = synth_edge_index
        else:
            combined_edge_index = torch.cat([edge_index, synth_edge_index], dim=1)

        if target_edge_dim <= 0:
            combined_edge_attr = None

        else:
            synth_edge_attr = torch.stack(new_attr, dim=0)

            if edge_attr is None:
                combined_edge_attr = synth_edge_attr
            else:
                combined_edge_attr = torch.cat([edge_attr, synth_edge_attr], dim=0)

        return (combined_edge_index, combined_edge_attr)
