"""Neural architecture; independent of datasets and Kubernetes."""

from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from config.settings import PYG_AVAILABLE

try:
    from torch_geometric.nn import GATv2Conv

    _PYG_IMPORT_OK = True
except ImportError:
    GATv2Conv = None
    _PYG_IMPORT_OK = False


class SimpleTGAT(nn.Module):
    """
    Compact temporal graph-attention predictor.

    Outputs
    -------
    Tensor [N, 7]

    Column order:
        0: request rate
        1: replicas
        2: CPU, millicores
        3: memory, MiB
        4: p95, ms
        5: p99, ms
        6: SLO violation probability
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        if heads <= 0:
            raise ValueError("heads must be > 0")

        if d_model % heads != 0:
            raise ValueError(
                "d_model must be divisible by heads: "
                f"d_model={d_model}, heads={heads}"
            )

        self.in_dim = int(in_dim)

        self.expected_edge_dim = int(max(0, edge_dim))

        self.d_model = int(d_model)

        self.heads = int(heads)

        self.layers = int(layers)

        self.dropout = float(dropout)

        self.gnn_enabled = bool(PYG_AVAILABLE and _PYG_IMPORT_OK)

        # ---------------------------------------------------------------------
        # Fallback branch
        #
        # Used for:
        # - noGraph ablation;
        # - graph windows without edges;
        # - environments without PyTorch Geometric.
        # ---------------------------------------------------------------------
        self.mlp_fallback = nn.Sequential(
            nn.Linear(self.in_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # ---------------------------------------------------------------------
        # Graph branch
        # ---------------------------------------------------------------------
        if self.gnn_enabled:
            self.input_projection = nn.Linear(self.in_dim, self.d_model)

            pyg_edge_dim = (
                self.expected_edge_dim if self.expected_edge_dim > 0 else None
            )

            self.convs = nn.ModuleList(
                [
                    GATv2Conv(
                        in_channels=self.d_model,
                        out_channels=(self.d_model // self.heads),
                        heads=self.heads,
                        concat=True,
                        edge_dim=pyg_edge_dim,
                        dropout=self.dropout,
                    )
                    for _ in range(self.layers)
                ]
            )

            self.out_norm = nn.LayerNorm(self.d_model)

        # ---------------------------------------------------------------------
        # Shared prediction representation
        # ---------------------------------------------------------------------
        self.shared_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Six continuous targets:
        #
        # rps, replicas, cpu, memory, p95, p99
        self.regression_head = nn.Linear(self.d_model // 2, 6)

        # SLO-risk probability.
        self.risk_head = nn.Linear(self.d_model // 2, 1)

        # ---------------------------------------------------------------------
        # Hidden-edge scorer
        #
        # p_ji =
        # sigmoid(
        #   a^T [
        #       h_j || h_i || |h_j-h_i| || psi_ji
        #   ] + b
        # )
        #
        # psi contains:
        #   rps similarity,
        #   latency similarity,
        #   architecture prior.
        #
        # The head is trainable.  It is conservatively initialized so
        # that auxiliary similarity features dominate before supervised
        # edge-reconstruction training is introduced.
        # ---------------------------------------------------------------------
        self.hidden_edge_psi_dim = 3

        hidden_edge_input_dim = 3 * self.d_model + self.hidden_edge_psi_dim

        self.hidden_edge_head = nn.Linear(hidden_edge_input_dim, 1)

        self._initialize_hidden_edge_prior()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _initialize_hidden_edge_prior(self) -> None:
        """
        Conservative initialization of hidden-edge probability.

        Embedding components initially receive zero weights.
        Auxiliary telemetry similarity controls the initial prior.

        The layer remains fully trainable.
        """

        with torch.no_grad():
            self.hidden_edge_head.weight.zero_()

            # Last three input dimensions correspond to:
            #
            # [rps_similarity,
            #  latency_similarity,
            #  architecture_prior]
            self.hidden_edge_head.weight[0, -3:] = torch.tensor(
                [1.50, 1.20, 1.80], dtype=(self.hidden_edge_head.weight.dtype)
            )

            # Require reasonably strong aggregate evidence.
            self.hidden_edge_head.bias.fill_(-2.20)

    # =========================================================================
    # ENCODER
    # =========================================================================

    def encode(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        *,
        force_no_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute latent service representations.
        """

        if not self.gnn_enabled:
            return self.mlp_fallback(x)
        if force_no_graph or edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_attr = torch.empty((0, self.expected_edge_dim), device=x.device)
        if self.expected_edge_dim and (
            edge_attr is None
            or edge_attr.shape != (edge_index.shape[1], self.expected_edge_dim)
        ):
            raise ValueError("Edge attributes do not match the network schema")
        # Empty neighborhoods retain the same trained projection and local self-message.
        h = self.input_projection(x)

        for conv in self.convs:
            if self.expected_edge_dim > 0:
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)

            h = F.elu(h)

            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.out_norm(h)

    # =========================================================================
    # FORECAST
    # =========================================================================

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Convert latent representation into seven prediction targets.
        """

        shared = self.shared_head(h)

        regression = self.regression_head(shared)

        risk = torch.sigmoid(self.risk_head(shared))

        return torch.cat([regression, risk], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        *,
        force_no_graph: bool = False,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr, force_no_graph=(force_no_graph))

        return self.decode(h)

    # =========================================================================
    # HIDDEN EDGE SCORING
    # =========================================================================

    def score_hidden_edges(
        self, h_src: torch.Tensor, h_dst: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate p(src -> dst).

        Parameters
        ----------
        h_src
            [C, d_model]

        h_dst
            [C, d_model]

        psi
            [C, 3]
        """

        if psi.ndim != 2:
            raise ValueError("psi must have shape [C, 3]")

        if psi.size(1) != self.hidden_edge_psi_dim:
            raise ValueError(
                "Expected hidden-edge "
                f"psi dimension "
                f"{self.hidden_edge_psi_dim}, "
                f"got {psi.size(1)}"
            )

        pair_repr = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), psi], dim=1)

        return torch.sigmoid(self.hidden_edge_head(pair_repr).squeeze(-1))
