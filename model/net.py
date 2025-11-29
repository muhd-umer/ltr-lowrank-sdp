"""\
This module implements the main GNN architecture for predicting the optimal
rank (r_oracle) required to solve SDP problems. The model performs graph
regression by encoding the constraint graph structure and predicting a single
scalar value.

Example usage:
    >>> from model.net import RankPredictor
    >>> model = RankPredictor()
    >>> out = model(x, edge_index, edge_attr, batch, global_attr)
    >>> print(out.shape)  # [batch_size, 1]
"""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from model.layers import EdgeEncoder, GlobalEncoder, NodeEncoder, PredictionHead


class RankPredictor(nn.Module):
    """Graph Neural Network for predicting optimal SDP rank.

    This model takes constraint graphs from SDP problem instances and predicts
    the oracle rank required for efficient solving. It uses GATv2Conv layers
    with attention over edges, residual connections, and concatenated pooling.

    The architecture is designed for graph regression where:
        - Nodes represent constraints with 8 spectral/structural features.
        - Edges represent constraint coupling with 2 similarity features.
        - Global features capture problem-level properties (log n, log m, etc).

    Attributes:
        node_encoder: Projects raw node features to hidden_dim.
        edge_encoder: Projects raw edge features to edge_dim.
        global_encoder: Projects global features to global_dim.
        convs: ModuleList of GATv2Conv layers.
        norms: ModuleList of LayerNorm layers for conv outputs.
        head: Prediction MLP mapping fused embedding to scalar rank.
        dropout: Dropout module for regularization.
        log_output: Whether to predict in log-space.
    """

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_in_dim: int = 2,
        global_in_dim: int = 5,
        hidden_dim: int = 64,
        edge_dim: int = 32,
        global_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm_type: Literal["batch", "layer", "none"] = "layer",
        log_output: bool = True,
    ) -> None:
        """Initialize the RankPredictor model.

        Args:
            node_in_dim: Input dimension for node features. Defaults to 8.
            edge_in_dim: Input dimension for edge features. Defaults to 2.
            global_in_dim: Input dimension for global features. Defaults to 5.
            hidden_dim: Hidden dimension for node embeddings. Defaults to 64.
            edge_dim: Latent dimension for edge embeddings. Defaults to 32.
            global_dim: Latent dimension for global embeddings. Defaults to 32.
            num_layers: Number of GATv2Conv layers. Defaults to 4.
            num_heads: Number of attention heads in GATv2Conv. Defaults to 4.
            dropout: Dropout probability. Defaults to 0.1.
            norm_type: Type of normalization ("batch", "layer", or "none").
                Defaults to "layer" for stability with variable graph sizes.
            log_output: If True, predict in log-space (recommended for labels
                spanning multiple orders of magnitude). Defaults to True.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.log_output = log_output
        self.dropout = nn.Dropout(p=dropout)

        self.node_encoder = NodeEncoder(
            in_dim=node_in_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.edge_encoder = EdgeEncoder(
            in_dim=edge_in_dim,
            out_dim=edge_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.global_encoder = GlobalEncoder(
            in_dim=global_in_dim,
            out_dim=global_dim,
            dropout=dropout,
            norm_type=norm_type,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_heads

            conv = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=True,
                share_weights=False,
            )
            self.convs.append(conv)

            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())

        graph_embed_dim = 2 * hidden_dim
        fused_dim = graph_embed_dim + global_dim
        self.head = PredictionHead(
            in_dim=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            log_output=log_output,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_attr: Tensor,
    ) -> Tensor:
        """Forward pass for rank prediction.

        Args:
            x: Node features of shape [num_nodes, node_in_dim].
            edge_index: Edge connectivity of shape [2, num_edges].
            edge_attr: Edge features of shape [num_edges, edge_in_dim].
            batch: Batch assignment vector of shape [num_nodes].
            global_attr: Global features of shape [batch_size, global_in_dim].

        Returns:
            Predicted rank of shape [batch_size, 1], guaranteed positive.
        """
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        g = self.global_encoder(global_attr)

        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
            x = self.dropout(x)
            x = x + x_res

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_embed = torch.cat([x_mean, x_max], dim=-1)
        fused = torch.cat([graph_embed, g], dim=-1)

        rank = self.head(fused)

        return rank

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_attr: Tensor,
        min_rank: int = 1,
    ) -> Tensor:
        """Predict integer ranks for inference.

        This is a convenience method that rounds the continuous output to
        integers suitable for use with the LoRADS solver.

        Args:
            x: Node features of shape [num_nodes, node_in_dim].
            edge_index: Edge connectivity of shape [2, num_edges].
            edge_attr: Edge features of shape [num_edges, edge_in_dim].
            batch: Batch assignment vector of shape [num_nodes].
            global_attr: Global features of shape [batch_size, global_in_dim].
            min_rank: Minimum rank to clamp to. Defaults to 1.

        Returns:
            Predicted integer ranks of shape [batch_size, 1], dtype=torch.long.
        """
        continuous_rank = self.forward(x, edge_index, edge_attr, batch, global_attr)
        integer_rank = torch.round(continuous_rank).long().clamp(min=min_rank)
        return integer_rank

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"log_output={self.log_output}, "
            f"params={self.count_parameters():,})"
        )
