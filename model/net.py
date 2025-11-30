"""\
Model architecture for predicting rank schedules/trajectories.

Example usage:
    >>> from model.net import RankSchedulePredictor
    >>> model = RankSchedulePredictor()
    >>> schedule, lengths, length_logits = model(
    ...     x, edge_index, edge_attr, batch, global_attr
    ... )
    >>> print(schedule.shape)
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from model.layers import (
    AttentionPooling,
    EdgeEncoder,
    GlobalEncoder,
    NodeEncoder,
    SequenceDecoder,
    TransformerDecoder,
)


class GNNEncoder(nn.Module):
    """GNN encoder for SDP problem instances.

    Processes the constraint graph using GATv2Conv layers with edge features,
    then aggregates node embeddings into a fixed-size graph representation.

    Attributes:
        node_encoder: Projects raw node features
        edge_encoder: Projects raw edge features
        global_encoder: Projects global features
        convs: ModuleList of GATv2Conv layers
        norms: ModuleList of normalization layers
        attn_pool: Attention-based pooling
    """

    def __init__(
        self,
        node_in_dim: int = 16,
        edge_in_dim: int = 5,
        global_in_dim: int = 17,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        global_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the GNN encoder.

        Args:
            node_in_dim: Input dimension for node features
            edge_in_dim: Input dimension for edge features
            global_in_dim: Input dimension for global features
            hidden_dim: Hidden dimension for node embeddings
            edge_dim: Latent dimension for edge embeddings
            global_dim: Latent dimension for global embeddings
            num_layers: Number of GATv2Conv layers
            num_heads: Number of attention heads in GATv2Conv
            dropout: Dropout probability
            norm_type: Type of normalization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.global_dim = global_dim
        self.num_layers = num_layers
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
        for _ in range(num_layers):
            out_channels = hidden_dim // num_heads
            conv = GATv2Conv(
                in_channels=hidden_dim,
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

        self.attn_pool = AttentionPooling(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
        )
        self.output_dim = 3 * hidden_dim + global_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_attr: Tensor,
    ) -> Tensor:
        """Forward pass for graph embedding.

        Args:
            x: Node features [num_nodes, node_in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_in_dim]
            batch: Batch assignment [num_nodes]
            global_attr: Global features [batch_size, global_in_dim]

        Returns:
            Graph embedding [batch_size, output_dim]
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
        x_attn = self.attn_pool(x, batch)

        graph_embed = torch.cat([x_mean, x_max, x_attn, g], dim=-1)

        return graph_embed


class RankSchedulePredictor(nn.Module):
    """Model for predicting rank schedules from SDP problem graphs.

    Combines GNN encoder with a sequence decoder to predict variable-length
    rank schedules. The model predicts both:
    - rank values at each step of the schedule
    - length of the schedule (number of unique rank values)

    Attributes:
        encoder: GNN encoder for graph processing
        decoder: Sequence decoder for rank schedule generation
        decoder_type: Type of decoder ("lstm" or "transformer")
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        node_in_dim: int = 16,
        edge_in_dim: int = 5,
        global_in_dim: int = 17,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        global_dim: int = 64,
        num_gnn_layers: int = 4,
        num_heads: int = 4,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        decoder_type: Literal["lstm", "transformer"] = "lstm",
        max_seq_len: int = 16,
        dropout: float = 0.1,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the RankSchedulePredictor.

        Args:
            node_in_dim: Input dimension for node features
            edge_in_dim: Input dimension for edge features
            global_in_dim: Input dimension for global features
            hidden_dim: Hidden dimension for GNN layers
            edge_dim: Latent dimension for edge embeddings
            global_dim: Latent dimension for global embeddings
            num_gnn_layers: Number of GATv2Conv layers
            num_heads: Number of attention heads in GATv2Conv
            decoder_hidden_dim: Hidden dimension for sequence decoder
            decoder_num_layers: Number of decoder layers
            decoder_type: Type of decoder ("lstm" or "transformer")
            max_seq_len: Maximum sequence length for rank schedules
            dropout: Dropout probability
            norm_type: Type of normalization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type
        self.max_seq_len = max_seq_len

        self.encoder = GNNEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            global_in_dim=global_in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            norm_type=norm_type,
        )

        context_dim = self.encoder.output_dim
        if decoder_type == "lstm":
            self.decoder = SequenceDecoder(
                context_dim=context_dim,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_num_layers,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
        else:
            self.decoder = TransformerDecoder(
                context_dim=context_dim,
                hidden_dim=decoder_hidden_dim,
                num_heads=num_heads,
                num_layers=decoder_num_layers,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_attr: Tensor,
        target_schedule: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for training.

        Args:
            x: Node features [num_nodes, node_in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_in_dim]
            batch: Batch assignment [num_nodes]
            global_attr: Global features [batch_size, global_in_dim]
            target_schedule: Target rank schedule [batch_size, max_seq_len]
            target_mask: Binary mask for valid positions
            teacher_forcing_ratio: Probability of teacher forcing

        Returns:
            Tuple of:
                - predictions: Rank predictions [batch_size, max_seq_len]
                - length_logits: Logits for sequence length [batch_size, max_seq_len]
        """

        context = self.encoder(x, edge_index, edge_attr, batch, global_attr)

        predictions, length_logits = self.decoder(
            context,
            target_schedule=target_schedule,
            target_mask=target_mask,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return predictions, length_logits

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_attr: Tensor,
        min_rank: int = 1,
        return_integers: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Generate rank schedule for inference.

        Args:
            x: Node features [num_nodes, node_in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_in_dim]
            batch: Batch assignment [num_nodes]
            global_attr: Global features [batch_size, global_in_dim]
            min_rank: Minimum rank to clamp predictions
            return_integers: If True, round predictions to integers

        Returns:
            Tuple of:
                - schedule: Predicted rank schedule [batch_size, max_seq_len]
                - lengths: Predicted sequence lengths [batch_size]
        """
        self.eval()

        context = self.encoder(x, edge_index, edge_attr, batch, global_attr)
        schedule, lengths = self.decoder.generate(context, min_rank=min_rank)
        if return_integers:
            schedule = torch.round(schedule).long().clamp(min=min_rank)

        return schedule, lengths

    def get_valid_schedule(
        self,
        schedule: Tensor,
        lengths: Tensor,
    ) -> list:
        """Extract valid rank schedules (without padding) as Python lists.

        Args:
            schedule: Full schedule tensor [batch_size, max_seq_len]
            lengths: Sequence lengths [batch_size]

        Returns:
            List of lists containing valid rank values for each sample
        """
        batch_size = schedule.size(0)
        result = []

        for i in range(batch_size):
            length = lengths[i].item()
            valid_ranks = schedule[i, :length].tolist()
            result.append(valid_ranks)

        return result

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"decoder_type={self.decoder_type}, "
            f"max_seq_len={self.max_seq_len}, "
            f"params={self.count_parameters():,})"
        )


RankPredictor = RankSchedulePredictor
