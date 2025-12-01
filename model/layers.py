"""\
Module defining various neural network layers for encoding and decoding
features for low-rank SDP problems.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MLPBlock(nn.Module):
    """Generic MLP block with configurable normalization and dropout.

    Structure: Linear -> Norm -> Activation -> [Dropout] -> Linear

    Attributes:
        net: Sequential container holding the MLP layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        norm_type: Literal["batch", "layer", "none"] = "layer",
        activation: Literal["relu", "leaky_relu", "gelu"] = "leaky_relu",
    ) -> None:
        """Initialize the MLP block.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            hidden_dim: Hidden layer dimension. Defaults to out_dim if None
            dropout: Dropout probability after activation
            bias: Whether to use bias in linear layers
            norm_type: Type of normalization ("batch", "layer", or "none")
            activation: Activation function type
        """
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]

        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(hidden_dim))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP."""
        return self.net(x)


class NodeEncoder(nn.Module):
    """Encoder for node features (constraint properties).

    Attributes:
        mlp: Two-layer MLP for feature projection.
        residual: Optional residual projection if dims mismatch.
    """

    def __init__(
        self,
        in_dim: int = 16,
        out_dim: int = 64,
        dropout: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the NodeEncoder.

        Args:
            in_dim: Input node feature dimension
            out_dim: Output latent dimension
            dropout: Dropout probability
            norm_type: Type of normalization
        """
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=out_dim * 2,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode node features.

        Args:
            x: Node features of shape [num_nodes, in_dim]

        Returns:
            Encoded features of shape [num_nodes, out_dim]
        """
        return self.mlp(x)


class EdgeEncoder(nn.Module):
    """Encoder for edge features (constraint coupling).

    Attributes:
        mlp: Two-layer MLP for feature projection.
    """

    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 32,
        dropout: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the EdgeEncoder.

        Args:
            in_dim: Input edge feature dimension
            out_dim: Output latent dimension
            dropout: Dropout probability
            norm_type: Type of normalization
        """
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=out_dim * 2,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        """Encode edge features.

        Args:
            edge_attr: Edge features of shape [num_edges, in_dim]

        Returns:
            Encoded features of shape [num_edges, out_dim]
        """
        return self.mlp(edge_attr)


class GlobalEncoder(nn.Module):
    """Encoder for global graph features (problem-level properties).

    Attributes:
        mlp: Two-layer MLP for feature projection
    """

    def __init__(
        self,
        in_dim: int = 17,
        out_dim: int = 64,
        dropout: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the GlobalEncoder.

        Args:
            in_dim: Input global feature dimension (17 from processor)
            out_dim: Output latent dimension
            dropout: Dropout probability
            norm_type: Type of normalization
        """
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=out_dim * 2,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, global_attr: Tensor) -> Tensor:
        """Encode global features.

        Args:
            global_attr: Global features of shape [batch_size, in_dim].

        Returns:
            Encoded features of shape [batch_size, out_dim].
        """
        return self.mlp(global_attr)


class AttentionPooling(nn.Module):
    """Attention-based graph pooling for variable-size graphs.

    Attributes:
        attention: Linear layer for computing attention logits
        transform: Optional transformation after pooling
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """Initialize AttentionPooling.

        Args:
            in_dim: Input node embedding dimension
            hidden_dim: Hidden dimension for attention computation
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """Compute attention-weighted graph pooling.

        Args:
            x: Node embeddings of shape [num_nodes, in_dim]
            batch: Batch assignment vector of shape [num_nodes]

        Returns:
            Graph embeddings of shape [batch_size, in_dim]
        """
        attn_logits = self.attention(x).squeeze(-1)
        batch_size = batch.max().item() + 1

        attn_logits_fp32 = attn_logits.float()
        attn_max = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        attn_max.scatter_reduce_(
            0, batch, attn_logits_fp32, reduce="amax", include_self=False
        )
        attn_logits_fp32 = attn_logits_fp32 - attn_max[batch]
        attn_exp = torch.exp(attn_logits_fp32)

        attn_sum = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        attn_sum.scatter_add_(0, batch, attn_exp)
        attn_weights = attn_exp / (attn_sum[batch] + 1e-8)

        attn_weights = attn_weights.to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        weighted_x = x * attn_weights.unsqueeze(-1)
        output = torch.zeros(batch_size, x.size(-1), device=x.device, dtype=x.dtype)
        output.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted_x), weighted_x)

        return output


class SequenceDecoder(nn.Module):
    """Autoregressive LSTM decoder for rank schedule prediction.

    Takes a context embedding and generates a sequence of rank predictions.
    Uses teacher forcing during training and autoregressive generation during inference.

    Attributes:
        embed_rank: Embedding for input ranks (during autoregressive decoding)
        lstm: LSTM for sequence modeling
        output_proj: Projects LSTM hidden state to rank prediction
        context_proj: Projects context to initial hidden state
        max_seq_len: Maximum sequence length to generate
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 16,
        min_rank: float = 1.0,
    ) -> None:
        """Initialize SequenceDecoder.

        Args:
            context_dim: Dimension of the context embedding from encoder
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for generation
            min_rank: Minimum rank value enforced through the initial prior
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.min_rank = min_rank

        self.embed_rank = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.context_to_h = nn.Linear(context_dim, hidden_dim * num_layers)
        self.context_to_c = nn.Linear(context_dim, hidden_dim * num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.length_predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, max_seq_len),
        )

        self.initial_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _init_hidden(self, context: Tensor) -> Tuple[Tensor, Tensor]:
        """Initialize LSTM hidden state from context.

        Args:
            context: Context embedding of shape [batch_size, context_dim]

        Returns:
            Tuple of (h_0, c_0) each of shape [num_layers, batch_size, hidden_dim]
        """
        batch_size = context.size(0)

        h = self.context_to_h(context)
        c = self.context_to_c(context)

        h = h.view(batch_size, self.num_layers, self.hidden_dim)
        c = c.view(batch_size, self.num_layers, self.hidden_dim)

        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        return h, c

    def forward(
        self,
        context: Tensor,
        target_schedule: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        use_target_init: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass with optional teacher forcing.

        Args:
            context: Context embedding of shape [batch_size, context_dim]
            target_schedule: Target rank schedule [batch_size, max_seq_len]
                Used for teacher forcing during training
            target_mask: Binary mask [batch_size, max_seq_len] indicating
                valid positions in target_schedule
            teacher_forcing_ratio: Probability of using ground truth as
                next input during training. 0.0 = fully autoregressive
            use_target_init: If True and target_schedule is provided, seed the
                decoder with the ground-truth initial rank instead of the prior

        Returns:
            Tuple of:
                - predictions: Rank predictions [batch_size, max_seq_len]
                - length_logits: Logits for sequence length [batch_size, max_seq_len]
                - init_rank: Prior initial rank prediction [batch_size, 1]
        """
        batch_size = context.size(0)
        device = context.device

        length_logits = self.length_predictor(context)

        h, c = self._init_hidden(context)

        init_rank = torch.nn.functional.softplus(self.initial_head(context)) + (
            self.min_rank
        )
        if use_target_init and target_schedule is not None:
            current_input = target_schedule[:, :1]
        else:
            current_input = init_rank

        predictions = []
        for t in range(self.max_seq_len):
            embedded = self.embed_rank(current_input.unsqueeze(-1))
            output, (h, c) = self.lstm(embedded, (h, c))

            log_rank = self.output_proj(output.squeeze(1))
            log_rank = torch.clamp(log_rank, min=-2.0, max=10.0)
            rank_pred = torch.exp(log_rank)

            predictions.append(rank_pred)

            if (
                target_schedule is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                current_input = target_schedule[:, t : t + 1]
            else:
                current_input = rank_pred.detach()

        predictions = torch.cat(predictions, dim=-1)

        return predictions, length_logits, init_rank

    @torch.no_grad()
    def generate(
        self,
        context: Tensor,
        min_rank: int = 1,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate rank schedule autoregressively.

        Args:
            context: Context embedding of shape [batch_size, context_dim]
            min_rank: Minimum rank value to clamp predictions
            temperature: Scaling factor for predictions (not used for regression)

        Returns:
            Tuple of:
                - schedule: Generated rank schedule [batch_size, predicted_length]
                - lengths: Predicted sequence lengths [batch_size]
                - init_rank: Initial rank prior [batch_size, 1]
        """
        batch_size = context.size(0)
        device = context.device

        length_logits = self.length_predictor(context)
        lengths = torch.argmax(length_logits, dim=-1) + 1
        lengths = lengths.clamp(min=1, max=self.max_seq_len)

        h, c = self._init_hidden(context)

        init_rank = torch.nn.functional.softplus(self.initial_head(context)) + (
            self.min_rank
        )
        current_input = init_rank

        predictions = []
        for t in range(self.max_seq_len):
            embedded = self.embed_rank(current_input.unsqueeze(-1))
            output, (h, c) = self.lstm(embedded, (h, c))
            log_rank = self.output_proj(output.squeeze(1))
            log_rank = torch.clamp(log_rank, min=-2.0, max=10.0)
            rank_pred = torch.exp(log_rank)
            predictions.append(rank_pred)
            current_input = rank_pred

        schedule = torch.cat(predictions, dim=-1)
        schedule = schedule.clamp(min=min_rank)

        return schedule, lengths, init_rank
