"""\
This module provides modular encoder layers and MLP blocks used to project
raw SDP problem features into latent representations for oracle rank prediction.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Generic MLP block with configurable normalization and dropout.

    Structure: Linear -> Norm -> LeakyReLU -> [Dropout] -> Linear

    This is the fundamental building block used by all encoders. The two-layer
    design provides sufficient capacity for feature transformation while
    maintaining computational efficiency.

    Note:
        LayerNorm is preferred for GNN encoders as it normalizes per-sample
        rather than per-batch, making it more stable for variable-size graphs.
        BatchNorm can still be used when batch statistics are reliable.

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
    ) -> None:
        """Initialize the MLP block.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            hidden_dim: Hidden layer dimension. Defaults to out_dim if None.
            dropout: Dropout probability after activation. Defaults to 0.0.
            bias: Whether to use bias in linear layers. Defaults to True.
            norm_type: Type of normalization ("batch", "layer", or "none").
                Defaults to "layer" for stability with variable graph sizes.
        """
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]

        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape [batch, in_dim] or [num_nodes, in_dim].

        Returns:
            Output tensor of shape [batch, out_dim] or [num_nodes, out_dim].
        """
        return self.net(x)


class NodeEncoder(nn.Module):
    """Encoder for node features (constraint properties).

    Projects the 8-dimensional raw node features into a latent space.
    Node features include spectral scale, trace, norm, sparsity, RHS,
    cost overlap, etc.

    Attributes:
        mlp: Two-layer MLP for feature projection.
    """

    def __init__(
        self,
        in_dim: int = 8,
        out_dim: int = 64,
        dropout: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the NodeEncoder.

        Args:
            in_dim: Input node feature dimension. Defaults to 8.
            out_dim: Output latent dimension. Defaults to 64.
            dropout: Dropout probability. Defaults to 0.0.
            norm_type: Type of normalization. Defaults to "layer".
        """
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=out_dim,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode node features.

        Args:
            x: Node features of shape [num_nodes, in_dim].

        Returns:
            Encoded features of shape [num_nodes, out_dim].
        """
        return self.mlp(x)


class EdgeEncoder(nn.Module):
    """Encoder for edge features (constraint coupling).

    Projects the 2-dimensional edge features into a latent space.
    Edge features include Jaccard similarity and scale.

    Attributes:
        mlp: Two-layer MLP for feature projection.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 32,
        dropout: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "layer",
    ) -> None:
        """Initialize the EdgeEncoder.

        Args:
            in_dim: Input edge feature dimension. Defaults to 2.
            out_dim: Output latent dimension. Defaults to 32.
            dropout: Dropout probability. Defaults to 0.0.
            norm_type: Type of normalization. Defaults to "layer".
        """
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=out_dim,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode edge features.

        Args:
            edge_attr: Edge features of shape [num_edges, in_dim].

        Returns:
            Encoded features of shape [num_edges, out_dim].
        """
        return self.mlp(edge_attr)


class GlobalEncoder(nn.Module):
    """Encoder for global graph features (problem-level properties).

    Projects the 5-dimensional global features into a latent space.
    Global features include log(n), log(m), cost scale, etc.

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
        """Initialize the GlobalEncoder.

        Args:
            in_dim: Input global feature dimension. Defaults to 5.
            out_dim: Output latent dimension. Defaults to 32.
            dropout: Dropout probability. Defaults to 0.0.
            norm_type: Type of normalization. Defaults to "layer".
        """
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=out_dim,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, global_attr: torch.Tensor) -> torch.Tensor:
        """Encode global features.

        Args:
            global_attr: Global features of shape [batch_size, in_dim].

        Returns:
            Encoded features of shape [batch_size, out_dim].
        """
        return self.mlp(global_attr)


class PredictionHead(nn.Module):
    """Multi-layer prediction head for final rank regression.

    A 3-layer MLP that maps the fused graph and global embeddings to a
    scalar rank prediction. Supports two output modes:

    1. Direct mode (log_output=False): Uses Softplus to ensure positive output.
       Best when labels have a small, bounded range.

    2. Log-space mode (log_output=True): Predicts log(rank), then exponentiates.
       Best when labels span multiple orders of magnitude (e.g., 1 to 8000).
       Training should use log-transformed labels with this mode.

    Attributes:
        layers: Sequential container of linear and activation layers.
        output: Final linear layer to scalar output.
        log_output: Whether to predict in log-space.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        log_output: bool = True,
    ) -> None:
        """Initialize the PredictionHead.

        Args:
            in_dim: Input dimension (fused embedding size).
            hidden_dim: Hidden layer dimension. Defaults to 64.
            dropout: Dropout probability. Defaults to 0.1.
            log_output: If True, predict log(rank) and exponentiate. This is
                recommended when ranks span multiple orders of magnitude.
                Defaults to True.
        """
        super().__init__()
        self.log_output = log_output

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
        )
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict rank.

        Args:
            x: Fused embedding of shape [batch_size, in_dim].

        Returns:
            Predicted rank of shape [batch_size, 1], guaranteed positive.
            If log_output=True, returns exp(prediction) for positive output.
            If log_output=False, returns softplus(prediction).
        """
        x = self.layers(x)
        x = self.output(x)

        if self.log_output:
            x = torch.clamp(x, max=15.0)
            return torch.exp(x)
        else:
            return torch.nn.functional.softplus(x, beta=1.0)
