"""Model package for Learning to Rank SDP solver.

This package provides the GNN architecture for predicting optimal rank
schedules (trajectories) required to solve SDP problems efficiently.

Main Components:
    - RankSchedulePredictor: Full encoder-decoder model for schedule prediction.
    - GNNEncoder: Graph neural network encoder for constraint graphs.
    - SequenceDecoder: LSTM-based autoregressive decoder.
    - TransformerDecoder: Transformer-based decoder (alternative).

Encoder Components:
    - NodeEncoder: Projects constraint node features to latent space.
    - EdgeEncoder: Projects edge coupling features to latent space.
    - GlobalEncoder: Projects problem-level features to latent space.
    - AttentionPooling: Attention-weighted graph aggregation.
    - MLPBlock: Generic feedforward building block.

Design Choices:
    - Encoder-decoder architecture for variable-length output sequences.
    - GATv2Conv backbone with residual connections for graph encoding.
    - Multi-head pooling (mean, max, attention) for robust aggregation.
    - Log-space prediction for handling wide rank ranges.
    - Teacher forcing support for stable training.

Example:
    >>> from model import RankSchedulePredictor
    >>> model = RankSchedulePredictor(hidden_dim=128, decoder_type="lstm")
    >>> schedule, length_logits = model(
    ...     x, edge_index, edge_attr, batch, global_attr
    ... )
"""

from model.layers import (
    AttentionPooling,
    EdgeEncoder,
    GlobalEncoder,
    MLPBlock,
    NodeEncoder,
    SequenceDecoder,
    TransformerDecoder,
)
from model.net import GNNEncoder, RankPredictor, RankSchedulePredictor

__all__ = [
    # main model
    "RankSchedulePredictor",
    "RankPredictor",  # alias for backward compatibility
    "GNNEncoder",
    # layers
    "NodeEncoder",
    "EdgeEncoder",
    "GlobalEncoder",
    "AttentionPooling",
    "MLPBlock",
    # decoders
    "SequenceDecoder",
    "TransformerDecoder",
]
