"""Model package for Learning to Rank SDP solver.

This package provides the GNN architecture for predicting the optimal rank
(r_oracle) required to solve SDP problems efficiently.

Main Components:
    - RankPredictor: Graph-to-scalar GNN for rank prediction.
    - NodeEncoder: Encoder for constraint node features.
    - EdgeEncoder: Encoder for constraint coupling edge features.
    - GlobalEncoder: Encoder for problem-level global features.
    - PredictionHead: MLP head with log-space output for rank prediction.
    - MLPBlock: Generic MLP building block.

Design Choices:
    - LayerNorm (default) over BatchNorm for stability with variable graph sizes.
    - Log-space prediction (default) to handle extreme label ranges (1 to 8000+).
    - GATv2Conv backbone with residual connections for expressive power.

Example:
    >>> from model import RankPredictor
    >>> model = RankPredictor(hidden_dim=64, num_layers=4, log_output=True)
    >>> rank = model(x, edge_index, edge_attr, batch, global_attr)
"""

from model.layers import (
    EdgeEncoder,
    GlobalEncoder,
    MLPBlock,
    NodeEncoder,
    PredictionHead,
)
from model.net import RankPredictor

__all__ = [
    "RankPredictor",
    "NodeEncoder",
    "EdgeEncoder",
    "GlobalEncoder",
    "PredictionHead",
    "MLPBlock",
]
