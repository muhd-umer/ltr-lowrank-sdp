"""\
This module contains a PyG Dataset class and utilities
to create train/val/test loaders for task at hand.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class SDPDataset(Dataset):
    """Custom PyTorch Geometric Dataset for SDP problem instances.

    This dataset loads graph representations of SDP problems (.pt files) and
    their corresponding solver labels (.json files). A sample is valid only if
    both the .pt graph file and the .json label file exist.

    Attributes:
        root: Root directory containing 'proc/' and 'sol_json/' subdirectories.
        valid_names: List of problem names that have both .pt and .json files.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Initialize the SDPDataset.

        Args:
            root: Root directory path (should contain 'proc/' and 'sol_json/').
            transform: Optional transform to apply to each data object.
            pre_transform: Optional pre-transform (not used, for API compat).
            pre_filter: Optional pre-filter (not used, for API compatibility).
        """
        self._root = Path(root)
        self.proc_dir = self._root / "proc"
        self.sol_dir = self._root / "sol_json"
        self.valid_names = self._find_valid_samples()

        self._indices = None
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

    def _find_valid_samples(self) -> List[str]:
        """Find problem names that have both .pt and .json files.

        Prioritizes .pt files as the base, since some instances may have
        .json files but missing .pt files.

        Returns:
            Sorted list of valid problem names (without extensions).
        """
        pt_files = set()
        if self.proc_dir.exists():
            for f in self.proc_dir.glob("*.pt"):
                pt_files.add(f.stem)

        json_files = set()
        if self.sol_dir.exists():
            for f in self.sol_dir.glob("*.json"):
                json_files.add(f.stem)
        valid = pt_files & json_files

        return sorted(list(valid))

    def len(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.valid_names)

    def get(self, idx: int) -> Data:
        """Load a single data sample by index.

        Args:
            idx: Index of the sample to load.

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, 8]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 2]
                - global_attr: Global graph features [5]
                - y: Oracle rank label (torch.float32, shape [1])
                - problem_id: Problem name string
                - solver_time: Solve time in seconds (float)
        """
        name = self.valid_names[idx]

        pt_path = self.proc_dir / f"{name}.pt"
        data = torch.load(pt_path, weights_only=False)

        json_path = self.sol_dir / f"{name}.json"
        with open(json_path, "r") as f:
            label_data = json.load(f)

        metrics = label_data.get("final_metrics", label_data.get("metrics", {}))
        oracle_rank = metrics.get("oracle_rank", None)
        if oracle_rank is None:
            raise ValueError(f"missing oracle_rank in {json_path}")

        data.y = torch.tensor([float(oracle_rank)], dtype=torch.float32)
        data.problem_id = name
        data.solver_time = float(metrics.get("solve_time_sec", 0.0))

        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def num_node_features(self) -> int:
        """Return the number of node features (expected to be 8)."""
        return 8

    @property
    def num_edge_features(self) -> int:
        """Return the number of edge features (expected to be 2)."""
        return 2


def create_dataloaders(
    root: str,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    train_split: float = 0.9,
    val_split: float = 0.05,
    test_split: float = 0.05,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with configurable split ratios.

    Args:
        root: Root directory containing 'proc/' and 'sol_json/' subdirectories.
        batch_size: Batch size for all loaders.
        seed: Random seed for reproducible shuffling.
        num_workers: Number of worker processes for data loading.
        train_split: Fraction of data for training (default: 0.9).
        val_split: Fraction of data for validation (default: 0.05).
        test_split: Fraction of data for testing (default: 0.05).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If split ratios don't sum to 1.0 or no valid samples found.
    """
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0, got {total_split:.4f} "
            f"({train_split} + {val_split} + {test_split})"
        )

    dataset = SDPDataset(root=root)
    num_samples = len(dataset)

    if num_samples == 0:
        raise ValueError(f"no valid samples found in {root}")

    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(train_split * num_samples)
    val_end = int((train_split + val_split) * num_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_dataset_statistics(dataset: SDPDataset) -> dict:
    """Compute statistics over the entire dataset.

    Args:
        dataset: The SDPDataset instance.

    Returns:
        Dictionary with total_samples, max_rank, min_rank, mean_rank.
    """
    ranks = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        ranks.append(data.y.item())

    return {
        "total_samples": len(ranks),
        "max_rank": max(ranks) if ranks else 0,
        "min_rank": min(ranks) if ranks else 0,
        "mean_rank": sum(ranks) / len(ranks) if ranks else 0,
    }


if __name__ == "__main__":
    root_dir = Path(__file__).parent

    print("\n[1] initializing dataset...")
    dataset = SDPDataset(root=str(root_dir))
    print(f"    total valid samples: {len(dataset)}")
    print(f"    valid problem names (first 10): {dataset.valid_names[:10]}")

    print("\n[2] creating dataloaders (75%/15%/10% split)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root=str(root_dir),
        batch_size=8,
        seed=42,
    )
    print(f"    train samples: {len(train_loader.dataset)}")
    print(f"    val samples:   {len(val_loader.dataset)}")
    print(f"    test samples:  {len(test_loader.dataset)}")

    print("\n[3] loading one batch from train loader...")
    batch = next(iter(train_loader))
    print(f"    batch type: {type(batch)}")
    print(f"    batch.x.shape:          {batch.x.shape}")
    print(f"    batch.edge_index.shape: {batch.edge_index.shape}")
    print(f"    batch.edge_attr.shape:  {batch.edge_attr.shape}")
    print(f"    batch.y.shape:          {batch.y.shape}")
    print(f"    batch.y (labels):       {batch.y}")
    print(f"    batch.batch.shape:      {batch.batch.shape}")

    if hasattr(batch, "problem_id"):
        print(f"    problem_ids:            {batch.problem_id}")

    print("\n[4] computing statistics...")
    stats = get_dataset_statistics(dataset)
    print(f"    total samples: {stats['total_samples']}")
    print(f"    max rank:      {stats['max_rank']:.0f}")
    print(f"    min rank:      {stats['min_rank']:.0f}")
    print(f"    mean rank:     {stats['mean_rank']:.2f}")

    print("\n[5] running checks...")
    expected_node_features = 8
    actual_node_features = batch.x.shape[1]
    assert actual_node_features == expected_node_features, (
        f"node feature dimension mismatch: expected {expected_node_features}, "
        f"got {actual_node_features}"
    )
    print(f"    [p] node features have expected dimension ({expected_node_features})")

    assert not torch.isnan(batch.y).any(), "labels contain NaN values"
    print("    [p] labels do not contain NaN values")

    assert (batch.y > 0).all(), "labels contain non-positive values"
    print("    [p] labels are positive (valid ranks)")

    assert batch.edge_index.shape[0] == 2, "edge_index should have 2 rows"
    print("    [p] edge_index has correct shape")

    print("\n[6] individual sample loading...")
    for i in range(min(3, len(dataset))):
        data = dataset.get(i)
        assert data.x is not None, f"sample {i}: x is None"
        assert data.y is not None, f"sample {i}: y is None"
        assert data.x.shape[1] == 8, f"sample {i}: wrong node feature dim"
        assert not torch.isnan(data.y).any(), f"sample {i}: y contains NaN"
        print(
            f"    [p] sample '{data.problem_id}': x={data.x.shape}, y={data.y.item():.0f}"
        )
