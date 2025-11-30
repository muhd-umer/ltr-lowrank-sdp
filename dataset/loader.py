"""\
This module contains a PyG Dataset class and utilities to create
train/val/test loaders for task at hand.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from dataset.processor import NUM_GLOBAL_FEATURES, NUM_NODE_FEATURES, NUM_EDGE_FEATURES


def extract_rank_schedule(trajectory: Dict) -> List[int]:
    """Extracts unique oracle rank schedule from solver trajectory.

    Args:
        trajectory: Dictionary with 'phase_1' and 'phase_2' trajectory data,
            each containing 'oracle_rank' lists

    Returns:
        List of unique consecutive oracle rank values representing the
        rank schedule. Returns empty list if trajectory is malformed
    """
    phase1 = trajectory.get("phase_1", {})
    phase2 = trajectory.get("phase_2", {})

    p1_oracle = phase1.get("oracle_rank", [])
    p2_oracle = phase2.get("oracle_rank", [])

    all_oracle = p1_oracle + p2_oracle

    if not all_oracle:
        return []

    schedule = []
    for rank in all_oracle:
        if not schedule or schedule[-1] != rank:
            schedule.append(int(rank))

    return schedule


def classify_schedule_type(schedule: List[int]) -> str:
    """Classifies the type of rank schedule.

    Args:
        schedule: List of unique consecutive rank values

    Returns:
        One of: 'constant', 'increasing', 'decreasing', 'mixed'
    """
    if len(schedule) <= 1:
        return "constant"

    diffs = [schedule[i + 1] - schedule[i] for i in range(len(schedule) - 1)]

    if all(d >= 0 for d in diffs):
        return "increasing"
    elif all(d <= 0 for d in diffs):
        return "decreasing"
    else:
        return "mixed"


def pad_schedule(
    schedule: List[int],
    max_length: int,
    pad_value: int = 0,
) -> Tuple[List[int], int]:
    """Pads or truncates schedule to fixed length.

    Args:
        schedule: Original rank schedule
        max_length: Target length for padding
        pad_value: Value to use for padding

    Returns:
        Tuple of (padded_schedule, original_length)
    """
    original_length = len(schedule)

    if len(schedule) >= max_length:
        return schedule[:max_length], min(original_length, max_length)

    padded = schedule + [pad_value] * (max_length - len(schedule))
    return padded, original_length


class SDPDataset(Dataset):
    """PyG Dataset for SDP problem instances with rank schedules.

    This dataset loads graph representations of SDP problems (.pt files) and
    their corresponding oracle rank schedules from solver JSON files.

    Features:
        - node features: 16 dimensions per constraint
        - edge features: 5 dimensions per edge
        - global features: 17 dimensions per problem

    Labels:
        - rank_schedule: Sequence of unique oracle ranks (variable length)
        - schedule_length: Number of unique ranks in schedule
        - final_rank: Final oracle rank
        - initial_rank: Initial oracle rank
        - max_rank: Maximum rank in schedule
        - schedule_type: Classification ('constant', 'increasing', 'decreasing', 'mixed')

    Attributes:
        root: Root directory containing 'proc/' and 'sol_json/' subdirectories
        max_schedule_length: Maximum length for padded schedules
        valid_names: List of problem names that have both .pt and .json files
    """

    def __init__(
        self,
        root: str,
        max_schedule_length: int = 16,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Initializes the SDPDataset.

        Args:
            root: Root directory path (should contain 'proc/' and 'sol_json/')
            max_schedule_length: Maximum length for rank schedule sequences
                Schedules longer than this are truncated; shorter ones are padded
            transform: Optional transform to apply to each data object
            pre_transform: Optional pre-transform (not used, for API compat)
            pre_filter: Optional pre-filter (not used, for API compat)
        """
        self._root = Path(root)
        self.proc_dir = self._root / "proc"
        self.sol_dir = self._root / "sol_json"
        self.max_schedule_length = max_schedule_length

        self.valid_names = self._find_valid_samples()

        self._indices = None
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

    def _find_valid_samples(self) -> List[str]:
        """Finds problem names that have both .pt and .json files.

        Returns:
            Sorted list of valid problem names (without extensions)
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
        """Returns the number of valid samples in the dataset."""
        return len(self.valid_names)

    def get(self, idx: int) -> Data:
        """Loads a single data sample by index.

        Args:
            idx: Index of the sample to load

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, 16]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 5]
                - global_attr: Global graph features [17]
                - rank_schedule: Padded oracle rank schedule [max_schedule_length]
                - schedule_length: Original schedule length (scalar)
                - schedule_mask: Binary mask for valid positions [max_schedule_length]
                - final_rank: Final oracle rank (scalar)
                - initial_rank: Initial oracle rank (scalar)
                - max_rank: Maximum rank in schedule (scalar)
                - schedule_type: Schedule classification string
                - problem_id: Problem name string
                - solver_time: Solve time in seconds (float)

        Raises:
            ValueError: If required data is missing from JSON file
        """
        name = self.valid_names[idx]

        pt_path = self.proc_dir / f"{name}.pt"
        data = torch.load(pt_path, weights_only=False)

        json_path = self.sol_dir / f"{name}.json"
        with open(json_path, "r") as f:
            label_data = json.load(f)

        trajectory = label_data.get("trajectory", {})
        schedule = extract_rank_schedule(trajectory)

        if not schedule:

            metrics = label_data.get("final_metrics", label_data.get("metrics", {}))
            final_rank = metrics.get("oracle_rank")
            if final_rank is not None:
                schedule = [int(final_rank)]
            else:
                raise ValueError(f"no valid trajectory or oracle_rank in {json_path}")

        padded_schedule, orig_length = pad_schedule(
            schedule, self.max_schedule_length, pad_value=0
        )
        schedule_type = classify_schedule_type(schedule)

        mask = [1] * min(orig_length, self.max_schedule_length)
        mask += [0] * (self.max_schedule_length - len(mask))

        data.rank_schedule = torch.tensor(padded_schedule, dtype=torch.float32)
        data.schedule_length = torch.tensor([orig_length], dtype=torch.long)
        data.schedule_mask = torch.tensor(mask, dtype=torch.float32)
        data.final_rank = torch.tensor([float(schedule[-1])], dtype=torch.float32)
        data.initial_rank = torch.tensor([float(schedule[0])], dtype=torch.float32)
        data.max_rank = torch.tensor([float(max(schedule))], dtype=torch.float32)
        data.schedule_type = schedule_type

        data.y = data.final_rank.clone()

        data.problem_id = name
        metrics = label_data.get("final_metrics", label_data.get("metrics", {}))
        data.solver_time = float(metrics.get("solve_time_sec", 0.0))

        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def num_node_features(self) -> int:
        """Returns the number of node features."""
        return NUM_NODE_FEATURES

    @property
    def num_edge_features(self) -> int:
        """Returns the number of edge features."""
        return NUM_EDGE_FEATURES

    @property
    def num_global_features(self) -> int:
        """Returns the number of global features."""
        return NUM_GLOBAL_FEATURES


def create_dataloaders(
    root: str,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    train_split: float = 0.9,
    val_split: float = 0.05,
    test_split: float = 0.05,
    max_schedule_length: int = 16,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train/val/test DataLoaders with configurable split ratios.

    Args:
        root: Root directory containing 'proc/' and 'sol_json/' subdirectories
        batch_size: Batch size for all loaders
        seed: Random seed for reproducible shuffling
        num_workers: Number of worker processes for data loading
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        max_schedule_length: Maximum length for padded rank schedules

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        ValueError: If split ratios don't sum to 1.0 or no valid samples found
    """
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0, got {total_split:.4f} "
            f"({train_split} + {val_split} + {test_split})"
        )

    dataset = SDPDataset(root=root, max_schedule_length=max_schedule_length)
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
