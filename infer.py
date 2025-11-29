"""\
Inference script for RankPredictor model.

This script loads a trained model checkpoint and predicts the optimal rank
for a given SDP problem instance. It compares the prediction against the
true oracle rank from the solution JSON file.

Example usage:
    >>> python infer.py --checkpoint logs/best/best_model.pt --input dataset/proc/G1.pt
    >>> python infer.py -c logs/20231128/best_model.pt -i dataset/proc/theta12.pt
    >>> python infer.py -c best_model.pt -i G1  # auto-resolves paths
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from model import RankPredictor


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    config_path: Optional[Path] = None,
) -> RankPredictor:
    """Load a trained RankPredictor model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file).
        device: Device to load the model onto.
        config_path: Optional path to config.json for model hyperparameters.
            If not provided, will look for config.json in same directory.

    Returns:
        Loaded RankPredictor model in eval mode.
    """

    if config_path is None:
        config_path = checkpoint_path.parent / "config.json"

    model_config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.2,
    }

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if "model" in config:
                model_config.update(config["model"])
        print(f"loaded config from: {config_path}")

    model = RankPredictor(
        node_in_dim=8,
        edge_in_dim=2,
        global_in_dim=5,
        hidden_dim=model_config["hidden_dim"],
        edge_dim=model_config["hidden_dim"] // 2,
        global_dim=model_config["hidden_dim"] // 2,
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        log_output=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"loaded model from: {checkpoint_path}")
    print(f"  parameters: {model.count_parameters():,}")

    return model


def load_instance(
    input_path: Path,
    sol_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Load a single SDP problem instance.

    Args:
        input_path: Path to the .pt graph file.
        sol_dir: Directory containing solution JSON files.
        device: Device to load tensors onto.

    Returns:
        Tuple of (x, edge_index, edge_attr, global_attr, label_info) where
        label_info contains oracle_rank and other metadata.
    """

    data = torch.load(input_path, weights_only=False)

    problem_name = input_path.stem
    json_path = sol_dir / f"{problem_name}.json"

    label_info = {
        "problem_name": problem_name,
        "oracle_rank": None,
        "solve_time_sec": None,
    }

    if json_path.exists():
        with open(json_path) as f:
            label_data = json.load(f)
        metrics = label_data.get("final_metrics", label_data.get("metrics", {}))
        label_info["oracle_rank"] = metrics.get("oracle_rank")
        label_info["solve_time_sec"] = metrics.get("solve_time_sec")
    else:
        print(f"warning: no label file found at {json_path}")

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    global_attr = data.global_attr.to(device)

    if global_attr.dim() == 1:
        global_attr = global_attr.unsqueeze(0)

    return x, edge_index, edge_attr, global_attr, label_info


@torch.no_grad()
def predict(
    model: RankPredictor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    global_attr: torch.Tensor,
) -> Tuple[float, int]:
    """Run inference on a single instance.

    Args:
        model: Trained RankPredictor model.
        x: Node features [num_nodes, 8].
        edge_index: Edge connectivity [2, num_edges].
        edge_attr: Edge features [num_edges, 2].
        global_attr: Global features [1, 5].

    Returns:
        Tuple of (continuous_prediction, integer_prediction).
    """

    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    pred = model(x, edge_index, edge_attr, batch, global_attr)

    continuous = pred.item()
    integer = max(1, round(continuous))

    return continuous, integer


def resolve_input_path(input_arg: str, data_root: Path) -> Path:
    """Resolve input argument to a valid .pt file path.

    Handles multiple input formats:
    - Full path: dataset/proc/G1.pt
    - Relative name: G1.pt
    - Problem name only: G1

    Args:
        input_arg: User-provided input argument.
        data_root: Dataset root directory.

    Returns:
        Resolved Path to the .pt file.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    input_path = Path(input_arg)

    if input_path.exists():
        return input_path

    if not input_path.suffix:
        input_path = input_path.with_suffix(".pt")

    if input_path.exists():
        return input_path

    proc_path = data_root / "proc" / input_path.name
    if proc_path.exists():
        return proc_path

    if not proc_path.suffix == ".pt":
        proc_path = proc_path.with_suffix(".pt")
    if proc_path.exists():
        return proc_path

    raise FileNotFoundError(
        f"Could not find input file: {input_arg}\n" f"Tried: {input_path}, {proc_path}"
    )


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Predict optimal rank for SDP problem instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input .pt file or problem name (e.g., G1, theta12)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root directory (for resolving input paths and labels)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: looks in checkpoint directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_root = Path(args.data_root)
    sol_dir = data_root / "sol_json"

    config_path = Path(args.config) if args.config else None

    input_path = resolve_input_path(args.input, data_root)
    print(f"input file: {input_path}")

    model = load_model(checkpoint_path, device, config_path)

    x, edge_index, edge_attr, global_attr, label_info = load_instance(
        input_path, sol_dir, device
    )

    print(f"\nproblem: {label_info['problem_name']}")
    print(f"  nodes: {x.size(0):,}")
    print(f"  edges: {edge_index.size(1):,}")

    continuous_pred, integer_pred = predict(
        model, x, edge_index, edge_attr, global_attr
    )

    print("\n[results]")
    print(f"  predicted rank (continuous): {continuous_pred:.2f}")
    print(f"  predicted rank (integer): {integer_pred}")

    if label_info["oracle_rank"] is not None:
        oracle = label_info["oracle_rank"]
        print(f"  true oracle rank: {oracle}")

        error = integer_pred - oracle
        rel_error = error / oracle * 100 if oracle > 0 else 0

        print(f"  absolute error: {error:+d}")
        print(f"  relative error: {rel_error:+.1f}%")
    else:
        print(f"  true oracle rank: (not available)")

    if label_info["solve_time_sec"] is not None:
        print(f"  original solve time: {label_info['solve_time_sec']:.2f}s")


if __name__ == "__main__":
    main()
