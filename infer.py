"""\
This script loads a trained model checkpoint and predicts optimal rank
schedules for SDP problem instances. It compares predictions against
ground truth trajectories from solution JSON files.

Example usage:
    >>> python infer.py --checkpoint logs/best/best_model.pt --input dataset/proc/G1.pt
    >>> python infer.py -c logs/20231128/best_model.pt -i dataset/proc/theta12.pt
    >>> python infer.py -c best_model.pt -i G1  

    >>> python infer.py -c best_model.pt --batch  
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from dataset.loader import extract_rank_schedule, SDPDataset
from model import RankSchedulePredictor


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    config_path: Optional[Path] = None,
) -> RankSchedulePredictor:
    """Load a trained RankSchedulePredictor model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        device: Device to load the model onto
        config_path: Optional path to config.json for model hyperparameters.
            If not provided, will look for config.json in same directory

    Returns:
        Loaded RankSchedulePredictor model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        print(f"loaded config from checkpoint")
    else:

        if config_path is None:
            config_path = checkpoint_path.parent / "config.json"

        model_config = {
            "node_in_dim": 16,
            "edge_in_dim": 5,
            "global_in_dim": 17,
            "hidden_dim": 128,
            "edge_dim": 64,
            "global_dim": 64,
            "num_gnn_layers": 4,
            "num_heads": 4,
            "decoder_hidden_dim": 128,
            "decoder_num_layers": 2,
            "max_seq_len": 16,
            "dropout": 0.1,
        }

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if "model" in config:
                    model_config.update(config["model"])
            print(f"loaded config from: {config_path}")

    model = RankSchedulePredictor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"loaded model from: {checkpoint_path}")
    print(f"  parameters: {model.count_parameters():,}")
    print(f"  max seq len: {model_config.get('max_seq_len', 16)}")

    return model


def load_instance(
    input_path: Path,
    sol_dir: Path,
    device: torch.device,
    max_seq_len: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Load a single SDP problem instance.

    Args:
        input_path: Path to the .pt graph file
        sol_dir: Directory containing solution JSON files
        device: Device to load tensors onto
        max_seq_len: Maximum sequence length for schedules

    Returns:
        Tuple of (x, edge_index, edge_attr, global_attr, label_info) where
        label_info contains the ground truth rank schedule
    """
    data = torch.load(input_path, weights_only=False)

    problem_name = input_path.stem
    json_path = sol_dir / f"{problem_name}.json"

    label_info = {
        "problem_name": problem_name,
        "rank_schedule": None,
        "schedule_length": None,
        "oracle_rank": None,
        "solve_time_sec": None,
    }

    if json_path.exists():
        with open(json_path) as f:
            label_data = json.load(f)

        trajectory = label_data.get("trajectory", {})
        schedule = extract_rank_schedule(trajectory)

        if not schedule:

            metrics = label_data.get("final_metrics", label_data.get("metrics", {}))
            final_rank = metrics.get("oracle_rank")
            if final_rank is not None:
                schedule = [int(final_rank)]

        if schedule:
            label_info["rank_schedule"] = schedule[:max_seq_len]
            label_info["schedule_length"] = len(schedule)
            label_info["oracle_rank"] = schedule[-1] if schedule else None

        metrics = label_data.get("final_metrics", label_data.get("metrics", {}))
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
def predict_schedule(
    model: RankSchedulePredictor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    global_attr: torch.Tensor,
    min_rank: int = 1,
) -> Tuple[List[int], int]:
    """Predict rank schedule for a single instance.

    Args:
        model: Trained RankSchedulePredictor model
        x: Node features [num_nodes, node_dim]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, edge_dim]
        global_attr: Global features [1, global_dim]
        min_rank: Minimum rank value to clamp predictions

    Returns:
        Tuple of (schedule, predicted_length) where schedule is a list
        of integer ranks
    """

    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    schedule_tensor, lengths = model.predict(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        global_attr=global_attr,
        min_rank=min_rank,
        return_integers=True,
    )

    pred_length = lengths[0].item()
    schedule = schedule_tensor[0, :pred_length].tolist()

    return schedule, pred_length


def resolve_input_path(input_arg: str, data_root: Path) -> Path:
    """Resolve input argument to a valid .pt file path.

    Handles multiple input formats:
    - Full path: dataset/proc/G1.pt
    - Relative name: G1.pt
    - Problem name only: G1

    Args:
        input_arg: User-provided input argument
        data_root: Dataset root directory

    Returns:
        Resolved Path to the .pt file

    Raises:
        FileNotFoundError: If the file cannot be found
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
        f"could not find input file: {input_arg}\n" f"tried: {input_path}, {proc_path}"
    )


def compute_schedule_metrics(
    pred_schedule: List[int],
    true_schedule: Optional[List[int]],
) -> Dict[str, float]:
    """Compute metrics comparing predicted and true schedules.

    Args:
        pred_schedule: Predicted rank schedule
        true_schedule: Ground truth rank schedule (may be None)

    Returns:
        Dictionary with various comparison metrics
    """
    metrics = {
        "pred_length": len(pred_schedule),
        "pred_final_rank": pred_schedule[-1] if pred_schedule else 0,
        "pred_initial_rank": pred_schedule[0] if pred_schedule else 0,
        "pred_max_rank": max(pred_schedule) if pred_schedule else 0,
    }

    if true_schedule is not None:
        import math

        metrics["true_length"] = len(true_schedule)
        metrics["true_final_rank"] = true_schedule[-1] if true_schedule else 0
        metrics["true_initial_rank"] = true_schedule[0] if true_schedule else 0
        metrics["true_max_rank"] = max(true_schedule) if true_schedule else 0

        metrics["length_error"] = metrics["pred_length"] - metrics["true_length"]
        metrics["length_match"] = metrics["pred_length"] == metrics["true_length"]

        metrics["final_rank_error"] = (
            metrics["pred_final_rank"] - metrics["true_final_rank"]
        )
        metrics["final_rank_rel_error"] = (
            metrics["final_rank_error"] / metrics["true_final_rank"] * 100
            if metrics["true_final_rank"] > 0
            else 0
        )

        min_len = min(len(pred_schedule), len(true_schedule))
        if min_len > 0:
            errors = [pred_schedule[i] - true_schedule[i] for i in range(min_len)]
            log_errors = [
                math.log(max(pred_schedule[i], 1)) - math.log(max(true_schedule[i], 1))
                for i in range(min_len)
            ]

            metrics["mae"] = sum(abs(e) for e in errors) / min_len
            metrics["log_mae"] = sum(abs(e) for e in log_errors) / min_len
            metrics["mse"] = sum(e**2 for e in errors) / min_len

    return metrics


def print_single_result(
    label_info: Dict,
    pred_schedule: List[int],
    pred_length: int,
    metrics: Dict[str, float],
) -> None:
    """Print inference results for a single instance."""
    print(f"\n[problem: {label_info['problem_name']}]")

    print(f"\n[predicted schedule]")
    print(f"  schedule: {pred_schedule}")
    print(f"  length: {pred_length}")
    print(f"  final rank: {pred_schedule[-1] if pred_schedule else 'n/a'}")

    if label_info["rank_schedule"] is not None:
        print(f"\n[gt schedule]")
        print(f"  schedule: {label_info['rank_schedule']}")
        print(f"  length: {label_info['schedule_length']}")
        print(f"  final rank: {label_info['oracle_rank']}")

        print(f"\n[metrics]")
        print(
            f"  length error: {metrics['length_error']:+d} ({'match' if metrics['length_match'] else 'mismatch'})"
        )
        print(
            f"  final rank error: {metrics['final_rank_error']:+d} ({metrics['final_rank_rel_error']:+.1f}%)"
        )
        print(f"  mae (overlapping positions): {metrics.get('mae', 'n/a'):.2f}")
        print(f"  log mae: {metrics.get('log_mae', 'n/a'):.4f}")
    else:
        print(f"\n[gt] not available")


def run_batch_inference(
    model: RankSchedulePredictor,
    data_root: Path,
    device: torch.device,
    output_path: Optional[Path] = None,
    max_seq_len: int = 16,
    seed: int = 42,
    train_split: float = 0.9,
    val_split: float = 0.05,
    test_split: float = 0.05,
) -> Dict:
    """Run inference on test set instances.

    Args:
        model: Trained RankSchedulePredictor model
        data_root: Dataset root directory
        device: Device to run on
        output_path: Optional path to save results JSON
        max_seq_len: Maximum sequence length
        seed: Random seed for reproducible split (default: 42)
        train_split: Fraction of data for training (default: 0.9)
        val_split: Fraction of data for validation (default: 0.05)
        test_split: Fraction of data for testing (default: 0.05)
    Returns:
        Dictionary with aggregated results and statistics
    """
    proc_dir = data_root / "proc"
    sol_dir = data_root / "sol_json"

    dataset = SDPDataset(root=str(data_root), max_schedule_length=max_seq_len)
    num_samples = len(dataset)

    if num_samples == 0:
        raise ValueError(f"no valid samples found in {data_root}")

    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(train_split * num_samples)
    val_end = int((train_split + val_split) * num_samples)
    test_indices = indices[val_end:]

    test_names = [dataset.valid_names[i] for i in test_indices]
    pt_files = [proc_dir / f"{name}.pt" for name in test_names]

    print(
        f"\nrunning batch inference on {len(pt_files)} test instances (out of {num_samples} total)..."
    )

    results = []
    all_metrics = {
        "length_errors": [],
        "final_rank_errors": [],
        "mae_values": [],
        "log_mae_values": [],
        "length_matches": 0,
    }

    for pt_path in tqdm(pt_files, desc="inference"):
        try:
            x, edge_index, edge_attr, global_attr, label_info = load_instance(
                pt_path, sol_dir, device, max_seq_len
            )

            pred_schedule, pred_length = predict_schedule(
                model, x, edge_index, edge_attr, global_attr
            )

            metrics = compute_schedule_metrics(
                pred_schedule, label_info["rank_schedule"]
            )

            result = {
                "problem": label_info["problem_name"],
                "pred_schedule": pred_schedule,
                "pred_length": pred_length,
                "true_schedule": label_info["rank_schedule"],
                "true_length": label_info["schedule_length"],
                "metrics": metrics,
            }
            results.append(result)

            if label_info["rank_schedule"] is not None:
                all_metrics["length_errors"].append(metrics["length_error"])
                all_metrics["final_rank_errors"].append(metrics["final_rank_error"])
                if "mae" in metrics:
                    all_metrics["mae_values"].append(metrics["mae"])
                if "log_mae" in metrics:
                    all_metrics["log_mae_values"].append(metrics["log_mae"])
                if metrics["length_match"]:
                    all_metrics["length_matches"] += 1

        except Exception as e:
            print(f"error processing {pt_path.name}: {e}")
            continue

    summary = {}
    n = len(results)
    n_with_labels = len(all_metrics["length_errors"])

    summary["total_instances"] = n
    summary["instances_with_labels"] = n_with_labels

    if n_with_labels > 0:
        import statistics

        summary["length_accuracy"] = all_metrics["length_matches"] / n_with_labels
        summary["mean_length_error"] = statistics.mean(all_metrics["length_errors"])
        summary["std_length_error"] = (
            statistics.stdev(all_metrics["length_errors"]) if n_with_labels > 1 else 0
        )

        summary["mean_final_rank_error"] = statistics.mean(
            all_metrics["final_rank_errors"]
        )

        if all_metrics["mae_values"]:
            summary["mean_mae"] = statistics.mean(all_metrics["mae_values"])
        if all_metrics["log_mae_values"]:
            summary["mean_log_mae"] = statistics.mean(all_metrics["log_mae_values"])

    print(f"\n[inference summary]")
    print(f"total instances: {summary['total_instances']}")
    print(f"instances with labels: {summary['instances_with_labels']}")
    if n_with_labels > 0:
        print(f"length accuracy: {summary['length_accuracy']:.2%}")
        print(
            f"mean length error: {summary['mean_length_error']:.2f} ± {summary['std_length_error']:.2f}"
        )
        print(f"mean final rank error: {summary['mean_final_rank_error']:.2f}")
        if "mean_mae" in summary:
            print(f"mean mae: {summary['mean_mae']:.2f}")
        if "mean_log_mae" in summary:
            print(f"mean log mae: {summary['mean_log_mae']:.4f}")

    output_data = {
        "summary": summary,
        "results": results,
    }

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nresults saved to: {output_path}")

    return output_data


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Predict rank schedules for SDP problem instances",
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
        default=None,
        help="Path to input .pt file or problem name (e.g., G1, theta12)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run inference on all instances in the dataset",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root directory (for resolving paths and labels)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: looks in checkpoint directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for batch results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test split (default: 42)",
    )

    args = parser.parse_args()

    if not args.batch and args.input is None:
        parser.error("either --input or --batch must be specified")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    data_root = Path(args.data_root)
    sol_dir = data_root / "sol_json"

    config_path = Path(args.config) if args.config else None

    model = load_model(checkpoint_path, device, config_path)

    if args.batch:

        output_path = Path(args.output) if args.output else None
        run_batch_inference(
            model, data_root, device, output_path, model.max_seq_len, seed=args.seed
        )
    else:

        input_path = resolve_input_path(args.input, data_root)
        print(f"input file: {input_path}")

        x, edge_index, edge_attr, global_attr, label_info = load_instance(
            input_path, sol_dir, device, model.max_seq_len
        )

        print(f"\ninstance info:")
        print(f"  nodes: {x.size(0):,}")
        print(f"  edges: {edge_index.size(1):,}")

        pred_schedule, pred_length = predict_schedule(
            model, x, edge_index, edge_attr, global_attr
        )

        metrics = compute_schedule_metrics(pred_schedule, label_info["rank_schedule"])

        print_single_result(label_info, pred_schedule, pred_length, metrics)


if __name__ == "__main__":
    main()
