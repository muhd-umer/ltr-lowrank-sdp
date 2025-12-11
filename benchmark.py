"""This script runs the LoRADS solver with and without predicted rank schedules
and compares the solving times.

Example usage:
    >>> python benchmark.py  # run all subtypes and all instances (default)
    >>> python benchmark.py --no-all-subtypes --subtype gset  # run all instances in gset
    >>> python benchmark.py --no-all-subtypes --subtype sdplib --no-all-instances --instance theta1
    >>> python benchmark.py --list  # list all available instances
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tabulate import tabulate

from model import RankSchedulePredictor

LORADS_EXECUTABLE = "./lorads/src/build/LoRADS_v_2_0_1-alpha"
BENCHMARK_DIR = Path("benchmark")
INSTANCES_DIR = BENCHMARK_DIR / "instances"
PT_DIR = BENCHMARK_DIR / "pt"
SOL_JSON_DIR = Path("dataset/sol_json")
CHECKPOINT_PATH = Path("ckpts/b_all_t/best_model.pt")
DEFAULT_TIMEOUT = 300
NEAR_STALL_FACTOR = 0.7

SUBTYPES = ["gset", "hansmittel", "matcomp", "maxcut", "sdplib"]
_config = {"timeout": DEFAULT_TIMEOUT}


def load_model(checkpoint_path: Path, device: torch.device) -> RankSchedulePredictor:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: path to model checkpoint
        device: device to load model onto

    Returns:
        loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    else:
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
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if "model" in config:
                    model_config.update(config["model"])

    model = RankSchedulePredictor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_rank_schedule(
    model: RankSchedulePredictor,
    pt_path: Path,
    device: torch.device,
) -> List[int]:
    """Predict rank schedule for a single instance.

    Args:
        model: trained model
        pt_path: path to processed .pt file
        device: device for inference

    Returns:
        list of predicted ranks
    """
    data = torch.load(pt_path, weights_only=False)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    global_attr = data.global_attr.to(device)

    if global_attr.dim() == 1:
        global_attr = global_attr.unsqueeze(0)

    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    schedule_tensor, lengths = model.predict(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        global_attr=global_attr,
        min_rank=1,
        return_integers=True,
    )

    pred_length = lengths[0].item()
    schedule = schedule_tensor[0, :pred_length].tolist()
    return schedule


def save_rank_schedule(schedule: List[int], output_path: Path) -> None:
    """Save rank schedule in JSON format for LoRADS.

    Args:
        schedule: list of rank values
        output_path: path to save json file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rank_schedule": schedule, "schedule_length": len(schedule)}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def get_lorads_params(problem_name: str, subtype: str) -> Dict[str, str]:
    """Get LoRADS parameters based on problem type.

    Args:
        problem_name: name of the problem instance
        subtype: problem subtype (gset, hansmittel, matcomp, maxcut, sdplib)

    Returns:
        dict of parameter name to value
    """
    params = {
        "phase1Tol": "1e-3",
        "heuristicFactor": "1.0",
        "rhoMax": "5000.0",
        "timeSecLimit": str(_config["timeout"]),
        "reoptLevel": "0",
    }

    name_lower = problem_name.lower()

    if subtype == "gset":
        params["phase1Tol"] = "1e-2"
        params["heuristicFactor"] = "10.0"

    elif subtype == "maxcut":
        is_large_maxcut = any(
            [
                name_lower.startswith("delaunay_n")
                and _extract_delaunay_n(name_lower) >= 14,
                name_lower.startswith("rgg_n_2_"),
                name_lower.startswith("amazon"),
                name_lower.startswith("cit-"),
                name_lower.startswith("fe_"),
                name_lower.startswith("p2p-"),
                name_lower.startswith("vsp_"),
                name_lower.startswith("cs"),
                name_lower.endswith("a") and name_lower[:-1].isdigit(),
            ]
        )

        if is_large_maxcut:
            params["phase1Tol"] = "1e+1"
            params["heuristicFactor"] = "100.0"
            params["timesLogRank"] = "0.25"
        else:
            params["phase1Tol"] = "1e-2"
            params["heuristicFactor"] = "10.0"

    elif subtype == "matcomp":
        if name_lower.startswith("mc_"):
            mc_n_str = name_lower.replace("mc_", "").split("_")[0]
            try:
                mc_n = int(mc_n_str)
                if mc_n >= 10000:
                    params["heuristicFactor"] = "10.0"
                    params["timesLogRank"] = "1.0"
                elif mc_n >= 1000:
                    params["heuristicFactor"] = "10.0"
            except ValueError:
                pass

        elif name_lower.startswith("mc_n"):
            params["heuristicFactor"] = "10.0"

    elif subtype == "sdplib":

        if name_lower.startswith("mcp"):
            params["phase1Tol"] = "1e-2"
            params["heuristicFactor"] = "10.0"

    return params


def _extract_delaunay_n(name: str) -> int:
    """Extract n value from delaunay_nX pattern."""
    import re

    match = re.search(r"delaunay_n(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return 0


def run_lorads(
    instance_path: Path,
    json_output_path: Path,
    params: Dict[str, str],
    fixed_rank: Optional[int] = None,
    rank_schedule_path: Optional[Path] = None,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """Run LoRADS solver on an instance.

    Args:
        instance_path: path to .dat-s file
        json_output_path: path for output json file
        params: solver parameters
        fixed_rank: optional fixed rank (mutually exclusive with rank_schedule_path)
        rank_schedule_path: optional path to rank schedule json (mutually exclusive with fixed_rank)

    Returns:
        tuple of (success, solve_time_sec, objective)
    """
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [LORADS_EXECUTABLE, str(instance_path)]

    for param_name, param_value in params.items():
        cmd.extend([f"--{param_name}", param_value])

    cmd.extend(["--jsonfile", str(json_output_path)])
    cmd.extend(["--disableOracle"])

    if fixed_rank is not None and rank_schedule_path is not None:
        raise ValueError("fixed_rank and rank_schedule_path are mutually exclusive")
    if rank_schedule_path is not None:
        cmd.extend(["--rankSchedule", str(rank_schedule_path)])
        cmd.extend(["--nearStallFactor", str(NEAR_STALL_FACTOR)])
    elif fixed_rank is not None:
        cmd.extend(["--fixedRank", str(fixed_rank)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_config["timeout"] + 60,
        )

        if result.returncode != 0:
            print(f"  [error] lorads failed: {result.stderr[:200]}")
            return False, None, None

        if json_output_path.exists():
            with open(json_output_path) as f:
                output_data = json.load(f)

            metrics = output_data.get("final_metrics", output_data.get("metrics", {}))
            solve_time = metrics.get("solve_time_sec")
            objective = metrics.get("primal_obj", metrics.get("primal_objective"))

            return True, solve_time, objective

        return False, None, None

    except subprocess.TimeoutExpired:
        print(f"  [error] lorads timed out")
        return False, None, None
    except Exception as e:
        print(f"  [error] unexpected error: {e}")
        return False, None, None


def find_matching_files(
    subtype: str,
    instance_name: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find matching instance and pt files.

    Args:
        subtype: problem subtype
        instance_name: problem name (without extension)

    Returns:
        tuple of (instance_path, pt_path) or (None, None) if not found
    """
    instance_dir = INSTANCES_DIR / subtype
    pt_dir = PT_DIR / subtype

    instance_path = instance_dir / f"{instance_name}.dat-s"
    pt_path = pt_dir / f"{instance_name}.pt"

    if instance_path.exists() and pt_path.exists():
        return instance_path, pt_path

    for dat_file in instance_dir.glob("*.dat-s"):
        if dat_file.stem.lower() == instance_name.lower():
            matching_pt = pt_dir / f"{dat_file.stem}.pt"
            if matching_pt.exists():
                return dat_file, matching_pt

    return None, None


def list_available_instances(subtype: str) -> List[str]:
    """List all available instances for a subtype.

    Args:
        subtype: problem subtype

    Returns:
        list of instance names that have both .dat-s and .pt files
    """
    instance_dir = INSTANCES_DIR / subtype
    pt_dir = PT_DIR / subtype

    if not instance_dir.exists() or not pt_dir.exists():
        return []

    instances = []
    for dat_file in sorted(instance_dir.glob("*.dat-s")):
        pt_file = pt_dir / f"{dat_file.stem}.pt"
        if pt_file.exists():
            instances.append(dat_file.stem)

    return instances


def run_benchmark(
    model: RankSchedulePredictor,
    device: torch.device,
    subtype: str,
    instance_name: str,
    output_dir: Path,
    use_rank_schedule: bool = False,
) -> Optional[Dict]:
    """Run benchmark for a single instance.

    Args:
        model: trained model for prediction
        device: device for inference
        subtype: problem subtype
        instance_name: problem name
        output_dir: directory for outputs
        use_rank_schedule: if True, use full rank schedule; if False (default), use fixed rank

    Returns:
        dict with benchmark results or None on failure
    """
    instance_path, pt_path = find_matching_files(subtype, instance_name)

    if instance_path is None or pt_path is None:
        print(f"  [skip] files not found for {subtype}/{instance_name}")
        return None

    instance_output_dir = output_dir / subtype / instance_name
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  predicting rank schedule...")
    schedule = predict_rank_schedule(model, pt_path, device)
    print(f"  predicted schedule: {schedule}")

    r_sched_path = instance_output_dir / "r_sched.json"
    save_rank_schedule(schedule, r_sched_path)

    final_rank = schedule[-1] if schedule else None
    if use_rank_schedule:
        print(f"  using rank schedule (length={len(schedule)})")
    else:
        print(f"  using fixed rank: {final_rank}")

    params = get_lorads_params(instance_name, subtype)

    print(f"  running original lorads...")
    sol_lorads_path = instance_output_dir / "sol_lorads.json"
    lorads_success, lorads_time, lorads_obj = run_lorads(
        instance_path, sol_lorads_path, params
    )

    if use_rank_schedule:
        print(f"  running with rank schedule...")
        sol_rsched_path = instance_output_dir / "sol_rsched.json"
        rsched_success, rsched_time, rsched_obj = run_lorads(
            instance_path, sol_rsched_path, params, rank_schedule_path=r_sched_path
        )
    else:
        print(f"  running with fixed rank...")
        sol_rsched_path = instance_output_dir / "sol_rsched.json"
        rsched_success, rsched_time, rsched_obj = run_lorads(
            instance_path, sol_rsched_path, params, fixed_rank=final_rank
        )

    result = {
        "instance": instance_name,
        "subtype": subtype,
        "schedule": schedule,
        "lorads_success": lorads_success,
        "lorads_time": lorads_time,
        "lorads_obj": lorads_obj,
        "rsched_success": rsched_success,
        "rsched_time": rsched_time,
        "rsched_obj": rsched_obj,
    }

    if lorads_success and rsched_success and lorads_time and rsched_time:
        result["speedup"] = lorads_time / rsched_time if rsched_time > 0 else None
    else:
        result["speedup"] = None

    return result


def print_results_table(results: List[Dict]) -> None:
    """Print benchmark results as a formatted table.

    Args:
        results: list of benchmark result dicts
    """
    headers = ["instance", "subtype", "LoRADS (s)", "rankSched (s)"]

    rows = []
    for r in results:
        lorads_time = f"{r['lorads_time']:.2f}" if r["lorads_time"] else "FAIL"
        rsched_time = f"{r['rsched_time']:.2f}" if r["rsched_time"] else "FAIL"

        rows.append([r["instance"], r["subtype"], lorads_time, rsched_time])

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def load_results_from_logs(logs_dir: Path) -> List[Dict]:
    """Load benchmark results from a logs directory."""
    results_path = logs_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rank schedule approach vs original LoRADS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        choices=SUBTYPES,
        default=None,
        help="Problem subtype to benchmark (defaults to all subtypes when omitted)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Specific instance name (without extension); requires --subtype",
    )
    parser.add_argument(
        "--all-instances",
        action="store_true",
        default=True,
        help="Run on all instances in the subtype(s)",
    )
    parser.add_argument(
        "--no-all-instances",
        action="store_false",
        dest="all_instances",
        help="Run only the specified instance",
    )
    parser.add_argument(
        "--all-subtypes",
        action="store_true",
        default=True,
        help="Run on all subtypes (gset, hansmittel, matcomp, maxcut, sdplib)",
    )
    parser.add_argument(
        "--no-all-subtypes",
        action="store_false",
        dest="all_subtypes",
        help="Run only the specified subtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Solver timeout in seconds",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available instances for the subtype(s) and exit",
    )
    parser.add_argument(
        "--rank-schedule",
        action="store_true",
        default=True,
        help="Use full rank schedule (default)",
    )
    parser.add_argument(
        "--fixed-rank",
        action="store_false",
        dest="rank_schedule",
        help="Use fixed rank instead of rank schedule",
    )
    parser.add_argument(
        "--print-res-from-logs",
        type=str,
        default=None,
        metavar="LOGS_DIR",
        help="Skip benchmarking and print results table from existing logs directory",
    )

    args = parser.parse_args()
    _config["timeout"] = args.timeout

    if args.print_res_from_logs:
        logs_dir = Path(args.print_res_from_logs)
        if not logs_dir.exists():
            print(f"[error] logs directory not found: {logs_dir}")
            sys.exit(1)
        results = load_results_from_logs(logs_dir)
        if results:
            print(f"\n[results from {logs_dir}]")
            print_results_table(results)
        else:
            print(f"[error] no results.json found in {logs_dir}")
            sys.exit(1)
        return

    if args.instance and not args.subtype:
        parser.error("--instance requires --subtype")

    if args.subtype:
        subtypes_to_run = [args.subtype]
    elif args.all_subtypes:
        subtypes_to_run = SUBTYPES
    else:
        parser.error("either --subtype or --all-subtypes must be specified")

    if not args.all_instances and args.instance is None:
        parser.error("--instance is required when using --no-all-instances")

    if args.list:
        for subtype in subtypes_to_run:
            instances = list_available_instances(subtype)
            print(f"Available instances for {subtype}:")
            for inst in instances:
                print(f"  - {inst}")
            print(f"  Total: {len(instances)} instances\n")
        return

    if not args.all_instances:
        if args.instance is None:
            parser.error("--instance is required when using --no-all-instances")
        if args.all_subtypes and args.subtype is None:
            parser.error("--subtype is required when using --no-all-instances")

    if not Path(LORADS_EXECUTABLE).exists():
        print(f"[error] lorads executable not found at {LORADS_EXECUTABLE}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[error] checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"using device: {device}")

    print(f"loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print(f"model loaded ({model.count_parameters():,} parameters)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs") / f"bench_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {output_dir}")

    all_results = []
    total_instances = 0

    for subtype in subtypes_to_run:
        if args.instance:
            instances = [args.instance]
        elif args.all_instances:
            instances = list_available_instances(subtype)
            if not instances:
                print(f"[warning] no instances found for subtype {subtype}")
                continue
        else:
            instances = [args.instance]

        print(f"\n[subtype]: {subtype} ({len(instances)} instances)")

        for i, instance_name in enumerate(instances, 1):
            total_instances += 1
            print(f"\n[{i}/{len(instances)}] {subtype}/{instance_name}")
            result = run_benchmark(
                model, device, subtype, instance_name, output_dir, args.rank_schedule
            )
            if result:
                all_results.append(result)

    if all_results:
        print("\n[results]")
        print_results_table(all_results)

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nresults saved to: {results_path}")
    else:
        print("\n[warning] no successful benchmarks")


if __name__ == "__main__":
    main()
