"""\
This script mirrors the training style in train.py but runs fixed-epoch
trials to optimize hyperparameters.

Example usage:
    python tune.py --trials 30 --timeout 21600
    python tune.py --trials 50 --epochs 80 --param-budget 800000
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import optuna
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from dataset.loader import create_dataloaders
from model import RankSchedulePredictor
from train import (
    RankScheduleLoss,
    evaluate,
    get_teacher_forcing_ratio,
    train_epoch,
)


def build_model_config(trial: optuna.Trial) -> Dict:
    """Sample a model configuration respecting memory constraints.

    Args:
        trial: Optuna trial

    Returns:
        Dictionary of model hyperparameters
    """
    hidden_dim = trial.suggest_int("hidden_dim", 32, 96, step=16)
    decoder_hidden_dim = trial.suggest_int("decoder_hidden_dim", 32, 96, step=16)

    valid_heads = [h for h in [2, 4] if hidden_dim % h == 0]
    num_heads = trial.suggest_categorical(
        "num_heads", valid_heads if valid_heads else [2]
    )

    gnn_layers = trial.suggest_int("num_gnn_layers", 2, 4)
    decoder_layers = trial.suggest_int("decoder_num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    return {
        "hidden_dim": hidden_dim,
        "edge_dim": hidden_dim // 2,
        "global_dim": hidden_dim // 2,
        "num_gnn_layers": gnn_layers,
        "num_heads": num_heads,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_num_layers": decoder_layers,
        "dropout": dropout,
    }


def build_loss_config(trial: optuna.Trial) -> Dict:
    """Sample loss-related hyperparameters.

    Returns:
        Dictionary of loss hyperparameters
    """
    return {
        "schedule_weight": 1.0,
        "length_weight": trial.suggest_float("length_weight", 0.2, 1.0),
        "mono_weight": trial.suggest_float("mono_weight", 0.0, 0.15),
        "initial_weight": trial.suggest_float("initial_weight", 0.1, 0.5),
        "final_weight": trial.suggest_float("final_weight", 0.1, 0.5),
        "under_weight": trial.suggest_float("under_weight", 1.0, 4.0),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
    }


def make_optimizer_config(trial: optuna.Trial) -> Dict:
    """Sample optimizer hyperparameters.

    Returns:
        Dictionary of optimizer hyperparameters
    """
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "plateau"])
    warmup_epochs = 0
    if scheduler == "cosine":
        warmup_epochs = trial.suggest_int("warmup_epochs", 0, 20)

    return {
        "lr": trial.suggest_float("lr", 5e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "scheduler": scheduler,
        "warmup_epochs": warmup_epochs,
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 2.0),
    }


def count_and_guard(model: RankSchedulePredictor, budget: int) -> None:
    """Prune the trial if parameter count exceeds the budget."""
    param_count = model.count_parameters()
    if param_count > budget:
        raise optuna.TrialPruned(
            f"parameter budget exceeded: {param_count:,} > {budget:,}"
        )


class BestTrialCallback:
    """Callback to save best.json after each completed trial."""

    def __init__(self, log_root: Path):
        self.log_root = log_root
        self.best_value = float("inf")
        self.best_params = None
        self.best_trial_number = None

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if trial.value is not None and trial.value < self.best_value:
            self.best_value = trial.value
            self.best_params = trial.params
            self.best_trial_number = trial.number

            best = {
                "value": self.best_value,
                "params": self.best_params,
                "trial_number": self.best_trial_number,
            }
            with open(self.log_root / "best.json", "w") as f:
                json.dump(best, f, indent=2)


def objective(
    trial: optuna.Trial,
    dataset_root: str,
    device: torch.device,
    log_root: Path,
    max_seq_len: int,
    param_budget: int,
    epochs: int,
    seed: int,
) -> float:
    """Run one Optuna trial.

    Args:
        trial: Optuna trial object
        dataset_root: Path to dataset root
        device: torch device
        log_root: Directory for trial logs
        max_seq_len: Maximum sequence length
        param_budget: Maximum model parameters
        epochs: Number of training epochs
        seed: Random seed

    Returns:
        Best validation log_mae achieved during training
    """

    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    accumulation_steps = trial.suggest_categorical("accumulation_steps", [2, 4, 8, 16])

    train_loader, val_loader, _ = create_dataloaders(
        root=dataset_root,
        batch_size=batch_size,
        seed=seed,
        num_workers=2,
        max_schedule_length=max_seq_len,
    )

    sample = train_loader.dataset[0]
    node_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    global_dim = sample.global_attr.numel()

    model_cfg = {
        "node_in_dim": node_dim,
        "edge_in_dim": edge_dim,
        "global_in_dim": global_dim,
        "max_seq_len": max_seq_len,
    }
    model_cfg.update(build_model_config(trial))

    model = RankSchedulePredictor(**model_cfg).to(device)
    count_and_guard(model, param_budget)

    loss_cfg = build_loss_config(trial)
    criterion = RankScheduleLoss(**loss_cfg)

    opt_cfg = make_optimizer_config(trial)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=(0.9, 0.999),
    )

    if opt_cfg["scheduler"] == "cosine":
        effective_epochs = max(1, epochs - opt_cfg["warmup_epochs"])
        scheduler = CosineAnnealingLR(optimizer, T_max=effective_epochs, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    trial_dir = log_root / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    tf_start = trial.suggest_float("tf_start", 0.7, 0.95)
    tf_end = trial.suggest_float("tf_end", 0.05, 0.3)

    best_val = float("inf")
    patience_counter = 0
    early_stop_patience = max(15, epochs // 4)

    for epoch in range(1, epochs + 1):

        tf_ratio = get_teacher_forcing_ratio(epoch, epochs, tf_start, tf_end)

        if opt_cfg["scheduler"] == "cosine" and epoch <= opt_cfg["warmup_epochs"]:
            warmup_lr = opt_cfg["lr"] * epoch / max(1, opt_cfg["warmup_epochs"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=tf_ratio,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            max_grad_norm=opt_cfg["max_grad_norm"],
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=(scaler is not None),
        )
        val_log_mae = val_metrics["log_mae"]

        if val_log_mae < best_val:
            best_val = val_log_mae
            patience_counter = 0
        else:
            patience_counter += 1

        if opt_cfg["scheduler"] == "cosine":
            if epoch > opt_cfg["warmup_epochs"]:
                scheduler.step()
        else:
            scheduler.step(val_log_mae)

        trial.report(val_log_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {epoch}")

        if patience_counter >= early_stop_patience:
            break

        if epoch % 10 == 0 or epoch == epochs:
            state = {
                "epoch": epoch,
                "val_log_mae": val_log_mae,
                "best_val": best_val,
                "batch_size": batch_size,
                "accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
                "config": {
                    "model": model_cfg,
                    "loss": loss_cfg,
                    "opt": opt_cfg,
                },
            }
            with open(trial_dir / "state.json", "w") as f:
                json.dump(state, f, indent=2)

    del model, optimizer, scheduler, criterion
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    return best_val


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna tuner for rank schedule model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs per trial")
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds for entire study"
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument(
        "--param-budget",
        type=int,
        default=600_000,
        help="Maximum number of model parameters (OOM observed >600k)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "cmaes", "random"],
        help="Optuna sampler type",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_dir) / f"tune_{timestamp}"
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"[tune] device: {device}")
    print(f"[tune] log dir: {log_root}")
    print(f"[tune] trials: {args.trials}, epochs/trial: {args.epochs}")
    print(f"[tune] param budget: {args.param_budget:,}")

    temp_loader, val_loader, _ = create_dataloaders(
        root=args.data_root,
        batch_size=1,
        seed=args.seed,
        num_workers=0,
        max_schedule_length=args.max_seq_len,
    )
    print(f"[tune] train samples: {len(temp_loader.dataset)}")
    print(f"[tune] val samples: {len(val_loader.dataset)}")
    del temp_loader, val_loader

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=15,
        interval_steps=5,
    )

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    elif args.sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name or f"rank_schedule_{timestamp}",
        pruner=pruner,
        sampler=sampler,
    )

    def wrapped_objective(trial: optuna.Trial) -> float:
        try:
            return objective(
                trial,
                args.data_root,
                device,
                log_root,
                args.max_seq_len,
                args.param_budget,
                args.epochs,
                args.seed,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(2)

            raise optuna.TrialPruned("cuda oom")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(2)
                raise optuna.TrialPruned(f"runtime oom: {e}")
            raise

    best_callback = BestTrialCallback(log_root)

    study.optimize(
        wrapped_objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[best_callback],
    )

    if study.best_trial is not None:
        best = {
            "value": study.best_value,
            "params": study.best_params,
            "trial_number": study.best_trial.number,
        }
        with open(log_root / "best.json", "w") as f:
            json.dump(best, f, indent=2)

    try:
        importance = optuna.importance.get_param_importances(study)
        with open(log_root / "importance.json", "w") as f:
            json.dump(importance, f, indent=2)
    except Exception:
        pass

    trials_data = []
    for t in study.trials:
        trials_data.append(
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
        )
    with open(log_root / "all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    print("[best trial]")
    print(f"  value (log_mae): {study.best_value:.4f}")
    print(f"  trial number: {study.best_trial.number}")
    print("\n[best params]")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
