"""\
This script trains RankPredictor for optimal rank (r_oracle) for SDP problems.

Example usage:
    >>> python train.py --epochs 300 --batch-size 32 --hidden-dim 128
    >>> python train.py --data-root dataset --log-dir runs/exp1 --patience 30
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from dataset.loader import create_dataloaders, SDPDataset
from model import RankPredictor


class LogMSELoss(nn.Module):
    """Log-space MSE loss for scale-invariant rank prediction.

    Loss formula:
        L = MSE(log(pred), log(target))

    Attributes:
        eps: Small constant for numerical stability in log computation.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize the log-space MSE loss.

        Args:
            eps: Small constant added before log for numerical stability.
                Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the log-space MSE loss.

        Args:
            pred: Predicted ranks of shape [batch_size, 1], in linear scale.
            target: True ranks of shape [batch_size, 1], in linear scale.

        Returns:
            Scalar loss tensor.
        """
        pred_log = torch.log(pred.clamp(min=self.eps))
        target_log = torch.log(target.clamp(min=self.eps))
        return F.mse_loss(pred_log, target_log)


def extract_global_attr(batch) -> torch.Tensor:
    """Extract global attributes from a PyG batch.

    The DataLoader concatenates global_attr into a 1D tensor. We need to
    reshape it back to [batch_size, global_dim].

    Args:
        batch: PyG Batch object with global_attr attribute.

    Returns:
        Tensor of shape [batch_size, global_dim].
    """
    batch_size = batch.batch.max().item() + 1
    global_dim = batch.global_attr.numel() // batch_size

    return batch.global_attr.view(batch_size, global_dim)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
) -> float:
    """Run one training epoch with optional AMP.

    Args:
        model: The RankPredictor model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run on.
        scaler: GradScaler for AMP (None to disable AMP).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    use_amp = scaler is not None and device.type == "cuda"

    for batch_idx, batch in enumerate(tqdm(loader, desc="training", leave=False)):
        batch = batch.to(device)
        global_attr = extract_global_attr(batch).to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_attr=global_attr,
            )
            loss = criterion(pred, batch.y.view(-1, 1))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if device.type == "cuda" and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: The RankPredictor model.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Device to run on.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dictionary with metrics: loss, mse, mae, safe_rate, log_mae.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_log_mae = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="evaluating", leave=False):
        batch = batch.to(device)
        global_attr = extract_global_attr(batch).to(device)

        with torch.amp.autocast(
            device_type="cuda", enabled=use_amp and device.type == "cuda"
        ):
            pred = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_attr=global_attr,
            )

        target = batch.y.view(-1, 1)

        loss = criterion(pred, target)
        total_loss += loss.item() * target.size(0)

        mse = F.mse_loss(pred, target, reduction="sum").item()
        mae = F.l1_loss(pred, target, reduction="sum").item()
        log_mae = F.l1_loss(
            torch.log(pred.clamp(min=1)),
            torch.log(target.clamp(min=1)),
            reduction="sum",
        ).item()

        total_mse += mse
        total_mae += mae
        total_log_mae += log_mae
        total_samples += target.size(0)

        all_preds.extend(pred.cpu().numpy().flatten().tolist())
        all_targets.extend(target.cpu().numpy().flatten().tolist())

    n = total_samples
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "mae": total_mae / n,
        "log_mae": total_log_mae / n,
        "rmse": (total_mse / n) ** 0.5,
        "predictions": all_preds,
        "targets": all_targets,
    }


def print_epoch_summary(
    epoch: int,
    train_loss: float,
    val_metrics: Dict[str, float],
    lr: float,
    is_best: bool,
    elapsed: float,
) -> None:
    """Print a summary line for the epoch."""
    best_marker = " *" if is_best else ""
    print(
        f"epoch {epoch:3d} | "
        f"train loss: {train_loss:.4f} | "
        f"val log_mae: {val_metrics['log_mae']:.3f} | "
        f"val mae: {val_metrics['mae']:.1f} | "
        f"lr: {lr:.2e} | "
        f"time: {elapsed:.1f}s{best_marker}"
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: Dict[str, float],
    path: Path,
) -> None:
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_log_mae": val_metrics["log_mae"],
            "val_mae": val_metrics["mae"],
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])


def generate_test_report(
    test_metrics: Dict[str, float],
    model_config: Dict,
    training_config: Dict,
) -> str:
    """Generate a detailed test report."""
    report = []
    report.append("[test report]")
    report.append("")
    report.append("model configuration:")
    for k, v in model_config.items():
        report.append(f"  {k}: {v}")
    report.append("")
    report.append("training configuration:")
    for k, v in training_config.items():
        report.append(f"  {k}: {v}")
    report.append("")
    report.append("test set metrics:")
    report.append(f"  log_mae (log-space mae): {test_metrics['log_mae']:.4f}")
    report.append(f"  mae (mean absolute error): {test_metrics['mae']:.4f}")
    report.append(f"  rmse (root mean squared error): {test_metrics['rmse']:.4f}")
    report.append(f"  mse (mean squared error): {test_metrics['mse']:.4f}")
    report.append("")

    # compute additional statistics
    preds = test_metrics["predictions"]
    targets = test_metrics["targets"]
    if preds and targets:
        import numpy as np

        preds = np.array(preds)
        targets = np.array(targets)
        errors = preds - targets

        report.append("error distribution:")
        report.append(f"  mean error: {errors.mean():.4f}")
        report.append(f"  std error: {errors.std():.4f}")
        report.append(f"  min error: {errors.min():.4f} (most under-predicted)")
        report.append(f"  max error: {errors.max():.4f} (most over-predicted)")

        # percentiles
        report.append("")
        report.append("prediction accuracy by percentile:")
        for p in [50, 75, 90, 95]:
            abs_err = np.abs(errors)
            report.append(
                f"  {p}th percentile |error|: {np.percentile(abs_err, p):.2f}"
            )

    return "\n".join(report)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train RankPredictor GNN for SDP rank prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducibility",
    )

    # training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs without improvement)",
    )

    # model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for GNN layers",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads in GATv2Conv",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )

    # system arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for saving checkpoints and logs",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (AMP) for memory efficiency",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable automatic mixed precision",
    )

    args = parser.parse_args()

    # set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"using device: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"train_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logging to: {log_dir}")

    print("\nloading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"  train samples: {len(train_loader.dataset)}")
    print(f"  val samples: {len(val_loader.dataset)}")
    print(f"  test samples: {len(test_loader.dataset)}")

    print("\ninitializing model...")
    model = RankPredictor(
        node_in_dim=8,
        edge_in_dim=2,
        global_in_dim=5,
        hidden_dim=args.hidden_dim,
        edge_dim=args.hidden_dim // 2,
        global_dim=args.hidden_dim // 2,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        log_output=True,
    ).to(device)
    print(f"  parameters: {model.count_parameters():,}")
    print(f"  {model}")

    criterion = LogMSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )
    if scaler:
        print("  using automatic mixed precision (AMP)")

    # training state
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    training_log: List[Dict] = []

    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "dropout": args.dropout,
    }
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
    }

    # save config
    config = {"model": model_config, "training": training_config}
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nstarting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=args.amp and device.type == "cuda",
        )
        scheduler.step(val_metrics["log_mae"])

        is_best = val_metrics["log_mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["log_mae"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics, log_dir / "best_model.pt"
            )
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print_epoch_summary(
            epoch, train_loss, val_metrics, current_lr, is_best, elapsed
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_log_mae": val_metrics["log_mae"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "lr": current_lr,
            "is_best": is_best,
        }
        training_log.append(log_entry)

        if epochs_without_improvement >= args.patience:
            print(f"\nearly stopping: no improvement for {args.patience} epochs")
            break

    with open(log_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("\nevaluating on test set...")
    load_checkpoint(model, log_dir / "best_model.pt", device)
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp=args.amp and device.type == "cuda",
    )
    report = generate_test_report(test_metrics, model_config, training_config)
    print(report)

    with open(log_dir / "test_report.txt", "w") as f:
        f.write(report)

    print(f"\nall outputs saved to: {log_dir}")


if __name__ == "__main__":
    main()
