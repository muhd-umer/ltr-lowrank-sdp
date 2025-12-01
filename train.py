"""\
This script trains a GNN encoder with sequence decoder to predict optimal
rank schedules (trajectories) for SDP problems.

Example usage:
    >>> python train.py --epochs 300 --batch-size 16 --hidden-dim 128
    >>> python train.py --data-root dataset --log-dir runs/exp1 --patience 30

For long training runs, set environment variable to reduce memory fragmentation:
    >>> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py ...
"""

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset.loader import create_dataloaders
from model import RankSchedulePredictor


class RankScheduleLoss(nn.Module):
    """Multi-objective loss for rank schedule prediction.

    Combines:
        1. Masked log-space MSE for rank values (scale-invariant)
        2. Cross-entropy for sequence length prediction
        3. Optional monotonicity penalty

    Loss formula:
        L = w_schedule * L_schedule + w_length * L_length + w_mono * L_mono

    Attributes:
        schedule_weight: Weight for schedule prediction loss
        length_weight: Weight for length prediction loss
        mono_weight: Weight for monotonicity penalty
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        schedule_weight: float = 1.0,
        length_weight: float = 0.5,
        mono_weight: float = 0.1,
        initial_weight: float = 0.25,
        final_weight: float = 0.25,
        under_weight: float = 2.5,
        eps: float = 1e-6,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize the RankScheduleLoss.

        Args:
            schedule_weight: Weight for schedule prediction loss
            length_weight: Weight for length prediction loss
            mono_weight: Weight for monotonicity penalty (0 to disable)
            initial_weight: Weight for auxiliary initial-rank loss
            final_weight: Weight for auxiliary final-rank loss
            under_weight: Multiplier applied to under-prediction errors
            eps: Small constant for numerical stability
            label_smoothing: Label smoothing for length classification
        """
        super().__init__()
        self.schedule_weight = schedule_weight
        self.length_weight = length_weight
        self.mono_weight = mono_weight
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.under_weight = under_weight
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pred_schedule: torch.Tensor,
        target_schedule: torch.Tensor,
        pred_length_logits: torch.Tensor,
        target_length: torch.Tensor,
        mask: torch.Tensor,
        pred_initial: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the multi-objective loss.

        Args:
            pred_schedule: Predicted ranks [batch_size, max_seq_len] in linscale
            target_schedule: Target ranks [batch_size, max_seq_len] in linscale
            pred_length_logits: Logits for length [batch_size, max_seq_len]
            target_length: True sequence lengths [batch_size, 1]
            mask: Binary mask [batch_size, max_seq_len] for valid positions
            pred_initial: Optional predicted initial rank [batch_size, 1]

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for logging
        """

        pred_log = torch.log(pred_schedule.clamp(min=self.eps))
        target_log = torch.log(target_schedule.clamp(min=self.eps))

        sq_error = (pred_log - target_log) ** 2
        under_mask = (pred_schedule < target_schedule).float()
        weights = torch.where(under_mask > 0, self.under_weight, 1.0)

        masked_sq_error = sq_error * mask * weights
        num_valid = (mask * weights).sum() + self.eps
        schedule_loss = masked_sq_error.sum() / num_valid

        target_length_class = (target_length.squeeze(-1) - 1).clamp(
            min=0, max=pred_length_logits.size(-1) - 1
        )
        length_loss = F.cross_entropy(
            pred_length_logits,
            target_length_class,
            label_smoothing=self.label_smoothing,
        )

        if self.mono_weight > 0:

            diff = pred_schedule[:, 1:] - pred_schedule[:, :-1]

            mono_penalty = F.relu(-diff)

            mono_mask = mask[:, 1:] * mask[:, :-1]
            mono_valid = mono_mask.sum() + self.eps
            mono_loss = (mono_penalty * mono_mask).sum() / mono_valid
        else:
            mono_loss = torch.tensor(0.0, device=pred_schedule.device)

        if pred_initial is not None:
            init_target = target_schedule[:, :1]
            init_mask = mask[:, :1]
            init_log_diff = torch.abs(
                torch.log(pred_initial.clamp(min=self.eps))
                - torch.log(init_target.clamp(min=self.eps))
            )
            init_loss = (init_log_diff * init_mask).sum() / (init_mask.sum() + self.eps)
        else:
            init_loss = torch.tensor(0.0, device=pred_schedule.device)

        batch_indices = torch.arange(target_length.size(0), device=pred_schedule.device)
        final_positions = (target_length.squeeze(-1) - 1).clamp(
            min=0, max=pred_schedule.size(1) - 1
        )
        pred_final = pred_schedule[batch_indices, final_positions]
        target_final = target_schedule[batch_indices, final_positions]
        final_under = (pred_final < target_final).float() * (
            self.under_weight - 1.0
        ) + 1.0
        final_log_diff = torch.abs(
            torch.log(pred_final.clamp(min=self.eps))
            - torch.log(target_final.clamp(min=self.eps))
        )
        final_loss = (final_log_diff * final_under).mean()

        total_loss = (
            self.schedule_weight * schedule_loss
            + self.length_weight * length_loss
            + self.mono_weight * mono_loss
            + self.initial_weight * init_loss
            + self.final_weight * final_loss
        )

        loss_dict = {
            "schedule_loss": schedule_loss.item(),
            "length_loss": length_loss.item(),
            "mono_loss": mono_loss.item(),
            "init_loss": init_loss.item(),
            "final_loss": final_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


def extract_global_attr(batch) -> torch.Tensor:
    """Extract global attributes from a PyG batch.

    Args:
        batch: PyG Batch object with global_attr attribute

    Returns:
        Tensor of shape [batch_size, global_dim]
    """
    batch_size = batch.batch.max().item() + 1
    global_dim = batch.global_attr.numel() // batch_size

    return batch.global_attr.view(batch_size, global_dim)


def get_teacher_forcing_ratio(
    epoch: int, max_epochs: int, start: float = 0.9, end: float = 0.2
) -> float:
    """Compute teacher forcing ratio with linear decay.

    Args:
        epoch: Current epoch (1-indexed)
        max_epochs: Total number of epochs
        start: Initial teacher forcing ratio
        end: Final teacher forcing ratio

    Returns:
        Teacher forcing ratio for the current epoch
    """
    progress = (epoch - 1) / max(1, max_epochs - 1)
    return start - progress * (start - end)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        model: The RankSchedulePredictor model
        loader: Training DataLoader
        optimizer: Optimizer instance
        criterion: Loss function (RankScheduleLoss)
        device: Device to run on
        teacher_forcing_ratio: Probability of teacher forcing
        scaler: GradScaler for AMP (None to disable AMP)
        max_grad_norm: Maximum gradient norm for clipping
        accumulation_steps: Number of steps to accumulate gradients

    Returns:
        Dictionary with average losses for the epoch
    """
    model.train()

    total_losses = {
        "schedule_loss": 0.0,
        "length_loss": 0.0,
        "mono_loss": 0.0,
        "init_loss": 0.0,
        "final_loss": 0.0,
        "total_loss": 0.0,
    }
    num_batches = 0

    use_amp = scaler is not None and device.type == "cuda"

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(loader, desc="training", leave=False)):
        batch = batch.to(device)
        global_attr = extract_global_attr(batch).to(device)

        target_schedule = batch.rank_schedule.view(-1, model.max_seq_len)
        target_length = batch.schedule_length.view(-1, 1)
        mask = batch.schedule_mask.view(-1, model.max_seq_len)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred_schedule, pred_length_logits, pred_initial = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_attr=global_attr,
                target_schedule=target_schedule,
                target_mask=mask,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            loss, loss_dict = criterion(
                pred_schedule=pred_schedule,
                target_schedule=target_schedule,
                pred_length_logits=pred_length_logits,
                target_length=target_length,
                mask=mask,
                pred_initial=pred_initial,
            )

            loss = loss / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()
            optimizer.zero_grad()

        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1

        del batch, global_attr, target_schedule, target_length, mask
        del pred_schedule, pred_length_logits, pred_initial, loss, loss_dict

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {key: val / num_batches for key, val in total_losses.items()}


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
        model: The RankSchedulePredictor model
        loader: Evaluation DataLoader
        criterion: Loss function
        device: Device to run on
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    total_losses = {
        "schedule_loss": 0.0,
        "length_loss": 0.0,
        "mono_loss": 0.0,
        "init_loss": 0.0,
        "final_loss": 0.0,
        "total_loss": 0.0,
    }
    total_samples = 0

    total_log_mae = 0.0
    total_mae = 0.0
    total_length_acc = 0.0
    total_exact_length = 0

    all_preds: List[List[int]] = []
    all_targets: List[List[int]] = []
    all_pred_lengths: List[int] = []
    all_target_lengths: List[int] = []

    for batch in tqdm(loader, desc="evaluating", leave=False):
        batch = batch.to(device)
        global_attr = extract_global_attr(batch).to(device)

        target_schedule = batch.rank_schedule.view(-1, model.max_seq_len)
        target_length = batch.schedule_length.view(-1, 1)
        mask = batch.schedule_mask.view(-1, model.max_seq_len)

        with torch.amp.autocast(
            device_type="cuda", enabled=use_amp and device.type == "cuda"
        ):

            pred_schedule, pred_lengths = model.predict(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_attr=global_attr,
                return_integers=False,
            )

            pred_schedule_tf, pred_length_logits, pred_initial_tf = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_attr=global_attr,
                target_schedule=target_schedule,
                target_mask=mask,
                teacher_forcing_ratio=0.0,
            )

            loss, loss_dict = criterion(
                pred_schedule=pred_schedule_tf,
                target_schedule=target_schedule,
                pred_length_logits=pred_length_logits,
                target_length=target_length,
                mask=mask,
                pred_initial=pred_initial_tf,
            )

        batch_size = target_length.size(0)
        total_samples += batch_size

        for key in total_losses:
            total_losses[key] += loss_dict[key] * batch_size

        pred_log = torch.log(pred_schedule.clamp(min=1e-6))
        target_log = torch.log(target_schedule.clamp(min=1e-6))
        log_ae = torch.abs(pred_log - target_log) * mask
        total_log_mae += log_ae.sum().item()

        ae = torch.abs(pred_schedule - target_schedule) * mask
        total_mae += ae.sum().item()

        target_len_class = target_length.squeeze(-1)
        length_correct = (pred_lengths == target_len_class).float()
        total_length_acc += length_correct.sum().item()
        total_exact_length += (pred_lengths == target_len_class).sum().item()

        valid_schedules = model.get_valid_schedule(
            torch.round(pred_schedule).long().clamp(min=1),
            pred_lengths,
        )
        all_preds.extend(valid_schedules)
        all_pred_lengths.extend(pred_lengths.tolist())

        for i in range(batch_size):
            tgt_len = target_len_class[i].item()
            tgt_sched = target_schedule[i, : int(tgt_len)].long().tolist()
            all_targets.append(tgt_sched)
            all_target_lengths.append(int(tgt_len))

        del pred_schedule, pred_lengths, pred_schedule_tf, pred_length_logits
        del pred_initial_tf
        del loss, loss_dict, pred_log, target_log, log_ae, ae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = total_samples
    total_valid_positions = sum(len(t) for t in all_targets)

    metrics = {
        "total_loss": total_losses["total_loss"] / n,
        "schedule_loss": total_losses["schedule_loss"] / n,
        "length_loss": total_losses["length_loss"] / n,
        "mono_loss": total_losses["mono_loss"] / n,
        "init_loss": total_losses["init_loss"] / n,
        "final_loss": total_losses["final_loss"] / n,
        "log_mae": (
            total_log_mae / total_valid_positions if total_valid_positions > 0 else 0.0
        ),
        "mae": total_mae / total_valid_positions if total_valid_positions > 0 else 0.0,
        "length_accuracy": total_length_acc / n if n > 0 else 0.0,
        "exact_length_count": total_exact_length,
        "predictions": all_preds,
        "targets": all_targets,
        "pred_lengths": all_pred_lengths,
        "target_lengths": all_target_lengths,
    }

    return metrics


def print_epoch_summary(
    epoch: int,
    train_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    tf_ratio: float,
    is_best: bool,
    elapsed: float,
) -> None:
    """Print a summary line for the epoch."""
    best_marker = " *" if is_best else ""
    print(
        f"epoch {epoch:3d} | "
        f"train: {train_losses['total_loss']:.4f} | "
        f"val log_mae: {val_metrics['log_mae']:.3f} | "
        f"val len_acc: {val_metrics['length_accuracy']:.2%} | "
        f"lr: {lr:.2e} | "
        f"tf: {tf_ratio:.2f} | "
        f"time: {elapsed:.1f}s{best_marker}"
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    val_metrics: Dict[str, float],
    path: Path,
    model_config: Dict,
) -> None:
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
            ),
            "val_log_mae": val_metrics["log_mae"],
            "val_total_loss": val_metrics["total_loss"],
            "val_length_accuracy": val_metrics["length_accuracy"],
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> Dict:
    """Load model checkpoint and return metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def eval_report(
    test_metrics: Dict, model_config: Dict = None, training_config: Dict = None
) -> str:
    """Generate evaluation report from test metrics."""
    report = []
    report.append("[eval report]")
    report.append(f"  total loss: {test_metrics['total_loss']:.4f}")
    report.append(
        f"  schedule loss (log-space mse): {test_metrics['schedule_loss']:.4f}"
    )
    report.append(f"  length loss (ce): {test_metrics['length_loss']:.4f}")
    report.append(f"  init loss: {test_metrics['init_loss']:.4f}")
    report.append(f"  final loss: {test_metrics['final_loss']:.4f}")
    report.append(f"  log mae: {test_metrics['log_mae']:.4f}")
    report.append(f"  mae: {test_metrics['mae']:.4f}")
    report.append(f"  length accuracy: {test_metrics['length_accuracy']:.2%}")
    report.append(f"  exact length matches: {test_metrics['exact_length_count']}")
    report.append("")

    preds = test_metrics["predictions"]
    targets = test_metrics["targets"]
    pred_lengths = test_metrics["pred_lengths"]
    target_lengths = test_metrics["target_lengths"]

    if preds and targets:
        import numpy as np

        report.append("[length distribution]")
        tgt_lens = np.array(target_lengths)
        pred_lens = np.array(pred_lengths)
        len_errors = pred_lens - tgt_lens

        report.append(
            f"  target lengths: mean={tgt_lens.mean():.2f}, std={tgt_lens.std():.2f}, min={tgt_lens.min()}, max={tgt_lens.max()}"
        )
        report.append(
            f"  pred lengths: mean={pred_lens.mean():.2f}, std={pred_lens.std():.2f}, min={pred_lens.min()}, max={pred_lens.max()}"
        )
        report.append(
            f"  length error: mean={len_errors.mean():.2f}, std={len_errors.std():.2f}"
        )
        report.append("")

        report.append("[per-position error]")
        max_pos = min(5, max(len(t) for t in targets))
        for pos in range(max_pos):
            pos_errors = []
            for p, t in zip(preds, targets):
                if pos < len(p) and pos < len(t):
                    pos_errors.append(p[pos] - t[pos])
            if pos_errors:
                errors = np.array(pos_errors)
                report.append(
                    f"  position {pos+1}: mean_err={errors.mean():.2f}, std={errors.std():.2f}, |mean_err|={np.abs(errors).mean():.2f}"
                )
        report.append("")

        report.append("[sample predictions]")
        for i in range(min(10, len(preds))):
            report.append(
                f"  [{i+1}] pred: {preds[i][:8]}{'...' if len(preds[i]) > 8 else ''}"
            )
            report.append(
                f"       true: {targets[i][:8]}{'...' if len(targets[i]) > 8 else ''}"
            )

    return "\n".join(report)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train RankSchedulePredictor for SDP rank schedule prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data and exp parameters
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16,
        help="Maximum sequence length for rank schedules",
    )

    # training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch-size * accumulation-steps)",
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
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--tf-start",
        type=float,
        default=0.9,
        help="Initial teacher forcing ratio",
    )
    parser.add_argument(
        "--tf-end",
        type=float,
        default=0.2,
        help="Final teacher forcing ratio",
    )

    # loss weights
    parser.add_argument(
        "--schedule-weight",
        type=float,
        default=1.0,
        help="Weight for schedule prediction loss",
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=0.5,
        help="Weight for length prediction loss",
    )
    parser.add_argument(
        "--initial-weight",
        type=float,
        default=0.25,
        help="Weight for auxiliary initial-rank regression loss",
    )
    parser.add_argument(
        "--final-weight",
        type=float,
        default=0.25,
        help="Weight for auxiliary final-rank regression loss",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for length classification",
    )
    parser.add_argument(
        "--under-weight",
        type=float,
        default=2.5,
        help="Multiplier for under-prediction errors",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["cosine", "plateau"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=15,
        help="Number of warmup epochs for cosine scheduler",
    )
    parser.add_argument(
        "--mono-weight",
        type=float,
        default=0.0,
        help="Weight for monotonicity penalty (0 to disable)",
    )

    # model arch parameters
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for GNN layers",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=4,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=2,
        help="Number of decoder layers",
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        default=64,
        help="Hidden dimension for decoder",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )

    # runtime settings
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
        default=4,
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
        help="Use automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable automatic mixed precision",
    )

    args = parser.parse_args()

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
        max_schedule_length=args.max_seq_len,
    )
    print(f"  train samples: {len(train_loader.dataset)}")
    print(f"  val samples: {len(val_loader.dataset)}")
    print(f"  test samples: {len(test_loader.dataset)}")
    print(
        f"  effective batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}"
    )

    sample_data = train_loader.dataset[0]
    node_in_dim = sample_data.x.size(-1)
    edge_in_dim = sample_data.edge_attr.size(-1)
    global_in_dim = sample_data.global_attr.numel()

    print(f"  node features: {node_in_dim}")
    print(f"  edge features: {edge_in_dim}")
    print(f"  global features: {global_in_dim}")

    model_config = {
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "global_in_dim": global_in_dim,
        "hidden_dim": args.hidden_dim,
        "edge_dim": args.hidden_dim // 2,
        "global_dim": args.hidden_dim // 2,
        "num_gnn_layers": args.gnn_layers,
        "num_heads": args.heads,
        "decoder_hidden_dim": args.decoder_hidden,
        "decoder_num_layers": args.decoder_layers,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
    }

    print("\ninitializing model...")
    model = RankSchedulePredictor(**model_config).to(device)
    print(f"  parameters: {model.count_parameters():,}")
    print(f"  {model}")

    criterion = RankScheduleLoss(
        schedule_weight=args.schedule_weight,
        length_weight=args.length_weight,
        mono_weight=args.mono_weight,
        initial_weight=args.initial_weight,
        final_weight=args.final_weight,
        under_weight=args.under_weight,
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-6,
        )
        print(
            f"  using cosine annealing scheduler (warmup: {args.warmup_epochs} epochs)"
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
        print("  using ReduceLROnPlateau scheduler")

    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )
    if scaler:
        print("  using automatic mixed precision (AMP)")

    best_val_metric = float("inf")
    epochs_without_improvement = 0
    training_log: List[Dict] = []

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "tf_start": args.tf_start,
        "tf_end": args.tf_end,
        "schedule_weight": args.schedule_weight,
        "length_weight": args.length_weight,
        "mono_weight": args.mono_weight,
        "initial_weight": args.initial_weight,
        "final_weight": args.final_weight,
        "under_weight": args.under_weight,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
    }

    config = {"model": model_config, "training": training_config}
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nstarting training...")
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda" and epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        epoch_start = time.time()

        tf_ratio = get_teacher_forcing_ratio(
            epoch, args.epochs, args.tf_start, args.tf_end
        )
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=tf_ratio,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=args.amp and device.type == "cuda",
        )

        if args.scheduler == "cosine":
            if epoch <= args.warmup_epochs:
                warmup_lr = args.lr * epoch / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            else:
                scheduler.step()
        else:
            scheduler.step(val_metrics["total_loss"])

        is_best = val_metrics["log_mae"] < best_val_metric
        if is_best:
            best_val_metric = val_metrics["log_mae"]
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                log_dir / "best_model.pt",
                model_config,
            )
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print_epoch_summary(
            epoch, train_losses, val_metrics, current_lr, tf_ratio, is_best, elapsed
        )

        log_entry = {
            "epoch": epoch,
            "train_total_loss": train_losses["total_loss"],
            "train_schedule_loss": train_losses["schedule_loss"],
            "train_length_loss": train_losses["length_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_schedule_loss": val_metrics["schedule_loss"],
            "val_length_loss": val_metrics["length_loss"],
            "val_init_loss": val_metrics["init_loss"],
            "val_final_loss": val_metrics["final_loss"],
            "val_log_mae": val_metrics["log_mae"],
            "val_mae": val_metrics["mae"],
            "val_length_accuracy": val_metrics["length_accuracy"],
            "lr": current_lr,
            "tf_ratio": tf_ratio,
            "is_best": is_best,
        }
        training_log.append(log_entry)

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n[early stopping] no improvement for {args.patience} epochs")
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

    report = eval_report(test_metrics, model_config, training_config)
    print("\n" + report)

    with open(log_dir / "eval_report.txt", "w") as f:
        f.write(report)

    test_results = {
        "predictions": test_metrics["predictions"],
        "targets": test_metrics["targets"],
        "pred_lengths": test_metrics["pred_lengths"],
        "target_lengths": test_metrics["target_lengths"],
    }
    with open(log_dir / "eval_predictions.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nall outputs saved to: {log_dir}")


if __name__ == "__main__":
    main()
