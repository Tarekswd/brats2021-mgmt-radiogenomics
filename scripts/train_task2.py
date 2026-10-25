"""
CLI Training Script for BraTS 2021 Task 2: MGMT Radiogenomic Classification.
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Zero patient-level data leakage, live metric tracking, and checkpoint saving.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add paths
task2_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(task2_root))

from src.models.attention_mil import GatedAttentionMIL
from src.models.cnn_lstm import CNNLSTMClassifier
from src.data.dataset import MGMTDataset
from src.data.splitting import create_patient_folds
from src.evaluation.metrics import compute_mgmt_metrics
from src.utils.logging import get_logger, CSVMetricsLogger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Task 2 MGMT Training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="experiments/task2_live_run", help="Output directory")
    parser.add_argument("--model", type=str, default="attention_mil", choices=["attention_mil", "cnn_lstm"])
    parser.add_argument("--num-slices", type=int, default=16, help="Number of 2D MRI slices per patient")
    parser.add_argument("--slice-size", type=int, default=128, help="Spatial H and W of slices")
    parser.add_argument("--data-root", type=str, default=None, help="Kaggle RSNA-MICCAI train/ folder (one sub-folder per BraTS21ID)")
    parser.add_argument("--labels-csv", type=str, default=None, help="Kaggle train_labels.csv (BraTS21ID, MGMT_value)")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold to train (0..n_folds-1)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified folds")
    parser.add_argument("--exclude", nargs="*", default=["00109", "00123", "00709"],
                        help="Patient IDs to drop (the competition hosts flagged these three as faulty)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()
    if (args.data_root is None) != (args.labels_csv is None):
        parser.error("--data-root and --labels-csv must be given together")

    out_dir = task2_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("Task2Trainer", str(out_dir / "train.log"))
    csv_logger = CSVMetricsLogger(str(out_dir / "training_log.csv"))

    logger.info("=" * 60)
    logger.info("Initiating BraTS 2021 Task 2: MGMT Radiogenomics Training")
    logger.info(f"Target model: {args.model} | Epochs: {args.epochs} | Batch size: {args.batch_size}")
    logger.info(f"Slice configuration: {args.num_slices} slices @ {args.slice_size}x{args.slice_size}")
    logger.info("=" * 60)

    if args.labels_csv:
        import pandas as pd
        df = pd.read_csv(args.labels_csv, dtype={"BraTS21ID": str})
        df["BraTS21ID"] = df["BraTS21ID"].str.zfill(5)
        df = df[~df["BraTS21ID"].isin(args.exclude)]
        missing = [p for p in df["BraTS21ID"] if not (Path(args.data_root) / p).is_dir()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} labelled patients missing under {args.data_root}, e.g. {missing[:5]}")
        patient_ids = df["BraTS21ID"].tolist()
        labels = dict(zip(df["BraTS21ID"], df["MGMT_value"].astype(int)))
        data_root = Path(args.data_root)
        logger.info(f"Real cohort: {len(patient_ids)} patients from {args.labels_csv}")
    else:
        logger.warning("No --data-root/--labels-csv given: SYNTHETIC demo data with random labels. Metrics are meaningless.")
        patient_ids = [f"{i:05d}" for i in range(20)]
        rng = np.random.RandomState(42)
        labels = {pid: int(rng.rand() > 0.45) for pid in patient_ids}
        data_root = task2_root / "data"

    folds = create_patient_folds(
        patient_ids=patient_ids,
        labels=[labels[pid] for pid in patient_ids],
        n_splits=args.n_folds,
        seed=42,
        output_dir=out_dir / "folds"
    )

    fold = folds[args.fold]
    train_pts = fold["train"]
    val_pts = fold["val"]

    logger.info(f"Zero-Leakage Fold {args.fold}: Train = {len(train_pts)} patients, Val = {len(val_pts)} patients")
    logger.info(f"Train Prevalence: {fold['train_prevalence']*100:.1f}% | Val Prevalence: {fold['val_prevalence']*100:.1f}%")

    # Datasets and Loaders
    train_ds = MGMTDataset(
        patient_ids=train_pts,
        data_root=data_root,
        labels=labels,
        num_slices=args.num_slices,
        slice_size=(args.slice_size, args.slice_size),
        is_training=True
    )
    val_ds = MGMTDataset(
        patient_ids=val_pts,
        data_root=data_root,
        labels=labels,
        num_slices=args.num_slices,
        slice_size=(args.slice_size, args.slice_size),
        is_training=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device for Task 2: {device}")

    if args.model == "attention_mil":
        model = GatedAttentionMIL(in_channels=4, embed_dim=128, attn_dim=64).to(device)
    else:
        model = CNNLSTMClassifier(in_channels=4, embed_dim=128, hidden_dim=64).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            vols = batch["volume"].to(device) # [B, 4, S, H, W]
            targets = batch["label"].to(device) # [B]

            optimizer.zero_grad()
            out = model(vols)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            logger.info(f"[Epoch {epoch:02d}] Step {step+1}/{len(train_loader)}: Batch BCE Loss = {batch_loss:.4f}")

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                vols = batch["volume"].to(device)
                targets = batch["label"].to(device)

                out = model(vols)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(1, len(val_loader))
        metrics = compute_mgmt_metrics(all_targets, all_probs, threshold=0.5)
        duration = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1_score": metrics["f1_score"],
            "brier_score": metrics["brier_score"],
            "epoch_time_sec": duration
        }
        csv_logger.log(log_entry)

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.3f} | Acc: {metrics['accuracy']*100:.1f}% | "
            f"Sens: {metrics['sensitivity']*100:.1f}% | Spec: {metrics['specificity']*100:.1f}% | "
            f"Time: {duration:.1f}s"
        )

        is_best = metrics["roc_auc"] > best_auc
        if is_best:
            best_auc = metrics["roc_auc"]
            torch.save(model.state_dict(), ckpt_dir / "best_task2_model.pt")
            logger.info(f"--> Saved new best Task 2 model with ROC-AUC: {best_auc:.3f}")

    logger.info("=" * 60)
    logger.info(f"Task 2 Training Complete. Best Validation ROC-AUC: {best_auc:.3f}")
    logger.info(f"Logs and metrics saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
