import os
from collections import Counter
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from pipeline.config import categorical_columns_packets

__all__ = ["train_model", "test_model"]

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions (assuming these are unchanged from your original code)
# ────────────────────────────────────────────────────────────────────────────────

def _build_loaders(train_ds, val_ds, batch_size: int, sampler_on: bool):
    """Return DataLoader objects; optionally oversample minority classes."""
    if sampler_on:
        labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        class_counts = Counter(labels)
        w_per_class = {c: 1.0 / cnt for c, cnt in class_counts.items()}
        sample_w = [w_per_class[y] for y in labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _balanced_loss(train_ds, device) -> nn.CrossEntropyLoss:
    """Cross‑entropy with class‑balanced weights.  Prints the weight vector and
    asserts finiteness so NaNs/Inf are caught early."""
    labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    classes = np.arange(max(labels) + 1)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Using class‑balanced weights: {np.round(weights, 3)}") # Slightly cleaner print
    assert torch.isfinite(weights_t).all(), "⚠ Non‑finite class weight detected — check label distribution!"
    return nn.CrossEntropyLoss(weight=weights_t)

# ────────────────────────────────────────────────────────────────────────────────
# Core API
# ────────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    clip_grad: float = 1.0,
    use_weighted_sampler: bool = False,
    save_dir: str = "checkpoints",
):
    """Train with out‑of‑range / NaN guards and detailed logging."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting training for {epochs} epochs on device '{device}'...")
    print(f"Saving checkpoints to '{save_dir}'")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = _balanced_loss(train_dataset, device)
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler)

    best_val_f1 = -1.0 # Track best validation F1 for saving best model (optional but good practice)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = seen = nan_batches = 0


        for batch_idx, batch in enumerate(train_loader): # Original loop without tqdm
            packet_seq = batch["packet_seq"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cat_feats = {c: batch[c].to(device) for c in categorical_columns}

            # ── Diagnostics & finite‑check before forward ─────────────────
            if batch_idx == 0 and epoch == 1: # Only print extensive checks once
                abs_max = float(packet_seq.abs().max())
                print(f"  Initial Batch 0: packet_seq abs‑max = {abs_max:.3e}")
                if abs_max > 1e3:
                    print("  ⚠️ Features extremely large — check standardisation stats")

            if not torch.isfinite(packet_seq).all():
                n_nan = (~torch.isfinite(packet_seq)).sum().item()
                print(f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: packet_seq contains {n_nan} NaNs/Infs — skipping")
                continue
            bad_cat = [c for c, t in cat_feats.items() if not torch.isfinite(t).all()]
            if bad_cat:
                print(f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: categorical ids non‑finite in columns {bad_cat} — skipping")
                continue

            optimizer.zero_grad()
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_batches += 1
                # Reduce frequency of NaN loss prints to avoid flooding console
                if nan_batches <= 5 or nan_batches % 50 == 0:
                    print(f"  ❌ Epoch {epoch:02d} NaN/Inf loss (Count: {nan_batches}) — skipping batch. Logits min/max:",
                          float(logits.min()), float(logits.max()))
                continue

            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)

        train_acc = correct / max(seen, 1)
        train_loss = total_loss / max(len(train_loader) - nan_batches, 1)

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                packet_seq = batch["packet_seq"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                cat_feats = {c: batch[c].to(device) for c in categorical_columns}
                logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, macro_f1, support = precision_recall_fscore_support(
            val_labels, val_preds, average="macro", zero_division=0
        )
        # Also calculate weighted F1, often useful for imbalanced datasets
        _, _, weighted_f1, _ = precision_recall_fscore_support(
             val_labels, val_preds, average="weighted", zero_division=0
        )


        # --- Improved Epoch Summary Print ---
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | " # Show current LR
            f"Loss: {train_loss:6.4f} | "
            f"Acc (Trn/Val): {train_acc:5.3f}/{val_acc:5.3f} | "
            f"F1 (Mac/Wgt): {macro_f1:5.3f}/{weighted_f1:5.3f} | "
            f"NaNs: {nan_batches}"
        )

        # Save checkpoint for the current epoch
        epoch_save_path = os.path.join(save_dir, f"model_ep{epoch}.pt")
        torch.save(model.state_dict(), epoch_save_path)

        # Save best model based on validation macro F1
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_save_path = os.path.join(save_dir, "model_best.pt")
            torch.save(model.state_dict(), best_save_path)
            print(f"  -> New best validation Macro-F1: {best_val_f1:.4f}. Saved to '{best_save_path}'")

    print("\n" + "=" * 30 + " Training Finished " + "=" * 30)
    print(f"Final model state saved for epoch {epochs}.")
    print(f"Best model (Val Macro-F1: {best_val_f1:.4f}) saved to '{os.path.join(save_dir, 'model_best.pt')}'")


# ────────────────────────────────────────────────────────────────────────────────

def test_model(model: nn.Module, test_dataset, batch_size=64, device="cuda", model_path: Optional[str] = None):
    """Evaluate the model on the test set and print detailed results."""

    # --- Load model if path is provided ---
    if model_path:
        print(f"\nLoading model state from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model state: {e}")
            print("Proceeding with the model currently in memory.")

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()

    print(f"\nEvaluating on Test Set (Device: '{device}')...")

    preds_all, labels_all = [], []
    with torch.no_grad():
        # Optional: Add tqdm progress bar here too
        # from tqdm.auto import tqdm
        # test_iterator = tqdm(loader, desc="Testing", leave=False)
        # for batch in test_iterator:

        for batch in loader: # Original loop
            packet_seq = batch["packet_seq"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cat_feats = {c: batch[c].to(device) for c in categorical_columns}
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    # --- Clearer Test Results Section ---
    print("\n" + "=" * 30 + " Test Set Results " + "=" * 30)

    # 1. Overall Accuracy
    acc = accuracy_score(labels_all, preds_all)
    print(f"\nOverall Test Accuracy: {acc:.4f}")

    # 2. Classification Report (Most comprehensive)
    print("\nClassification Report:")
    # Ensure target names are provided if available, otherwise uses indices
    target_names = getattr(test_dataset, 'classes', None) # Attempt to get class names if dataset has them
    report = classification_report(
        labels_all,
        preds_all,
        digits=4,
        zero_division=0,
        target_names=target_names
    )
    print(report)

    # 3. Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels_all, preds_all)
    print(cm)


    # Removed the redundant per-class accuracy loop, as 'recall' in the
    # classification report provides the same information (accuracy per class).

    print("\n" + "=" * 78) # Footer for test results