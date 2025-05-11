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

from pipeline.config import categorical_columns

__all__ = ["train_model", "test_model"]

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
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
    print("Class‑balanced weights:", weights)
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

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = _balanced_loss(train_dataset, device)
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = seen = nan_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            packet_seq = batch["packet_seq"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cat_feats = {c: batch[c].to(device) for c in categorical_columns}

            # ── Diagnostics & finite‑check before forward ─────────────────
            if batch_idx == 0:  # only print once per epoch
                abs_max = float(packet_seq.abs().max())
                print(f"Epoch {epoch:02d} Batch 0: packet_seq abs‑max = {abs_max:.3e}")
                if abs_max > 1e3:
                    print("⚠️  Features extremely large — check standardisation stats")

            # ---- Finite check ------------------------------------------------
            if not torch.isfinite(packet_seq).all():
                n_nan = (~torch.isfinite(packet_seq)).sum().item()
                print(f"❌ Batch {batch_idx}: packet_seq contains {n_nan} NaNs/Infs — skipping")
                continue
            bad_cat = [c for c,t in cat_feats.items() if not torch.isfinite(t).all()]
            if bad_cat:
                print(f"❌ Batch {batch_idx}: categorical ids non‑finite in columns {bad_cat} — skipping")
                continue

            optimizer.zero_grad()
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            # ── NaN guard ────────────────────────────────────────────────
            if not torch.isfinite(loss):
                nan_batches += 1
                print("❌ NaN/Inf loss — skipping batch.  Logits min/max:",
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
        _, _, macro_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="macro", zero_division=0
        )

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"loss {train_loss:.4f}  "
            f"acc {train_acc:.3f}/{val_acc:.3f}  "
            f"macro‑F1 {macro_f1:.3f}  "
            f"NaN‑batches {nan_batches}"
        )

        torch.save(model.state_dict(), os.path.join(save_dir, f"iot_transformer_ep{epoch}.pt"))


# ────────────────────────────────────────────────────────────────────────────────

def test_model(model: nn.Module, test_dataset, batch_size=64, device="cuda"):
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            packet_seq = batch["packet_seq"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cat_feats = {c: batch[c].to(device) for c in categorical_columns}
            preds = model(packet_seq, cat_feats, attention_mask=attention_mask).argmax(1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    print("\nTest accuracy:", round(acc, 4))
    print("Classification report:\n", classification_report(labels_all, preds_all, digits=4))
    cm = confusion_matrix(labels_all, preds_all)
    print("Confusion matrix:\n", cm)

    for cls, (correct, total) in enumerate(zip(np.diag(cm), cm.sum(1))):
        print(f"Class {cls} acc: {correct/total if total else 0:.4f}")
