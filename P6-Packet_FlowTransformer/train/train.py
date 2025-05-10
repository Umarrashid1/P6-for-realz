import os
from collections import Counter
from typing import List, Dict, Optional

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

def _build_loaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    use_weighted_sampler: bool,
):
    """Return DataLoader objects. If *use_weighted_sampler* is True, training
    samples are drawn with replacement using class‑balanced weights.
    """
    if use_weighted_sampler:
        # One weight per sample, inverse‑frequency by label
        labels = [train_dataset[idx]["label"].item() for idx in range(len(train_dataset))]
        class_counts = Counter(labels)
        weights_per_class = {
            c: 1.0 / count for c, count in class_counts.items()
        }
        sample_weights = [weights_per_class[y] for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _balanced_ce_loss(train_dataset, device) -> nn.CrossEntropyLoss:
    """Return CrossEntropyLoss with *balanced* weights computed from the full
    training set label distribution."""
    labels = [train_dataset[idx]["label"].item() for idx in range(len(train_dataset))]
    classes = np.arange(max(labels) + 1)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    clip_grad: float = 1.0,
    use_weighted_sampler: bool = False,
    save_dir: str = "checkpoints",
):
    """Train *model* with class‑balanced loss and per‑class metrics.

    Parameters
    ----------
    model : nn.Module
        The IoTTransformer model.
    train_dataset, val_dataset : torch.utils.data.Dataset
        Datasets already split.
    epochs : int, default 10
    batch_size : int, default 64
    lr : float, default 1e‑3
    device : str, default "cuda"
    clip_grad : float, max gradient norm (0 to disable)
    use_weighted_sampler : bool, if True use class‑balanced sampler in DataLoader.
    save_dir : str, directory for epoch checkpoints.
    """
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = _balanced_ce_loss(train_dataset, device)
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler)

    for epoch in range(1, epochs + 1):
        # ---- Train phase -------------------------------------------------
        model.train()
        total_loss, correct, seen = 0.0, 0, 0

        for batch in train_loader:
            packet_seq = batch["packet_seq"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cat_feats = {col: batch[col].to(device) for col in categorical_columns}

            optimizer.zero_grad()
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)

        train_acc = correct / seen
        train_loss = total_loss / len(train_loader)

        # ---- Validation phase -------------------------------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                packet_seq = batch["packet_seq"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                cat_feats = {col: batch[col].to(device) for col in categorical_columns}

                logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        _, _, macro_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="macro", zero_division=0)

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"loss {train_loss:.3f}  "
            f"acc {train_acc:.3f}/{val_acc:.3f}  "
            f"macro‑F1 {macro_f1:.3f}"
        )

        torch.save(model.state_dict(), os.path.join(save_dir, f"iot_transformer_ep{epoch}.pt"))


def test_model(model: nn.Module, test_dataset, batch_size: int = 64, device: str = "cuda"):
    """Evaluate *model* on *test_dataset* and print detailed metrics."""
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            packet_seq = batch["packet_seq"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cat_feats = {col: batch[col].to(device) for col in categorical_columns}
            preds = model(packet_seq, cat_feats, attention_mask=attention_mask).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:\n", classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)

    cm_diag = np.diag(cm)
    for cls, (correct, total) in enumerate(zip(cm_diag, cm.sum(1))):
        print(f"Class {cls} accuracy: {correct / total if total>0 else 0:.4f}")
