# train/train.py
import os
from collections import Counter
from typing import Dict, Optional, List  # Added Optional, List
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
import logging  # Import logging

# Project-specific imports (assuming it needs this, adjust if not)
# from pipeline.config import categorical_columns_packets # This was in your original train.py

# Get a default logger for this module
module_logger = logging.getLogger(__name__)
if not module_logger.hasHandlers():
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    module_logger.addHandler(_handler)
    module_logger.setLevel(logging.INFO)



# ────────────────────────────────────────────────────────────────────────────────
# Helper functions (Updated)
# ────────────────────────────────────────────────────────────────────────────────

# Modified to accept and use num_workers and logger
def _build_loaders(train_ds, val_ds, batch_size: int, sampler_on: bool, num_workers: int = 0,
                   logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = module_logger

    # logger.info(f"Building loaders: Sampler {'ON' if sampler_on else 'OFF'}, BatchSize: {batch_size}, NumWorkers: {num_workers}")
    if sampler_on:
        # Ensure labels are available and valid for sampling
        if not hasattr(train_ds, '__getitem__') or not hasattr(train_ds, '__len__') or len(train_ds) == 0:
            logger.warning("Training dataset is invalid or empty for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        try:
            labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(
                f"Error accessing labels for weighted sampler: {e}. Ensure dataset items have 'label' as a tensor. Using standard DataLoader.",
                exc_info=True)
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        if not labels:
            logger.warning("No labels extracted for sampler. Using standard DataLoader.")
            # (Same return as above)
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        class_counts = Counter(labels)
        # ... (rest of your sampler logic, replacing print with logger.warning) ...
        if not class_counts:  # Example
            logger.warning("Class counts for sampler are empty. Using standard DataLoader.")
            # (Same return as above)
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        w_per_class = {c: 1.0 / cnt for c, cnt in class_counts.items() if cnt > 0}
        if not w_per_class:
            logger.warning("w_per_class for sampler is empty. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        sample_w = [w_per_class.get(y, 0) for y in
                    labels]  # get(y,0) handles cases where a label might not be in w_per_class

        # Filter out samples with zero weight to avoid error with WeightedRandomSampler if any
        valid_indices = [i for i, w in enumerate(sample_w) if w > 0]
        if not valid_indices:
            logger.warning("No samples with positive weights for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        # If you only want to sample from valid_indices, you'd need to use a Subset.
        # Simpler: WeightedRandomSampler expects weights for all original samples.
        # Ensure all weights are positive or handle cases where some classes might have 0 samples / 0 weight.
        # For now, assuming labels list is for the full train_ds and w_per_class handles it.

        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_w), num_samples=len(train_ds),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True if num_workers > 0 else False)
    return train_loader, val_loader


# Modified to accept and use logger
def _balanced_loss(train_ds, device, logger: Optional[logging.Logger] = None) -> nn.CrossEntropyLoss:
    if logger is None:
        logger = module_logger

    if len(train_ds) == 0:  # Added check for empty train_ds
        logger.warning("Training dataset is empty for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    try:  # Added try-except for robustness
        labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"Error accessing labels for balanced loss: {e}. Using unweighted CrossEntropyLoss.",
                     exc_info=True)
        return nn.CrossEntropyLoss()

    if not labels:  # Added check for empty labels list
        logger.warning("No labels extracted for balanced loss computation. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    classes = np.unique(labels)  # More robust way to get actual classes present
    if len(classes) <= 1:  # Handle if only one class or no class variation
        logger.warning(
            f"Only {len(classes)} unique class(es) found in training data for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info(f"Using class‑balanced weights for classes {classes}: {np.round(weights_t.cpu().numpy(), 3)}")
    if not torch.isfinite(weights_t).all():
        logger.error(
            "⚠ Non‑finite class weight detected — check label distribution! Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=weights_t)


# ────────────────────────────────────────────────────────────────────────────────
# Core API (Updated)
# ────────────────────────────────────────────────────────────────────────────────

# Modified to accept logger and num_workers
def train_model(
        model: nn.Module,
        train_dataset,
        val_dataset,
        categorical_columns: List[str],  # Added to know which cat_feats to expect
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 3e-4,
        device: str = "cuda",
        clip_grad: float = 1.0,
        use_weighted_sampler: bool = False,
        save_dir: str = "checkpoints",
        logger: Optional[logging.Logger] = None,  # <<<< NEW
        num_workers: int = 0  # <<<< NEW
):
    if logger is None:
        logger = module_logger

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Starting training for {epochs} epochs on device '{device}'...")
    logger.info(f"Saving checkpoints to '{save_dir}'")
    logger.info(f"Using {num_workers} workers for DataLoaders.")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = _balanced_loss(train_dataset, device, logger=logger)  # Pass logger
    train_loader, val_loader = _build_loaders(
        train_dataset, val_dataset, batch_size, use_weighted_sampler,
        num_workers=num_workers, logger=logger  # Pass num_workers and logger
    )

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_preds_train = 0
        total_samples_train = 0
        nan_batches_train = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                packet_seq = batch["packet_seq"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                # Dynamically get categorical features based on provided list
                cat_feats = {c: batch[c].to(device) for c in categorical_columns if c in batch}
            except KeyError as e:
                logger.error(f"❌ Batch missing expected key: {e}. Check your IoTSequenceDataset __getitem__.",
                             exc_info=True)
                continue

            if batch_idx == 0 and epoch == 1:
                abs_max = float(packet_seq.abs().max())
                logger.info(f"  Initial Batch 0: packet_seq abs‑max = {abs_max:.3e}")
                if abs_max > 1e3:  # Reduced from 1e4 for packet data based on original script's diagnostics
                    logger.warning("  ⚠️ Features extremely large — check standardisation stats")

            if not torch.isfinite(packet_seq).all():
                n_nan = (~torch.isfinite(packet_seq)).sum().item()
                logger.warning(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: packet_seq contains {n_nan} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue

            bad_cat_cols = []
            for c, t in cat_feats.items():
                if not torch.isfinite(t.float()).all():  # Cast to float for check if int tensor
                    bad_cat_cols.append(c)
            if bad_cat_cols:
                logger.warning(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: categorical ids non‑finite in columns {bad_cat_cols} — skipping")
                nan_batches_train += 1
                continue

            optimizer.zero_grad()
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_batches_train += 1
                if nan_batches_train <= 5 or nan_batches_train % 50 == 0:
                    logger.warning(
                        f"  ❌ Epoch {epoch:02d} NaN/Inf loss (Train Batch {batch_idx}, Count: {nan_batches_train}). Logits min/max: {float(logits.min()):.3e}/{float(logits.max()):.3e}")
                continue

            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct_preds_train += (preds == labels).sum().item()
            total_samples_train += labels.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader) - nan_batches_train)
        train_accuracy = correct_preds_train / max(1, total_samples_train)

        model.eval()
        val_preds_list, val_labels_list = [], []
        nan_batches_val = 0  # Added for consistency
        with torch.no_grad():
            for batch in val_loader:
                try:  # Added try-except for robustness during validation loading
                    packet_seq = batch["packet_seq"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    cat_feats = {c: batch[c].to(device) for c in categorical_columns if c in batch}
                except KeyError as e:
                    logger.error(f"❌ Validation Batch missing expected key: {e}.", exc_info=True)
                    continue

                # Added NaN check for validation inputs
                if not torch.isfinite(packet_seq).all():
                    nan_batches_val += 1
                    logger.warning(f"  ❌ Epoch {epoch:02d} NaN/Inf packet_seq in validation. Skipping batch.")
                    continue

                logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
                if not torch.isfinite(logits).all():  # Check logits for NaNs
                    nan_batches_val += 1
                    logger.warning(f"  ❌ Epoch {epoch:02d} NaN/Inf logits in validation. Skipping batch.")
                    continue

                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        if not val_labels_list:  # Handle empty validation results
            logger.warning(f"No validation predictions made for epoch {epoch}. Skipping validation metrics.")
            val_accuracy, macro_f1, weighted_f1 = 0.0, 0.0, 0.0
        else:
            val_accuracy = accuracy_score(val_labels_list, val_preds_list)
            _, _, macro_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="macro",
                                                                zero_division=0)
            _, _, weighted_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="weighted",
                                                                   zero_division=0)

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Loss (Trn): {avg_train_loss:6.4f} | "
            f"Acc (Trn/Val): {train_accuracy:5.3f}/{val_accuracy:5.3f} | "
            f"F1 (Mac/Wgt - Val): {macro_f1:5.3f}/{weighted_f1:5.3f} | "
            f"NaN Batches (Trn/Val): {nan_batches_train}/{nan_batches_val}"
        )

        epoch_save_path = os.path.join(save_dir, f"model_ep{epoch}.pt")
        torch.save(model.state_dict(), epoch_save_path)  # Save epoch checkpoint

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_save_path = os.path.join(save_dir, "model_best.pt")
            torch.save(model.state_dict(), best_save_path)
            logger.info(f"  -> New best validation Macro-F1: {best_val_f1:.4f}. Saved to '{best_save_path}'")

    logger.info("\n" + "=" * 30 + " Training Finished " + "=" * 30)
    logger.info(f"Final model state saved for epoch {epochs} to {epoch_save_path}.")  # Corrected variable
    logger.info(f"Best model (Val Macro-F1: {best_val_f1:.4f}) saved to '{os.path.join(save_dir, 'model_best.pt')}'")


# ────────────────────────────────────────────────────────────────────────────────

# Modified to accept logger and num_workers (for consistency if test_loader uses it)
def test_model(
        model: nn.Module,
        test_dataset,
        categorical_columns: List[str],  # Added to know which cat_feats to expect
        batch_size: int = 64,
        device: str = "cuda",
        model_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,  # <<<< NEW
        num_workers: int = 0  # <<<< NEW for DataLoader consistency
):
    if logger is None:
        logger = module_logger

    if model_path and os.path.exists(model_path):  # Added os.path.exists check
        logger.info(f"Loading model state for testing from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model state from {model_path}: {e}", exc_info=True)
            logger.info("Proceeding with the model currently in memory.")
    elif model_path:  # If path provided but not found
        logger.warning(f"Specified model_path '{model_path}' not found. Using model currently in memory.")

    # Use num_workers here as well if desired for testing
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True if num_workers > 0 else False)
    model.to(device).eval()

    logger.info(f"Evaluating on Test Set (Device: '{device}', NumWorkers: {num_workers})...")

    preds_all, labels_all = [], []
    nan_batches_test = 0  # Added
    with torch.no_grad():
        for batch in loader:
            try:  # Added try-except
                packet_seq = batch["packet_seq"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                cat_feats = {c: batch[c].to(device) for c in categorical_columns if c in batch}
            except KeyError as e:
                logger.error(f"❌ Test Batch missing expected key: {e}.", exc_info=True)
                continue

            # Added NaN check for test inputs
            if not torch.isfinite(packet_seq).all():
                nan_batches_test += 1
                logger.warning(f"  ❌ NaN/Inf packet_seq in test data. Skipping batch.")
                continue

            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            if not torch.isfinite(logits).all():  # Check logits for NaNs
                nan_batches_test += 1
                logger.warning(f"  ❌ NaN/Inf logits in test data. Skipping batch.")
                continue

            preds = logits.argmax(dim=1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    logger.info(f"Test completed. NaN batches skipped: {nan_batches_test}")
    if not labels_all:  # Handle empty results
        logger.error("No predictions made from the test set. Check data or NaN issues.")
        return

    logger.info("\n" + "=" * 30 + " Test Set Results " + "=" * 30)
    acc = accuracy_score(labels_all, preds_all)
    logger.info(f"Overall Test Accuracy: {acc:.4f}")

    target_names = None
    if hasattr(test_dataset, 'classes'):  # Check for 'classes' attribute
        target_names = test_dataset.classes
    elif hasattr(test_dataset, 'get_target_names'):  # Your original check
        try:
            target_names = test_dataset.get_target_names()
        except Exception as e:
            logger.warning(f"Could not get target names from dataset: {e}")
            pass

    report = classification_report(
        labels_all,
        preds_all,
        digits=4,
        zero_division=0,
        target_names=target_names
    )
    logger.info(f"Classification Report:\n{report}")

    cm = confusion_matrix(labels_all, preds_all)
    # Could log cm as a string or use a library to pretty print if needed for file logs
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info("\n" + "=" * 78)