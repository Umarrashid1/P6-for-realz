# train_flows_resumable.py
import os
from collections import Counter
from typing import Dict, Optional, List
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
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
from pathlib import Path  # Added for Path object usage

# Assuming utils.train_utils.get_balanced_loss is available if needed,
# but using nn.CrossEntropyLoss as per original train_flows.py for simplicity here.

# Get a default logger for this module.
module_logger = logging.getLogger(__name__)
if not module_logger.hasHandlers():
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    module_logger.addHandler(_handler)
    module_logger.setLevel(logging.INFO)


def _build_loaders(train_ds, val_ds, batch_size: int, sampler_on: bool, num_workers: int = 0,
                   logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = module_logger

    if sampler_on:
        try:
            labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        except KeyError:
            logger.error("Dataset items must have a 'label' key for weighted sampler.")
            raise
        except AttributeError:
            logger.error("Label in dataset must be a tensor for .item() for weighted sampler.")
            raise

        if not labels:
            logger.warning("Training dataset is empty for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        class_counts = Counter(labels)
        if not class_counts:
            logger.warning("Class counts for sampler are empty. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        w_per_class = {c: 1.0 / count for c, count in class_counts.items() if count > 0}
        if not w_per_class:
            logger.warning("All class counts for sampler are zero. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        sample_w = [w_per_class.get(y, 0) for y in labels]
        positive_weights_indices = [i for i, w in enumerate(sample_w) if w > 0]
        if not positive_weights_indices:
            logger.warning("No valid samples with positive weights for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

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


def fine_tune_flow_model_resumable(
        model: nn.Module,
        train_dataset,
        val_dataset,
        optimizer: optim.Optimizer,  # Optimizer is passed in
        start_epoch: int,  # Epoch to start/resume from
        best_val_f1_so_far: float,  # Best F1 score from previous run (if any)
        *,  # Following are keyword-only arguments
        epochs: int = 10,  # Total epochs to run for
        batch_size: int = 64,
        device: str = "cuda",
        clip_grad: float = 1.0,
        use_weighted_sampler: bool = False,
        save_dir: str = "checkpoints_flow_finetuned",
        resumable_checkpoint_name: str = "flow_finetune_checkpoint_latest.pt",
        best_model_weights_name: str = "model_flow_best.pt",
        num_workers_loader: int = 0,
        logger: Optional[logging.Logger] = None
):
    if logger is None:
        logger = module_logger

    save_dir_path = Path(save_dir)
    os.makedirs(save_dir_path, exist_ok=True)

    resumable_checkpoint_path = save_dir_path / resumable_checkpoint_name
    best_model_weights_path = save_dir_path / best_model_weights_name

    logger.info(f"Starting/Resuming FINE-TUNING for a total of {epochs} epochs on device '{device}' using FLOW data.")
    logger.info(f"Will train from epoch {start_epoch} up to {epochs}.")
    logger.info(f"Current best_val_f1_so_far: {best_val_f1_so_far:.4f}")
    logger.info(f"Saving fine-tuned resumable checkpoints to '{resumable_checkpoint_path}'")
    logger.info(f"Saving best model weights to '{best_model_weights_path}'")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler,
                                              num_workers=num_workers_loader, logger=logger)

    current_best_val_f1 = best_val_f1_so_far

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_preds_train = 0
        total_samples_train = 0
        nan_batches_train = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                categorical_flow_data = batch.get("categorical_features")
                if categorical_flow_data is not None and categorical_flow_data.numel() > 0:  # Check if not empty
                    categorical_flow_data = categorical_flow_data.to(device)
                else:
                    categorical_flow_data = None  # Ensure it's None if empty
                labels = batch["label"].to(device)
            except KeyError as e:
                logger.error(f"❌ Batch missing expected key: {e}. Check your flow IoTDataset __getitem__.",
                             exc_info=True)
                continue

            if not torch.isfinite(numerical_flow_data).all():
                num_nans = (~torch.isfinite(numerical_flow_data)).sum().item()
                logger.warning(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: numerical_flow_data contains {num_nans} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue
            if categorical_flow_data is not None and not torch.isfinite(categorical_flow_data.float()).all():
                num_nans_cat = (~torch.isfinite(categorical_flow_data.float())).sum().item()
                logger.warning(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: categorical_flow_data contains {num_nans_cat} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue

            optimizer.zero_grad()
            logits = model(numerical_flow_data, categorical_flow_data)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_batches_train += 1
                if nan_batches_train <= 5 or nan_batches_train % 20 == 0:
                    logger.warning(
                        f"  ❌ Epoch {epoch:02d} NaN/Inf loss (Train Batch {batch_idx}, Count: {nan_batches_train}). Logits min/max: {float(logits.min()):.3e}/{float(logits.max()):.3e}")
                continue

            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_grad)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct_preds_train += (preds == labels).sum().item()
            total_samples_train += labels.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader) - nan_batches_train)
        train_accuracy = correct_preds_train / max(1, total_samples_train)

        model.eval()
        val_preds_list, val_labels_list = [], []
        nan_batches_val = 0
        with torch.no_grad():
            for batch_val in val_loader:  # Renamed batch to batch_val to avoid conflict
                try:
                    numerical_flow_data_val = batch_val["numerical_features"].to(device)
                    categorical_flow_data_val = batch_val.get("categorical_features")
                    if categorical_flow_data_val is not None and categorical_flow_data_val.numel() > 0:
                        categorical_flow_data_val = categorical_flow_data_val.to(device)
                    else:
                        categorical_flow_data_val = None
                    labels_val = batch_val["label"].to(device)
                except KeyError as e:
                    logger.error(f"❌ Validation Batch missing expected key: {e}.", exc_info=True)
                    continue

                if not torch.isfinite(numerical_flow_data_val).all():
                    nan_batches_val += 1
                    continue

                logits_val = model(numerical_flow_data_val, categorical_flow_data_val)
                if not torch.isfinite(logits_val).all():
                    nan_batches_val += 1
                    logger.warning(f"  ❌ Epoch {epoch:02d} NaN/Inf logits in validation. Skipping batch.")
                    continue

                val_preds_list.extend(logits_val.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels_val.cpu().tolist())

        if not val_labels_list:
            logger.warning(f"No validation predictions made for epoch {epoch}. Skipping validation metrics.")
            val_accuracy, macro_f1, weighted_f1 = 0.0, 0.0, 0.0
        else:
            val_accuracy = accuracy_score(val_labels_list, val_preds_list)
            # Use beta=1.0 for F1-score. Setting precision_recall_fscore_support directly
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

        # Save comprehensive checkpoint at the end of each epoch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': current_best_val_f1,  # Save the current best F1 known at this point
            # Add scheduler.state_dict() here if you use a learning rate scheduler
        }
        torch.save(checkpoint_data, resumable_checkpoint_path)
        logger.debug(f"  Saved resumable checkpoint to '{resumable_checkpoint_path}' after epoch {epoch}")

        if macro_f1 > current_best_val_f1:
            current_best_val_f1 = macro_f1
            torch.save(model.state_dict(), best_model_weights_path)  # Save only model weights for "best"
            logger.info(
                f"  -> New best validation Macro-F1: {current_best_val_f1:.4f}. Best model weights saved to '{best_model_weights_path}'")

            # Update the best_val_f1 in the latest resumable checkpoint as well
            checkpoint_data['best_val_f1'] = current_best_val_f1
            torch.save(checkpoint_data, resumable_checkpoint_path)
            logger.debug(f"  Updated best_val_f1 in '{resumable_checkpoint_path}'")

    logger.info("\n" + "=" * 30 + " Flow Fine-tuning Finished " + "=" * 30)
    logger.info(f"Best validation Macro-F1 achieved during this complete run: {current_best_val_f1:.4f}")


def test_flow_model(
        model: nn.Module,
        test_dataset,
        batch_size: int = 64,
        device: str = "cuda",
        model_path: Optional[str] = None,  # Path to best model weights
        logger: Optional[logging.Logger] = None
):
    if logger is None:
        logger = module_logger

    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model state for testing from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Model loaded successfully for testing.")
        except Exception as e:
            logger.error(f"Error loading model state from {model_path}: {e}", exc_info=True)
            logger.info("Proceeding with the model currently in memory (if any).")
    elif model_path:
        logger.warning(f"Specified model_path '{model_path}' not found. Using model currently in memory (if any).")
    else:
        logger.info("No model_path provided for testing. Using model currently in memory.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0)  # num_workers often 0 for test
    model.to(device)
    model.eval()

    logger.info(f"Evaluating FLOW model on Test Set (Device: '{device}')...")
    all_preds, all_labels = [], []
    nan_batches_test = 0

    with torch.no_grad():
        for batch_test in test_loader:  # Renamed batch to batch_test
            try:
                numerical_flow_data_test = batch_test["numerical_features"].to(device)
                categorical_flow_data_test = batch_test.get("categorical_features")
                if categorical_flow_data_test is not None and categorical_flow_data_test.numel() > 0:
                    categorical_flow_data_test = categorical_flow_data_test.to(device)
                else:
                    categorical_flow_data_test = None
                labels_test = batch_test["label"].to(device)
            except KeyError as e:
                logger.error(f"❌ Test Batch missing expected key: {e}.", exc_info=True)
                continue

            if not torch.isfinite(numerical_flow_data_test).all():
                nan_batches_test += 1
                continue

            logits_test = model(numerical_flow_data_test, categorical_flow_data_test)
            if not torch.isfinite(logits_test).all():
                nan_batches_test += 1
                continue

            all_preds.extend(logits_test.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels_test.cpu().tolist())

    logger.info(f"Test completed. NaN batches skipped during test: {nan_batches_test}")
    if not all_labels:
        logger.error("No predictions made from the test set. Check data or NaN issues during testing.")
        return

    logger.info("\n" + "=" * 30 + " Flow Model Test Set Results " + "=" * 30)
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Overall Test Accuracy: {accuracy:.4f}")

    target_names = None
    if hasattr(test_dataset, 'get_target_names'):  # Assuming your dataset might have this method
        try:
            target_names = test_dataset.get_target_names()
        except:  # pylint: disable=bare-except
            pass

    report = classification_report(all_labels, all_preds, digits=4, zero_division=0, target_names=target_names)
    logger.info(f"Classification Report:\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info("\n" + "=" * 78)