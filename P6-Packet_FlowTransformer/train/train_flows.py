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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging  # Import logging


# --- Helper Functions ---

def _build_loaders(train_ds, val_ds, batch_size: int, sampler_on: bool, num_workers: int = 0):
    logging.debug(
        f"Building loaders: Sampler {'ON' if sampler_on else 'OFF'}, Batch Size: {batch_size}, Num Workers: {num_workers}")  # Changed from commented print
    if sampler_on:
        try:
            # Ensure dataset items are accessible and labels exist
            if len(train_ds) == 0:  # Check before trying to access elements
                logging.warning("Training dataset is empty for sampler. Using standard DataLoader.")
                # Fall through to standard loader logic if train_ds is empty
                return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False), \
                    DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=True if num_workers > 0 else False)

            labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        except KeyError:
            logging.error("Dataset items must have a 'label' key for weighted sampler.")  # Changed from raise
            raise  # Re-raise after logging
        except AttributeError:
            logging.error("Label in dataset must be a tensor for .item() for weighted sampler.")  # Changed from raise
            raise  # Re-raise after logging
        except IndexError:  # Handle empty train_ds if not caught by len check earlier
            logging.warning(
                "Training dataset is empty or became empty before label extraction for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        if not labels:
            logging.warning(
                "Warning: Training dataset yielded no labels for sampler. Using standard DataLoader.")  # Changed from print
            # This return path is similar to the one above, could be refactored.
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        class_counts = Counter(labels)
        if not class_counts:
            logging.warning(
                "Warning: Class counts for sampler are empty. Using standard DataLoader.")  # Changed from print
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        w_per_class = {c: 1.0 / count for c, count in class_counts.items() if count > 0}
        if not w_per_class:
            logging.warning(
                "Warning: All class counts for sampler are zero or invalid. Using standard DataLoader.")  # Changed from print
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        sample_w = [w_per_class.get(y, 0) for y in labels]

        # Check if any positive weights exist before creating sampler
        if not any(w > 0 for w in sample_w):
            logging.warning(
                "Warning: No valid samples with positive weights for sampler. Using standard DataLoader.")  # Changed from print
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        try:
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_w), num_samples=len(train_ds),
                                            replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                      pin_memory=True if num_workers > 0 else False)
        except RuntimeError as e:  # Catch potential errors from WeightedRandomSampler (e.g. all weights zero)
            logging.warning(f"Error creating WeightedRandomSampler ({e}). Using standard DataLoader.")
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=True if num_workers > 0 else False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True if num_workers > 0 else False)
    return train_loader, val_loader


def _balanced_loss(train_ds, device) -> nn.CrossEntropyLoss:
    if len(train_ds) == 0:
        logging.warning(
            "Warning: Training dataset is empty for balanced loss. Using unweighted CrossEntropyLoss.")  # Changed from print
        return nn.CrossEntropyLoss()
    try:
        labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    except (KeyError, AttributeError, IndexError):  # Added IndexError
        logging.warning(
            "Warning: Could not extract labels for balanced loss (KeyError, AttributeError, or IndexError). Using unweighted CrossEntropyLoss.")  # Changed from print
        return nn.CrossEntropyLoss()

    if not labels:
        logging.warning(
            "Warning: No labels found in training dataset for balanced loss. Using unweighted CrossEntropyLoss.")  # Changed from print
        return nn.CrossEntropyLoss()

    classes = np.unique(labels)
    if len(classes) <= 1:
        logging.warning(
            "Warning: <= 1 class in training data for balanced loss. Using unweighted CrossEntropyLoss.")  # Changed from print
        return nn.CrossEntropyLoss()

    try:
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        logging.info(
            f"Using class‑balanced weights for classes {classes.tolist()}: {np.round(weights, 3).tolist()}")  # Changed from print, .tolist() for logging
        if not torch.isfinite(weights_t).all() or torch.isnan(weights_t).any():  # More robust check
            logging.warning(
                "⚠ Non‑finite or NaN class weight detected. Using unweighted CrossEntropyLoss.")  # Changed from print
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(weight=weights_t)
    except ValueError as e:
        logging.warning(
            f"Warning: Could not compute class weights ({e}). Using unweighted CrossEntropyLoss.")  # Changed from print
        return nn.CrossEntropyLoss()


# --- New Fine-tuning Function for Flows ---
def fine_tune_flow_model(
        model: nn.Module,
        train_dataset,
        val_dataset,
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        device: str = "cuda",
        clip_grad: float = 1.0,
        use_weighted_sampler: bool = False,
        save_dir: str = "checkpoints_flow_finetuned",
        num_workers_loader: int = 0
):
    os.makedirs(save_dir, exist_ok=True)
    logging.info(
        f"Starting FINE-TUNING for {epochs} epochs on device '{device}' using FLOW data...")  # Changed from print
    logging.info(f"Saving fine-tuned checkpoints to '{save_dir}'")  # Changed from print

    model.to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr)
    logging.info(f"Optimizer AdamW initialized with LR: {lr} for trainable parameters.")  # Changed from print

    criterion = _balanced_loss(train_dataset, device)
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler,
                                              num_workers=num_workers_loader)

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_preds_train = 0
        total_samples_train = 0
        nan_batches_train = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                categorical_flow_data = batch.get("categorical_features")
                if categorical_flow_data is not None:
                    categorical_flow_data = categorical_flow_data.to(device)
                labels = batch["label"].to(device)
            except KeyError as e:
                logging.error(
                    f"❌ Batch missing expected key: {e}. Check your flow IoTDataset __getitem__.")  # Changed from print
                continue

            if not torch.isfinite(numerical_flow_data).all():
                num_nans = (~torch.isfinite(numerical_flow_data)).sum().item()
                logging.warning(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: numerical_flow_data contains {num_nans} NaNs/Infs — skipping")  # Changed from print
                nan_batches_train += 1
                continue
            if categorical_flow_data is not None and not torch.isfinite(categorical_flow_data.float()).all():
                num_nans_cat = (~torch.isfinite(categorical_flow_data.float())).sum().item()
                logging.warning(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: categorical_flow_data contains {num_nans_cat} NaNs/Infs — skipping")  # Changed from print
                nan_batches_train += 1
                continue

            if batch_idx == 0 and epoch == 1:  # Log only once
                abs_max_num = float(numerical_flow_data.abs().max())
                logging.debug(
                    f"  Initial Batch 0 (Flows): numerical_flow_data abs-max = {abs_max_num:.3e}")  # Changed from print
                if abs_max_num > 1e4:
                    logging.warning(
                        "  ⚠️ Flow numerical features seem large — check standardization for flows.")  # Changed from print

            optimizer.zero_grad()
            logits = model(numerical_flow_data, categorical_flow_data)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_batches_train += 1
                if nan_batches_train <= 5 or nan_batches_train % 20 == 0:  # Log periodically
                    logging.warning(
                        f"  ❌ Epoch {epoch:02d} NaN/Inf loss (Train Batch {batch_idx}, Count: {nan_batches_train}). Logits min/max: {float(logits.min()):.3e}/{float(logits.max()):.3e}")  # Changed from print
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
            for batch_val_idx, batch in enumerate(val_loader):  # Added batch_val_idx for logging
                try:
                    numerical_flow_data = batch["numerical_features"].to(device)
                    categorical_flow_data = batch.get("categorical_features")
                    if categorical_flow_data is not None:
                        categorical_flow_data = categorical_flow_data.to(device)
                    labels = batch["label"].to(device)
                except KeyError as e:
                    logging.error(
                        f"❌ Validation Batch {batch_val_idx} missing expected key: {e}.")  # Changed from print
                    continue

                if not torch.isfinite(numerical_flow_data).all():
                    logging.warning(
                        f"  Epoch {epoch:02d} Val Batch {batch_val_idx}: numerical_flow_data contains NaNs/Infs — skipping")
                    nan_batches_val += 1
                    continue

                logits = model(numerical_flow_data, categorical_flow_data)
                if not torch.isfinite(logits).all():
                    nan_batches_val += 1
                    logging.warning(
                        f"  ❌ Epoch {epoch:02d} Val Batch {batch_val_idx}: NaN/Inf logits in validation. Skipping batch.")  # Changed from print
                    continue

                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        if not val_labels_list:
            logging.warning(
                f"Warning: No validation predictions made for epoch {epoch}. Skipping validation metrics.")  # Changed from print
            val_accuracy, macro_f1, weighted_f1 = 0.0, 0.0, 0.0
        else:
            val_accuracy = accuracy_score(val_labels_list, val_preds_list)
            _, _, macro_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="macro",
                                                                zero_division=0)
            _, _, weighted_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="weighted",
                                                                   zero_division=0)

        logging.info(
            f"Epoch {epoch:02d}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Loss (Trn): {avg_train_loss:6.4f} | "
            f"Acc (Trn/Val): {train_accuracy:5.3f}/{val_accuracy:5.3f} | "
            f"F1 (Mac/Wgt - Val): {macro_f1:5.3f}/{weighted_f1:5.3f} | "
            f"NaN Batches (Trn/Val): {nan_batches_train}/{nan_batches_val}"
        )  # Changed from print

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_path = os.path.join(save_dir, "model_flow_best.pt")
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"  -> New best validation Macro-F1: {best_val_f1:.4f}. Saved to '{best_model_path}'")  # Changed from print

    logging.info("=" * 30 + " Flow Fine-tuning Finished " + "=" * 30)  # Changed from print
    logging.info(f"Best validation Macro-F1 achieved: {best_val_f1:.4f}")  # Changed from print

def test_flow_model(
        model: torch.nn.Module,
        test_dataset,
        batch_size: int = 64,
        device: str = "cuda",
        model_path: Optional[str] = None
):
    if model_path and os.path.exists(model_path):
        logging.info(f"Loading model state for testing from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info("Model loaded successfully for testing.")
        except Exception as e:
            logging.error(f"Error loading model state from {model_path}: {e}")
            logging.info("Proceeding with the model currently in memory.")
    elif model_path:
        logging.warning(f"Warning: Specified model_path '{model_path}' not found. Using model currently in memory.")

    # For simplicity, creating a simple loader here. num_workers can be passed if needed.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device)
    model.eval()  # Set model to evaluation mode

    logging.info(f"Evaluating FLOW model on Test Set (Device: '{device}')...")
    all_preds, all_labels = [], []
    nan_batches_test = 0

    with torch.no_grad():
        for batch_test_idx, batch in enumerate(test_loader):
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                categorical_flow_data = batch.get("categorical_features")
                if categorical_flow_data is not None:
                    categorical_flow_data = categorical_flow_data.to(device)
                labels = batch["label"].to(device)
            except KeyError as e:
                logging.error(f"❌ Test Batch {batch_test_idx} missing expected key: {e}.")
                continue

            if not torch.isfinite(numerical_flow_data).all():
                logging.warning(f"  Test Batch {batch_test_idx}: numerical_flow_data contains NaNs/Infs — skipping")
                nan_batches_test += 1
                continue

            logits = model(numerical_flow_data, categorical_flow_data)
            if not torch.isfinite(logits).all():
                logging.warning(f"  Test Batch {batch_test_idx}: NaN/Inf logits in testing. Skipping batch.")
                nan_batches_test += 1
                continue

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    logging.info(f"Test completed. NaN batches skipped: {nan_batches_test}")
    if not all_labels:
        logging.error("Error: No predictions made from the test set. Check data, NaN issues, or if test set was empty.")
        return

    logging.info("=" * 30 + " Flow Model Test Set Results " + "=" * 30)
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Overall Test Accuracy: {accuracy:.4f}")


    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    logging.info(f"Classification Report:\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info("=" * 78)