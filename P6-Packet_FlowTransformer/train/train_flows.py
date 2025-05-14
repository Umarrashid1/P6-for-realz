# train_flows.py (or add to your existing train.py)
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


# Assuming these helper functions are accessible (e.g., from the original train.py or a shared utils)
# If they are in the same file, no import needed. If in another, adjust import.
# from .train_utils import _build_loaders, _balanced_loss # Example if moved to train_utils.py

# --- Re-define or Import Helper Functions ---
# For clarity, I'm including them here. If they are in another file, import them.

def _build_loaders(train_ds, val_ds, batch_size: int, sampler_on: bool, num_workers: int = 0):  # Added num_workers
    """Return DataLoader objects; optionally oversample minority classes."""
    # print(f"Building loaders: Sampler {'ON' if sampler_on else 'OFF'}, Batch Size: {batch_size}, Num Workers: {num_workers}")
    if sampler_on:
        try:
            labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        except KeyError:
            raise KeyError("Dataset items must have a 'label' key for weighted sampler.")
        except AttributeError:
            raise AttributeError("Label in dataset must be a tensor for .item() for weighted sampler.")

        if not labels:  # Handle empty train_ds
            print("Warning: Training dataset is empty for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        class_counts = Counter(labels)
        if not class_counts:
            print("Warning: Class counts for sampler are empty. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        w_per_class = {c: 1.0 / count for c, count in class_counts.items() if count > 0}
        if not w_per_class:
            print("Warning: All class counts for sampler are zero. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        sample_w = [w_per_class.get(y, 0) for y in labels]
        # Ensure all weights are positive for WeightedRandomSampler
        positive_weights_indices = [i for i, w in enumerate(sample_w) if w > 0]
        if not positive_weights_indices:
            print("Warning: No valid samples with positive weights for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        # If some samples had 0 weight, the sampler might behave unexpectedly or error.
        # It's safer to ensure all weights passed to sampler are > 0 for the samples it considers.
        # However, WeightedRandomSampler expects weights for *all* samples in the dataset.
        # A common approach if some weights are 0 is to either filter the dataset
        # or add a very small epsilon to zero weights if the sampler must see all indices.
        # For now, assume that if a label exists, its count > 0.

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


def _balanced_loss(train_ds, device) -> nn.CrossEntropyLoss:
    """Cross‑entropy with class‑balanced weights."""
    if len(train_ds) == 0:
        print("Warning: Training dataset is empty for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    try:
        labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    except (KeyError, AttributeError):
        print("Warning: Could not extract labels for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    if not labels:
        print("Warning: No labels found in training dataset for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    classes = np.unique(labels)  # Use unique labels present in the data
    if len(classes) <= 1:  # Handle cases with only one class or no class variation
        print("Warning: <= 1 class in training data for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    try:
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Using class‑balanced weights for classes {classes}: {np.round(weights, 3)}")
        if not torch.isfinite(weights_t).all():
            print("⚠ Non‑finite class weight detected. Using unweighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(weight=weights_t)
    except ValueError as e:  # compute_class_weight can raise ValueError for certain label distributions
        print(f"Warning: Could not compute class weights ({e}). Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


# --- New Fine-tuning Function for Flows ---
def fine_tune_flow_model(
        model: nn.Module,  # Expected to be FlowFineTuningModel
        train_dataset,  # IoTDataset returning flow data
        val_dataset,  # IoTDataset returning flow data
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,  # Typically smaller for fine-tuning
        device: str = "cuda",
        clip_grad: float = 1.0,
        use_weighted_sampler: bool = False,
        save_dir: str = "checkpoints_flow_finetuned",
        num_workers_loader: int = 0  # Added for DataLoader
):
    """Fine-tune a model on flow data."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting FINE-TUNING for {epochs} epochs on device '{device}' using FLOW data...")
    print(f"Saving fine-tuned checkpoints to '{save_dir}'")

    model.to(device)

    # --- Optimizer for Fine-tuning: Only optimize unfrozen parameters ---
    # This assumes layers to be frozen have param.requires_grad = False
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr)
    print(f"Optimizer AdamW initialized with LR: {lr} for trainable parameters.")

    criterion = _balanced_loss(train_dataset, device)
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler,
                                              num_workers=num_workers_loader)

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        total_loss = 0.0
        correct_preds_train = 0
        total_samples_train = 0
        nan_batches_train = 0

        for batch_idx, batch in enumerate(train_loader):
            # --- Adapt data unpacking for FLOW data ---
            # These keys must match what your flow IoTDataset returns
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                # Assuming categorical_features is a single tensor [B, NumCatFlowFeatures]
                # If it's a dict, this needs to change.
                categorical_flow_data = batch.get("categorical_features")  # Optional
                if categorical_flow_data is not None:
                    categorical_flow_data = categorical_flow_data.to(device)
                labels = batch["label"].to(device)
            except KeyError as e:
                print(f"❌ Batch missing expected key: {e}. Check your flow IoTDataset __getitem__.")
                # Skip batch or raise error
                continue

            # --- Basic NaN/Inf check for flow input features ---
            if not torch.isfinite(numerical_flow_data).all():
                num_nans = (~torch.isfinite(numerical_flow_data)).sum().item()
                print(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: numerical_flow_data contains {num_nans} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue
            if categorical_flow_data is not None and not torch.isfinite(
                    categorical_flow_data.float()).all():  # Cast to float for check if int
                num_nans_cat = (~torch.isfinite(categorical_flow_data.float())).sum().item()
                print(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: categorical_flow_data contains {num_nans_cat} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue

            # --- Diagnostics (Optional, adapt for flows) ---
            if batch_idx == 0 and epoch == 1:
                abs_max_num = float(numerical_flow_data.abs().max())
                print(f"  Initial Batch 0 (Flows): numerical_flow_data abs-max = {abs_max_num:.3e}")
                if abs_max_num > 1e4:  # Adjusted threshold
                    print("  ⚠️ Flow numerical features seem large — check standardization for flows.")

            optimizer.zero_grad()

            # --- Adapt model call for FlowFineTuningModel ---
            # The forward pass of FlowFineTuningModel expects (flow_numerical_data, flow_categorical_data)
            logits = model(numerical_flow_data, categorical_flow_data)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_batches_train += 1
                if nan_batches_train <= 5 or nan_batches_train % 20 == 0:
                    print(
                        f"  ❌ Epoch {epoch:02d} NaN/Inf loss (Train Batch {batch_idx}, Count: {nan_batches_train}). Logits min/max: {float(logits.min()):.3e}/{float(logits.max()):.3e}")
                continue

            loss.backward()
            if clip_grad > 0:
                # Clip gradients only for trainable parameters
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_grad)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct_preds_train += (preds == labels).sum().item()
            total_samples_train += labels.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader) - nan_batches_train)
        train_accuracy = correct_preds_train / max(1, total_samples_train)

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        val_preds_list, val_labels_list = [], []
        nan_batches_val = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    numerical_flow_data = batch["numerical_features"].to(device)
                    categorical_flow_data = batch.get("categorical_features")
                    if categorical_flow_data is not None:
                        categorical_flow_data = categorical_flow_data.to(device)
                    labels = batch["label"].to(device)
                except KeyError as e:
                    print(f"❌ Validation Batch missing expected key: {e}.")
                    continue

                if not torch.isfinite(numerical_flow_data).all():
                    nan_batches_val += 1
                    continue  # Skip batch with NaNs

                logits = model(numerical_flow_data, categorical_flow_data)
                if not torch.isfinite(logits).all():  # Check logits for NaNs
                    nan_batches_val += 1
                    print(f"  ❌ Epoch {epoch:02d} NaN/Inf logits in validation. Skipping batch.")
                    continue

                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        if not val_labels_list:  # Handle case where validation set might be empty or all batches had NaNs
            print(f"Warning: No validation predictions made for epoch {epoch}. Skipping validation metrics.")
            val_accuracy, macro_f1, weighted_f1 = 0.0, 0.0, 0.0
        else:
            val_accuracy = accuracy_score(val_labels_list, val_preds_list)
            # Use zero_division=0 for precision_recall_fscore_support
            _, _, macro_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="macro",
                                                                zero_division=0)
            _, _, weighted_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="weighted",
                                                                   zero_division=0)

        print(
            f"Epoch {epoch:02d}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Loss (Trn): {avg_train_loss:6.4f} | "
            f"Acc (Trn/Val): {train_accuracy:5.3f}/{val_accuracy:5.3f} | "
            f"F1 (Mac/Wgt - Val): {macro_f1:5.3f}/{weighted_f1:5.3f} | "
            f"NaN Batches (Trn/Val): {nan_batches_train}/{nan_batches_val}"
        )

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_path = os.path.join(save_dir, "model_flow_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best validation Macro-F1: {best_val_f1:.4f}. Saved to '{best_model_path}'")

        # Save epoch checkpoint (optional)
        # epoch_model_path = os.path.join(save_dir, f"model_flow_ep{epoch}.pt")
        # torch.save(model.state_dict(), epoch_model_path)

    print("\n" + "=" * 30 + " Flow Fine-tuning Finished " + "=" * 30)
    print(f"Best validation Macro-F1 achieved: {best_val_f1:.4f}")


# --- New Test Function for Flows ---
def test_flow_model(
        model: nn.Module,  # Expected to be FlowFineTuningModel or similar
        test_dataset,  # IoTDataset returning flow data
        batch_size: int = 64,
        device: str = "cuda",
        model_path: Optional[str] = None  # Path to load specific model weights
):
    """Evaluate a flow-based model on the test set."""
    if model_path and os.path.exists(model_path):
        print(f"\nLoading model state for testing from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully for testing.")
        except Exception as e:
            print(f"Error loading model state from {model_path}: {e}")
            print("Proceeding with the model currently in memory.")
    elif model_path:
        print(f"Warning: Specified model_path '{model_path}' not found. Using model currently in memory.")

    # Use _build_loaders to get a test_loader, num_workers can be passed if needed
    # For simplicity, creating a simple loader here.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0)  # Adjust num_workers if needed

    model.to(device)
    model.eval()  # Set model to evaluation mode

    print(f"\nEvaluating FLOW model on Test Set (Device: '{device}')...")
    all_preds, all_labels = [], []
    nan_batches_test = 0

    with torch.no_grad():
        for batch in test_loader:
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                categorical_flow_data = batch.get("categorical_features")
                if categorical_flow_data is not None:
                    categorical_flow_data = categorical_flow_data.to(device)
                labels = batch["label"].to(device)
            except KeyError as e:
                print(f"❌ Test Batch missing expected key: {e}.")
                continue

            if not torch.isfinite(numerical_flow_data).all():
                nan_batches_test += 1
                continue  # Skip batch with NaNs

            logits = model(numerical_flow_data, categorical_flow_data)
            if not torch.isfinite(logits).all():
                nan_batches_test += 1
                continue

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print(f"Test completed. NaN batches skipped: {nan_batches_test}")
    if not all_labels:
        print("Error: No predictions made from the test set. Check data or NaN issues.")
        return

    print("\n" + "=" * 30 + " Flow Model Test Set Results " + "=" * 30)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")

    # Attempt to get target names if dataset object supports it
    target_names = None
    if hasattr(test_dataset, 'get_target_names'):  # Example method name
        try:
            target_names = test_dataset.get_target_names()
        except:
            pass  # Ignore if method fails or doesn't exist

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0, target_names=target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n" + "=" * 78)


# train_flows.py (or add to your existing train.py)
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


# Assuming these helper functions are accessible (e.g., from the original train.py or a shared utils)
# If they are in the same file, no import needed. If in another, adjust import.
# from .train_utils import _build_loaders, _balanced_loss # Example if moved to train_utils.py

# --- Re-define or Import Helper Functions ---
# For clarity, I'm including them here. If they are in another file, import them.

def _build_loaders(train_ds, val_ds, batch_size: int, sampler_on: bool, num_workers: int = 0):  # Added num_workers
    """Return DataLoader objects; optionally oversample minority classes."""
    # print(f"Building loaders: Sampler {'ON' if sampler_on else 'OFF'}, Batch Size: {batch_size}, Num Workers: {num_workers}")
    if sampler_on:
        try:
            labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        except KeyError:
            raise KeyError("Dataset items must have a 'label' key for weighted sampler.")
        except AttributeError:
            raise AttributeError("Label in dataset must be a tensor for .item() for weighted sampler.")

        if not labels:  # Handle empty train_ds
            print("Warning: Training dataset is empty for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        class_counts = Counter(labels)
        if not class_counts:
            print("Warning: Class counts for sampler are empty. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        w_per_class = {c: 1.0 / count for c, count in class_counts.items() if count > 0}
        if not w_per_class:
            print("Warning: All class counts for sampler are zero. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        sample_w = [w_per_class.get(y, 0) for y in labels]
        # Ensure all weights are positive for WeightedRandomSampler
        positive_weights_indices = [i for i, w in enumerate(sample_w) if w > 0]
        if not positive_weights_indices:
            print("Warning: No valid samples with positive weights for sampler. Using standard DataLoader.")
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True if num_workers > 0 else False), \
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True if num_workers > 0 else False)

        # If some samples had 0 weight, the sampler might behave unexpectedly or error.
        # It's safer to ensure all weights passed to sampler are > 0 for the samples it considers.
        # However, WeightedRandomSampler expects weights for *all* samples in the dataset.
        # A common approach if some weights are 0 is to either filter the dataset
        # or add a very small epsilon to zero weights if the sampler must see all indices.
        # For now, assume that if a label exists, its count > 0.

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


def _balanced_loss(train_ds, device) -> nn.CrossEntropyLoss:
    """Cross‑entropy with class‑balanced weights."""
    if len(train_ds) == 0:
        print("Warning: Training dataset is empty for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    try:
        labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    except (KeyError, AttributeError):
        print("Warning: Could not extract labels for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    if not labels:
        print("Warning: No labels found in training dataset for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    classes = np.unique(labels)  # Use unique labels present in the data
    if len(classes) <= 1:  # Handle cases with only one class or no class variation
        print("Warning: <= 1 class in training data for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    try:
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Using class‑balanced weights for classes {classes}: {np.round(weights, 3)}")
        if not torch.isfinite(weights_t).all():
            print("⚠ Non‑finite class weight detected. Using unweighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(weight=weights_t)
    except ValueError as e:  # compute_class_weight can raise ValueError for certain label distributions
        print(f"Warning: Could not compute class weights ({e}). Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


# --- New Fine-tuning Function for Flows ---
def fine_tune_flow_model(
        model: nn.Module,  # Expected to be FlowFineTuningModel
        train_dataset,  # IoTDataset returning flow data
        val_dataset,  # IoTDataset returning flow data
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,  # Typically smaller for fine-tuning
        device: str = "cuda",
        clip_grad: float = 1.0,
        use_weighted_sampler: bool = False,
        save_dir: str = "checkpoints_flow_finetuned",
        num_workers_loader: int = 0  # Added for DataLoader
):
    """Fine-tune a model on flow data."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting FINE-TUNING for {epochs} epochs on device '{device}' using FLOW data...")
    print(f"Saving fine-tuned checkpoints to '{save_dir}'")

    model.to(device)

    # --- Optimizer for Fine-tuning: Only optimize unfrozen parameters ---
    # This assumes layers to be frozen have param.requires_grad = False
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr)
    print(f"Optimizer AdamW initialized with LR: {lr} for trainable parameters.")

    criterion = _balanced_loss(train_dataset, device)
    train_loader, val_loader = _build_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler,
                                              num_workers=num_workers_loader)

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        total_loss = 0.0
        correct_preds_train = 0
        total_samples_train = 0
        nan_batches_train = 0

        for batch_idx, batch in enumerate(train_loader):
            # --- Adapt data unpacking for FLOW data ---
            # These keys must match what your flow IoTDataset returns
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                # Assuming categorical_features is a single tensor [B, NumCatFlowFeatures]
                # If it's a dict, this needs to change.
                categorical_flow_data = batch.get("categorical_features")  # Optional
                if categorical_flow_data is not None:
                    categorical_flow_data = categorical_flow_data.to(device)
                labels = batch["label"].to(device)
            except KeyError as e:
                print(f"❌ Batch missing expected key: {e}. Check your flow IoTDataset __getitem__.")
                # Skip batch or raise error
                continue

            # --- Basic NaN/Inf check for flow input features ---
            if not torch.isfinite(numerical_flow_data).all():
                num_nans = (~torch.isfinite(numerical_flow_data)).sum().item()
                print(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: numerical_flow_data contains {num_nans} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue
            if categorical_flow_data is not None and not torch.isfinite(
                    categorical_flow_data.float()).all():  # Cast to float for check if int
                num_nans_cat = (~torch.isfinite(categorical_flow_data.float())).sum().item()
                print(
                    f"  ❌ Epoch {epoch:02d} Batch {batch_idx}: categorical_flow_data contains {num_nans_cat} NaNs/Infs — skipping")
                nan_batches_train += 1
                continue

            # --- Diagnostics (Optional, adapt for flows) ---
            if batch_idx == 0 and epoch == 1:
                abs_max_num = float(numerical_flow_data.abs().max())
                print(f"  Initial Batch 0 (Flows): numerical_flow_data abs-max = {abs_max_num:.3e}")
                if abs_max_num > 1e4:  # Adjusted threshold
                    print("  ⚠️ Flow numerical features seem large — check standardization for flows.")

            optimizer.zero_grad()

            # --- Adapt model call for FlowFineTuningModel ---
            # The forward pass of FlowFineTuningModel expects (flow_numerical_data, flow_categorical_data)
            logits = model(numerical_flow_data, categorical_flow_data)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_batches_train += 1
                if nan_batches_train <= 5 or nan_batches_train % 20 == 0:
                    print(
                        f"  ❌ Epoch {epoch:02d} NaN/Inf loss (Train Batch {batch_idx}, Count: {nan_batches_train}). Logits min/max: {float(logits.min()):.3e}/{float(logits.max()):.3e}")
                continue

            loss.backward()
            if clip_grad > 0:
                # Clip gradients only for trainable parameters
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_grad)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct_preds_train += (preds == labels).sum().item()
            total_samples_train += labels.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader) - nan_batches_train)
        train_accuracy = correct_preds_train / max(1, total_samples_train)

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        val_preds_list, val_labels_list = [], []
        nan_batches_val = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    numerical_flow_data = batch["numerical_features"].to(device)
                    categorical_flow_data = batch.get("categorical_features")
                    if categorical_flow_data is not None:
                        categorical_flow_data = categorical_flow_data.to(device)
                    labels = batch["label"].to(device)
                except KeyError as e:
                    print(f"❌ Validation Batch missing expected key: {e}.")
                    continue

                if not torch.isfinite(numerical_flow_data).all():
                    nan_batches_val += 1
                    continue  # Skip batch with NaNs

                logits = model(numerical_flow_data, categorical_flow_data)
                if not torch.isfinite(logits).all():  # Check logits for NaNs
                    nan_batches_val += 1
                    print(f"  ❌ Epoch {epoch:02d} NaN/Inf logits in validation. Skipping batch.")
                    continue

                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        if not val_labels_list:  # Handle case where validation set might be empty or all batches had NaNs
            print(f"Warning: No validation predictions made for epoch {epoch}. Skipping validation metrics.")
            val_accuracy, macro_f1, weighted_f1 = 0.0, 0.0, 0.0
        else:
            val_accuracy = accuracy_score(val_labels_list, val_preds_list)
            # Use zero_division=0 for precision_recall_fscore_support
            _, _, macro_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="macro",
                                                                zero_division=0)
            _, _, weighted_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds_list, average="weighted",
                                                                   zero_division=0)

        print(
            f"Epoch {epoch:02d}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Loss (Trn): {avg_train_loss:6.4f} | "
            f"Acc (Trn/Val): {train_accuracy:5.3f}/{val_accuracy:5.3f} | "
            f"F1 (Mac/Wgt - Val): {macro_f1:5.3f}/{weighted_f1:5.3f} | "
            f"NaN Batches (Trn/Val): {nan_batches_train}/{nan_batches_val}"
        )

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_path = os.path.join(save_dir, "model_flow_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best validation Macro-F1: {best_val_f1:.4f}. Saved to '{best_model_path}'")

        # Save epoch checkpoint (optional)
        # epoch_model_path = os.path.join(save_dir, f"model_flow_ep{epoch}.pt")
        # torch.save(model.state_dict(), epoch_model_path)

    print("\n" + "=" * 30 + " Flow Fine-tuning Finished " + "=" * 30)
    print(f"Best validation Macro-F1 achieved: {best_val_f1:.4f}")


# --- New Test Function for Flows ---
def test_flow_model(
        model: nn.Module,  # Expected to be FlowFineTuningModel or similar
        test_dataset,  # IoTDataset returning flow data
        batch_size: int = 64,
        device: str = "cuda",
        model_path: Optional[str] = None  # Path to load specific model weights
):
    """Evaluate a flow-based model on the test set."""
    if model_path and os.path.exists(model_path):
        print(f"\nLoading model state for testing from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully for testing.")
        except Exception as e:
            print(f"Error loading model state from {model_path}: {e}")
            print("Proceeding with the model currently in memory.")
    elif model_path:
        print(f"Warning: Specified model_path '{model_path}' not found. Using model currently in memory.")

    # Use _build_loaders to get a test_loader, num_workers can be passed if needed
    # For simplicity, creating a simple loader here.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0)  # Adjust num_workers if needed

    model.to(device)
    model.eval()  # Set model to evaluation mode

    print(f"\nEvaluating FLOW model on Test Set (Device: '{device}')...")
    all_preds, all_labels = [], []
    nan_batches_test = 0

    with torch.no_grad():
        for batch in test_loader:
            try:
                numerical_flow_data = batch["numerical_features"].to(device)
                categorical_flow_data = batch.get("categorical_features")
                if categorical_flow_data is not None:
                    categorical_flow_data = categorical_flow_data.to(device)
                labels = batch["label"].to(device)
            except KeyError as e:
                print(f"❌ Test Batch missing expected key: {e}.")
                continue

            if not torch.isfinite(numerical_flow_data).all():
                nan_batches_test += 1
                continue  # Skip batch with NaNs

            logits = model(numerical_flow_data, categorical_flow_data)
            if not torch.isfinite(logits).all():
                nan_batches_test += 1
                continue

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print(f"Test completed. NaN batches skipped: {nan_batches_test}")
    if not all_labels:
        print("Error: No predictions made from the test set. Check data or NaN issues.")
        return

    print("\n" + "=" * 30 + " Flow Model Test Set Results " + "=" * 30)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")

    # Attempt to get target names if dataset object supports it
    target_names = None
    if hasattr(test_dataset, 'get_target_names'):  # Example method name
        try:
            target_names = test_dataset.get_target_names()
        except:
            pass  # Ignore if method fails or doesn't exist

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0, target_names=target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n" + "=" * 78)

