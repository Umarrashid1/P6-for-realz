import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path
import random
import numpy as np
import logging
import os
import sys
# REMOVE: from collections import Counter # No longer needed here
# REMOVE: from sklearn.utils.class_weight import compute_class_weight # No longer needed here

# --- Project-specific Imports ---
from pipeline.iot_flow_dataset import IoTFlowDataset
from models.flow_finetuning_model import FlowFineTuningModel

# --- Configuration & Basic Setup ---
CONFIG_FILE_PATH = Path("config.json")
LOG_FILE_OPTUNA_BASELINE_FLOW = "optuna_baseline_flow_log.txt"

script_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("tune_baseline_flow_script.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

silent_logger = logging.getLogger('silent_optuna_worker_baseline_flow')
silent_logger.addHandler(logging.NullHandler())
silent_logger.propagate = False


# --- Helper: Load Main Config ---
def load_main_config(config_path):
    if not config_path.is_file():
        script_logger.error(f"CRITICAL: Main configuration file not found at {config_path}")
        raise FileNotFoundError(f"Main configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    script_logger.info(f"Main configuration loaded from {config_path}")
    return config


# --- Global Variables (Loaded Once) ---
MAIN_CONFIG = load_main_config(CONFIG_FILE_PATH)
DEVICE = MAIN_CONFIG['general_settings']['device']
if DEVICE == "cuda" and not torch.cuda.is_available():
    script_logger.warning("CUDA specified but not available. Falling back to CPU.")
    DEVICE = "cpu"
script_logger.info(f"Using device: {DEVICE}")

PROCESSED_FLOW_DATA_PATH = Path(MAIN_CONFIG['dataset_paths']['processed_flow_pt'])

# --- Load Pre-calculated Class Weights (NEW) ---
WEIGHTS_DIR = Path(MAIN_CONFIG['dataset_paths']['raw_packet_dir']).parent / "weights"
FLOW_WEIGHTS_PATH = WEIGHTS_DIR / "flow_class_weights.pt"
PRECOMPUTED_FLOW_WEIGHTS = None
try:
    PRECOMPUTED_FLOW_WEIGHTS = torch.load(FLOW_WEIGHTS_PATH, map_location='cpu')  # Load to CPU first
    script_logger.info(f"Successfully loaded precomputed flow weights from {FLOW_WEIGHTS_PATH}")
except FileNotFoundError:
    script_logger.error(
        f"ERROR: Precomputed flow weights file not found at {FLOW_WEIGHTS_PATH}. Please run calculate_global_weights.py first. Exiting.")
    sys.exit(1)  # Exit if weights are essential and not found
except Exception as e:
    script_logger.error(f"ERROR: Could not load flow weights from {FLOW_WEIGHTS_PATH}: {e}. Exiting.", exc_info=True)
    sys.exit(1)


# --- Modified Helper: Balanced Loss to USE Precomputed Weights (NEW) ---
def get_balanced_loss_criterion(device, precomputed_weights_tensor, logger_instance=None) -> nn.CrossEntropyLoss:
    if logger_instance is None:
        logger_instance = silent_logger

    if precomputed_weights_tensor is not None:
        weights_t = precomputed_weights_tensor.to(device)  # Move to target device
        if torch.isfinite(weights_t).all():
            logger_instance.debug(f"Using precomputed class-balanced weights for flow: {weights_t.cpu().numpy()}")
            return nn.CrossEntropyLoss(weight=weights_t)
        else:
            logger_instance.warning("Precomputed weights for flow contain NaN/Inf. Using unweighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss()
    else:
        # This case should ideally not be reached if we exit on PRECOMPUTED_FLOW_WEIGHTS load failure
        logger_instance.warning("Precomputed weights for flow not available. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


try:
    FULL_FLOW_DATASET = IoTFlowDataset(PROCESSED_FLOW_DATA_PATH)
    script_logger.info(f"Full flow dataset loaded. Total flows: {len(FULL_FLOW_DATASET)}")
    flow_data_pt_sample = torch.load(PROCESSED_FLOW_DATA_PATH, map_location='cpu')
    NUM_FLOW_NUMERICAL_F = flow_data_pt_sample['numerical_features'].shape[1]
    flow_cat_tensor_sample = flow_data_pt_sample['categorical_features']
    NUM_FLOW_CATEGORICAL_F = flow_cat_tensor_sample.shape[1]
    FLOW_CAT_CARDINALITIES = []
    if NUM_FLOW_CATEGORICAL_F > 0:
        for i in range(NUM_FLOW_CATEGORICAL_F):
            if flow_cat_tensor_sample[:, i].numel() > 0:
                FLOW_CAT_CARDINALITIES.append(int(torch.max(flow_cat_tensor_sample[:, i])) + 1)
            else:
                FLOW_CAT_CARDINALITIES.append(1)
    del flow_data_pt_sample
except Exception as e:
    script_logger.error(f"Error loading full flow dataset or its characteristics: {e}", exc_info=True)
    raise

NUM_CLASSES_FLOW = MAIN_CONFIG['model_architecture']['num_classes_flow']


# --- Optuna Objective Function for Flow-Only Baseline ---
def objective_baseline_flow(trial: optuna.trial.Trial):
    objective_logger = logging.getLogger(f"optuna_trial_baseline_flow_{trial.number}")
    objective_logger.info(f"--- Starting Optuna Flow-Only Baseline Trial: {trial.number} ---")

    lr = trial.suggest_float("lr_baseline_flow", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size_baseline_flow", [32, 64, 128])
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    if d_model % num_heads != 0:
        objective_logger.warning(f"d_model ({d_model}) not divisible by num_heads ({num_heads}). Pruning trial.")
        raise optuna.TrialPruned(f"d_model ({d_model}) not divisible by num_heads ({num_heads}).")
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout_transformer_body = trial.suggest_float("dropout_transformer_body", 0.05, 0.3)
    classifier_dropout = trial.suggest_float("classifier_dropout_flow", 0.05, 0.5)

    epochs_baseline_flow = MAIN_CONFIG['training_params']['baseline_flow_only']['epochs']
    clip_grad_baseline_flow = MAIN_CONFIG['training_params']['baseline_flow_only'].get('clip_grad', 1.0)

    objective_logger.info(f"Trial {trial.number} Hyperparameters: LR={lr:.2e}, BatchSize={batch_size}, "
                          f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, "
                          f"dropout_body={dropout_transformer_body:.2f}, classifier_dropout={classifier_dropout:.2f}, Epochs={epochs_baseline_flow}")

    trial_seed = random.randint(1, 100000)
    random.seed(trial_seed);
    np.random.seed(trial_seed);
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(trial_seed)

    val_ratio = MAIN_CONFIG['dataset_params'].get('val_ratio_flow', 0.1)
    test_ratio_dummy = MAIN_CONFIG['dataset_params'].get('test_ratio_flow', 0.1)
    total_size = len(FULL_FLOW_DATASET)
    val_size = int(total_size * val_ratio)
    if total_size * (1 - val_ratio - test_ratio_dummy) <= 0:
        train_size = max(1, int(total_size * 0.7));
        val_size = max(1, int(total_size * 0.15))
        dummy_test_size = total_size - train_size - val_size
        if dummy_test_size < 0: dummy_test_size = 0
    else:
        train_size = total_size - val_size - int(total_size * test_ratio_dummy)
        dummy_test_size = total_size - train_size - val_size
    if train_size <= 0 or val_size <= 0:
        objective_logger.error(
            f"Invalid train/val split for Trial {trial.number}. Train: {train_size}, Val: {val_size}. Skipping.")
        return -float('inf')

    generator = torch.Generator().manual_seed(MAIN_CONFIG['general_settings']['random_seed'])
    # IMPORTANT: We do not need train_dataset for criterion anymore if weights are precomputed globally
    _, val_dataset, _ = random_split(FULL_FLOW_DATASET, [train_size, val_size, dummy_test_size], generator=generator)

    # Create a new train_dataset for the DataLoader, Optuna doesn't need access to its labels for criterion
    train_dataset_for_loader, _, _ = random_split(FULL_FLOW_DATASET, [train_size, val_size, dummy_test_size],
                                                  generator=generator)

    num_dataloader_workers = MAIN_CONFIG['general_settings'].get('num_workers_loader', 0)
    objective_logger.info(f"Trial {trial.number} DataLoaders using num_workers={num_dataloader_workers}")
    pin_memory_flag = (DEVICE == "cuda" and num_dataloader_workers > 0)

    train_loader = DataLoader(train_dataset_for_loader, batch_size=batch_size, shuffle=True,
                              num_workers=num_dataloader_workers, pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_dataloader_workers,
                            pin_memory=pin_memory_flag)

    flow_encoder_layer_scratch = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
        dropout=dropout_transformer_body, batch_first=True, activation=torch.nn.functional.relu
    )
    flow_transformer_body_scratch = nn.TransformerEncoder(
        encoder_layer=flow_encoder_layer_scratch, num_layers=num_layers
    )
    model = FlowFineTuningModel(
        num_flow_numerical_features=NUM_FLOW_NUMERICAL_F,
        flow_cat_cardinalities=FLOW_CAT_CARDINALITIES, d_model=d_model,
        pretrained_transformer_encoder_body=flow_transformer_body_scratch,
        num_classes=NUM_CLASSES_FLOW, classifier_dropout=classifier_dropout
    )
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Use the new function with precomputed weights
    criterion = get_balanced_loss_criterion(DEVICE, PRECOMPUTED_FLOW_WEIGHTS, logger_instance=objective_logger)

    best_val_macro_f1_for_trial = -1.0
    for epoch in range(1, epochs_baseline_flow + 1):
        model.train()
        total_train_loss = 0;
        nan_batches_train = 0
        for batch_idx, batch in enumerate(train_loader):
            try:
                num_flow = batch["numerical_features"].to(DEVICE)
                cat_flow = batch.get("categorical_features")
                if cat_flow is not None and cat_flow.numel() > 0:
                    cat_flow = cat_flow.to(DEVICE)
                else:
                    cat_flow = None
                labels = batch["label"].to(DEVICE)
            except KeyError as e:
                objective_logger.error(f"T{trial.number} E{epoch} B{batch_idx} missing key: {e}. Skip.", exc_info=True);
                continue

            if not torch.isfinite(num_flow).all():
                objective_logger.warning(f"T{trial.number} E{epoch} B{batch_idx} NaN/Inf num_flow. Skip.");
                nan_batches_train += 1;
                continue
            if cat_flow is not None and not torch.isfinite(cat_flow.float()).all():  # Check for NaN/Inf in categoricals
                objective_logger.warning(f"T{trial.number} E{epoch} B{batch_idx} NaN/Inf cat_flow. Skip.");
                nan_batches_train += 1;
                continue

            optimizer.zero_grad()
            logits = model(num_flow, cat_flow)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                objective_logger.warning(
                    f"T{trial.number} E{epoch} B{batch_idx} NaN/Inf loss. Logits: {float(logits.min()):.3e}/{float(logits.max()):.3e}. Skip.");
                nan_batches_train += 1;
                continue
            loss.backward()
            if clip_grad_baseline_flow > 0: nn.utils.clip_grad_norm_(model.parameters(), clip_grad_baseline_flow)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / max(1, len(train_loader) - nan_batches_train)

        model.eval()
        val_preds_list, val_labels_list = [], [];
        total_val_loss = 0;
        nan_batches_val = 0
        with torch.no_grad():
            for batch_idx_val, batch in enumerate(val_loader):
                try:
                    num_flow = batch["numerical_features"].to(DEVICE)
                    cat_flow = batch.get("categorical_features")
                    if cat_flow is not None and cat_flow.numel() > 0:
                        cat_flow = cat_flow.to(DEVICE)
                    else:
                        cat_flow = None
                    labels = batch["label"].to(DEVICE)
                except KeyError as e:
                    objective_logger.error(f"T{trial.number} E{epoch} ValB{batch_idx_val} missing key: {e}. Skip.",
                                           exc_info=True);
                    continue
                if not torch.isfinite(num_flow).all():
                    objective_logger.warning(f"T{trial.number} E{epoch} ValB{batch_idx_val} NaN/Inf num_flow. Skip.");
                    nan_batches_val += 1;
                    continue
                logits = model(num_flow, cat_flow)
                if not torch.isfinite(logits).all():  # Check for NaN/Inf logits
                    objective_logger.warning(f"T{trial.number} E{epoch} ValB{batch_idx_val} NaN/Inf logits. Skip.");
                    nan_batches_val += 1;
                    continue
                loss_val_b = criterion(logits, labels)
                if torch.isfinite(loss_val_b):
                    total_val_loss += loss_val_b.item()
                else:
                    nan_batches_val += 1
                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())
        avg_val_loss = total_val_loss / max(1, len(val_loader) - nan_batches_val)
        current_val_macro_f1 = \
        precision_recall_fscore_support(val_labels_list, val_preds_list, average="macro", zero_division=0)[
            2] if val_labels_list else 0.0

        log_message = (f"Trial {trial.number} (BaselineFlow) Epoch {epoch}/{epochs_baseline_flow} - "
                       f"TrainLoss: {avg_train_loss:.4f}, ValLoss: {avg_val_loss:.4f}, ValMacroF1: {current_val_macro_f1:.4f}")
        optuna.logging.get_logger("optuna").info(log_message)
        objective_logger.info(log_message)

        if current_val_macro_f1 > best_val_macro_f1_for_trial: best_val_macro_f1_for_trial = current_val_macro_f1
        trial.report(current_val_macro_f1, epoch)
        if trial.should_prune():
            objective_logger.info(f"Trial {trial.number} (BaselineFlow) pruned at epoch {epoch}.")
            if DEVICE == "cuda": torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    if DEVICE == "cuda": torch.cuda.empty_cache()
    objective_logger.info(
        f"Trial {trial.number} (BaselineFlow) Finished. Best Val Macro F1: {best_val_macro_f1_for_trial:.4f}")
    return best_val_macro_f1_for_trial


if __name__ == "__main__":
    study_name_baseline_flow = "baseline-flow-transformer-study"
    storage_name_baseline_flow = f"sqlite:///{study_name_baseline_flow}.db"

    optuna_stream_handler = logging.StreamHandler(sys.stdout)
    optuna_file_handler = logging.FileHandler(LOG_FILE_OPTUNA_BASELINE_FLOW, mode="a")
    optuna.logging.enable_propagation();
    optuna.logging.disable_default_handler()
    optuna_logger_instance = optuna.logging.get_logger("optuna")
    optuna_logger_instance.addHandler(optuna_stream_handler);
    optuna_logger_instance.addHandler(optuna_file_handler)
    optuna_logger_instance.setLevel(logging.INFO)

    script_logger.info(
        f"Starting Optuna study for Flow-Only Baseline: {study_name_baseline_flow}. Results: {storage_name_baseline_flow}")
    script_logger.info(f"Optuna's own logs for flow-only baseline will be in: {LOG_FILE_OPTUNA_BASELINE_FLOW}")
    script_logger.info(f"This script's general logs will be in: tune_baseline_flow_script.log")

    N_TRIALS_BASELINE_FLOW = MAIN_CONFIG.get("optuna_params", {}).get("n_trials_baseline_flow", 30)
    script_logger.info(f"Optuna will run for {N_TRIALS_BASELINE_FLOW} trials for flow-only baseline.")

    study_baseline_flow = optuna.create_study(
        study_name=study_name_baseline_flow, storage=storage_name_baseline_flow,
        direction="maximize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    )

    baseline_flow_cfg = MAIN_CONFIG['training_params']['baseline_flow_only']
    baseline_transformer_cfg = MAIN_CONFIG['model_architecture']['transformer_body']
    enqueued_params = {
        'lr_baseline_flow': baseline_flow_cfg['lr'], 'batch_size_baseline_flow': baseline_flow_cfg['batch_size'],
        'd_model': baseline_transformer_cfg['d_model'], 'num_heads': baseline_transformer_cfg['num_heads'],
        'num_layers': baseline_transformer_cfg['num_layers'],
        'dropout_transformer_body': baseline_transformer_cfg['dropout'],
        'classifier_dropout_flow': MAIN_CONFIG['model_architecture'].get('classifier_dropout_flow',
                                                                         baseline_transformer_cfg['dropout'])
    }
    try:
        study_baseline_flow.enqueue_trial(enqueued_params)
        script_logger.info(f"Enqueued default baseline flow parameters: {enqueued_params}")
    except Exception as e:
        script_logger.error(f"Could not enqueue trial with params {enqueued_params}. Error: {e}.")

    study_baseline_flow.optimize(objective_baseline_flow, n_trials=N_TRIALS_BASELINE_FLOW)

    script_logger.info("\n--- Optuna Flow-Only Baseline Study Complete ---")
    if study_baseline_flow.best_trial:
        script_logger.info(f"Best trial number: {study_baseline_flow.best_trial.number}")
        script_logger.info(f"Best validation macro F1-score (baseline flow): {study_baseline_flow.best_value:.4f}")
        script_logger.info("Best hyperparameters (baseline flow):")
        for key, value in study_baseline_flow.best_params.items(): script_logger.info(f"  {key}: {value}")
    else:
        script_logger.info("No successful trials completed in this study session.")
    script_logger.info("\nNext steps: Update config.json with best params, then run train_baseline_flow.py.")