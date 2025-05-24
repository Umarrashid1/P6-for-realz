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
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# --- Project-specific Imports ---
from pipeline.iot_flow_dataset import IoTFlowDataset
from models.flow_finetuning_model import FlowFineTuningModel

# We don't need PacketPretrainingModel or packet-specific configs here

# --- Configuration & Basic Setup ---
CONFIG_FILE_PATH = Path("config.json")
LOG_FILE_OPTUNA_BASELINE_FLOW = "optuna_baseline_flow_log.txt"  # New log file

script_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("tune_baseline_flow_script.log"),  # New script log
        logging.StreamHandler(sys.stdout)
    ]
)

silent_logger = logging.getLogger('silent_optuna_worker_baseline_flow')  # New silent logger
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


# --- Helper: Balanced Loss (can be adapted from train_flows.py or train.py) ---
def _balanced_loss_flow(train_ds, device, logger_instance=None) -> nn.CrossEntropyLoss:
    if logger_instance is None:
        logger_instance = silent_logger
    try:
        labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    except (KeyError, AttributeError, IndexError) as e:
        logger_instance.warning(f"Could not extract labels for balanced loss ({e}). Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    if not labels:
        logger_instance.warning("No labels in training dataset for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    classes = np.unique(labels)
    if len(classes) <= 1:
        logger_instance.warning("<= 1 class in training data. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    try:
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        if not torch.isfinite(weights_t).all():
            logger_instance.warning("Non-finite class weights computed. Using unweighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss()
        logger_instance.debug(f"Using class-balanced weights for flow: {weights_t.cpu().numpy()}")
        return nn.CrossEntropyLoss(weight=weights_t)
    except ValueError as e:
        logger_instance.warning(f"Could not compute class weights ({e}). Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


# --- Global Variables (Loaded Once) ---
MAIN_CONFIG = load_main_config(CONFIG_FILE_PATH)
DEVICE = MAIN_CONFIG['general_settings']['device']
if DEVICE == "cuda" and not torch.cuda.is_available():
    script_logger.warning("CUDA specified but not available. Falling back to CPU.")
    DEVICE = "cpu"
script_logger.info(f"Using device: {DEVICE}")

PROCESSED_FLOW_DATA_PATH = Path(MAIN_CONFIG['dataset_paths']['processed_flow_pt'])

try:
    FULL_FLOW_DATASET = IoTFlowDataset(PROCESSED_FLOW_DATA_PATH)
    script_logger.info(f"Full flow dataset loaded. Total flows: {len(FULL_FLOW_DATASET)}")
    # Get flow dataset characteristics
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

    # Hyperparameters to tune
    lr = trial.suggest_float("lr_baseline_flow", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size_baseline_flow", [32, 64, 128])

    # Transformer body parameters (will be used to build a new encoder from scratch)
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])  # Inspired by FlowTransformer & your pretrain search
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    if d_model % num_heads != 0:
        objective_logger.warning(f"d_model ({d_model}) not divisible by num_heads ({num_heads}). Pruning trial.")
        raise optuna.TrialPruned(f"d_model ({d_model}) not divisible by num_heads ({num_heads}).")
    num_layers = trial.suggest_int("num_layers", 1, 4)  # Exploring slightly more depth than pretrain default
    dropout_transformer_body = trial.suggest_float("dropout_transformer_body", 0.05, 0.3)

    classifier_dropout = trial.suggest_float("classifier_dropout_flow", 0.05, 0.5)

    epochs_baseline_flow = MAIN_CONFIG['training_params']['baseline_flow_only']['epochs']
    clip_grad_baseline_flow = MAIN_CONFIG['training_params']['baseline_flow_only'].get('clip_grad', 1.0)

    objective_logger.info(f"Trial {trial.number} Hyperparameters: LR={lr:.2e}, BatchSize={batch_size}, "
                          f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, "
                          f"dropout_body={dropout_transformer_body:.2f}, classifier_dropout={classifier_dropout:.2f}, Epochs={epochs_baseline_flow}")

    # Seed for this trial
    trial_seed = random.randint(1, 100000)
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)

    # Split dataset
    val_ratio = MAIN_CONFIG['dataset_params'].get('val_ratio_flow', 0.1)
    test_ratio_dummy = MAIN_CONFIG['dataset_params'].get('test_ratio_flow', 0.1)
    total_size = len(FULL_FLOW_DATASET)
    val_size = int(total_size * val_ratio)
    if total_size * (1 - val_ratio - test_ratio_dummy) <= 0:
        train_size = max(1, int(total_size * 0.7))
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
    train_dataset, val_dataset, _ = random_split(FULL_FLOW_DATASET, [train_size, val_size, dummy_test_size],
                                                 generator=generator)

    num_dataloader_workers = MAIN_CONFIG['general_settings'].get('num_workers_loader', 0)
    # num_dataloader_workers = min(num_dataloader_workers, 4) # Optional cap for Optuna
    objective_logger.info(f"Trial {trial.number} DataLoaders using num_workers={num_dataloader_workers}")
    pin_memory_flag = (DEVICE == "cuda" and num_dataloader_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers,
                              pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_dataloader_workers,
                            pin_memory=pin_memory_flag)

    # Create Transformer encoder body from scratch for this trial
    flow_encoder_layer_scratch = nn.TransformerEncoderLayer(
        d_model=d_model,  # Tuned
        nhead=num_heads,  # Tuned
        dim_feedforward=d_model * 4,  # Standard FFN expansion
        dropout=dropout_transformer_body,  # Tuned
        batch_first=True,
        activation=torch.nn.functional.relu
    )
    flow_transformer_body_scratch = nn.TransformerEncoder(
        encoder_layer=flow_encoder_layer_scratch,
        num_layers=num_layers  # Tuned
    )

    # Instantiate FlowFineTuningModel with the scratch-built body
    model = FlowFineTuningModel(
        num_flow_numerical_features=NUM_FLOW_NUMERICAL_F,
        flow_cat_cardinalities=FLOW_CAT_CARDINALITIES,
        d_model=d_model,  # This d_model is for the flow feature projections AND the encoder
        pretrained_transformer_encoder_body=flow_transformer_body_scratch,  # Crucially, this is new
        num_classes=NUM_CLASSES_FLOW,
        classifier_dropout=classifier_dropout  # Tuned
    )
    model.to(DEVICE)
    # All parameters should be trainable as the body is new

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = _balanced_loss_flow(train_dataset, DEVICE, logger_instance=objective_logger)

    best_val_macro_f1_for_trial = -1.0

    for epoch in range(1, epochs_baseline_flow + 1):
        model.train()
        total_train_loss = 0
        nan_batches_train = 0
        for batch_idx, batch in enumerate(train_loader):
            try:
                numerical_flow_data = batch["numerical_features"].to(DEVICE)
                categorical_flow_data = batch.get("categorical_features")
                if categorical_flow_data is not None and categorical_flow_data.numel() > 0:
                    categorical_flow_data = categorical_flow_data.to(DEVICE)
                else:
                    categorical_flow_data = None
                labels = batch["label"].to(DEVICE)
            except KeyError as e:
                objective_logger.error(f"T{trial.number} E{epoch} B{batch_idx} missing key: {e}. Skip.", exc_info=True)
                continue

            if not torch.isfinite(numerical_flow_data).all():
                objective_logger.warning(f"T{trial.number} E{epoch} B{batch_idx} NaN/Inf num_flow. Skip.")
                nan_batches_train += 1;
                continue
            if categorical_flow_data is not None and not torch.isfinite(categorical_flow_data.float()).all():
                objective_logger.warning(f"T{trial.number} E{epoch} B{batch_idx} NaN/Inf cat_flow. Skip.")
                nan_batches_train += 1;
                continue

            optimizer.zero_grad()
            logits = model(numerical_flow_data, categorical_flow_data)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                objective_logger.warning(
                    f"T{trial.number} E{epoch} B{batch_idx} NaN/Inf loss. Logits min/max: {float(logits.min()):.3e}/{float(logits.max()):.3e}. Skip.")
                nan_batches_train += 1;
                continue

            loss.backward()
            if clip_grad_baseline_flow > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_baseline_flow)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader) - nan_batches_train)

        model.eval()
        val_preds_list, val_labels_list = [], []
        total_val_loss = 0
        nan_batches_val = 0
        with torch.no_grad():
            for batch_idx_val, batch in enumerate(val_loader):
                try:
                    numerical_flow_data = batch["numerical_features"].to(DEVICE)
                    categorical_flow_data = batch.get("categorical_features")
                    if categorical_flow_data is not None and categorical_flow_data.numel() > 0:
                        categorical_flow_data = categorical_flow_data.to(DEVICE)
                    else:
                        categorical_flow_data = None
                    labels = batch["label"].to(DEVICE)
                except KeyError as e:
                    objective_logger.error(f"T{trial.number} E{epoch} ValB{batch_idx_val} missing key: {e}. Skip.",
                                           exc_info=True)
                    continue

                if not torch.isfinite(numerical_flow_data).all():
                    objective_logger.warning(f"T{trial.number} E{epoch} ValB{batch_idx_val} NaN/Inf num_flow. Skip.")
                    nan_batches_val += 1;
                    continue

                logits = model(numerical_flow_data, categorical_flow_data)
                if not torch.isfinite(logits).all():
                    objective_logger.warning(f"T{trial.number} E{epoch} ValB{batch_idx_val} NaN/Inf logits. Skip.")
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

        if not val_labels_list:
            objective_logger.warning(f"T{trial.number} E{epoch}: No val preds. F1=0.")
            current_val_macro_f1 = 0.0
        else:
            current_val_macro_f1 = \
            precision_recall_fscore_support(val_labels_list, val_preds_list, average="macro", zero_division=0)[2]

        log_message = (f"Trial {trial.number} (BaselineFlow) Epoch {epoch}/{epochs_baseline_flow} - "
                       f"TrainLoss: {avg_train_loss:.4f}, ValLoss: {avg_val_loss:.4f}, "
                       f"ValMacroF1: {current_val_macro_f1:.4f}")
        optuna.logging.get_logger("optuna").info(log_message)
        objective_logger.info(log_message)

        if current_val_macro_f1 > best_val_macro_f1_for_trial:
            best_val_macro_f1_for_trial = current_val_macro_f1

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
    study_name_baseline_flow = "baseline-flow-transformer-study"  # New study name
    storage_name_baseline_flow = f"sqlite:///{study_name_baseline_flow}.db"  # New DB file

    optuna_stream_handler = logging.StreamHandler(sys.stdout)
    optuna_file_handler = logging.FileHandler(LOG_FILE_OPTUNA_BASELINE_FLOW, mode="a")
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna_logger_instance = optuna.logging.get_logger("optuna")
    optuna_logger_instance.addHandler(optuna_stream_handler)
    optuna_logger_instance.addHandler(optuna_file_handler)
    optuna_logger_instance.setLevel(logging.INFO)

    script_logger.info(
        f"Starting Optuna study for Flow-Only Baseline: {study_name_baseline_flow}. Results: {storage_name_baseline_flow}")
    script_logger.info(f"Optuna's own logs for flow-only baseline will be in: {LOG_FILE_OPTUNA_BASELINE_FLOW}")
    script_logger.info(f"This script's general logs will be in: tune_baseline_flow_script.log")

    N_TRIALS_BASELINE_FLOW = MAIN_CONFIG.get("optuna_params", {}).get("n_trials_baseline_flow", 30)  # New config entry
    script_logger.info(f"Optuna will run for {N_TRIALS_BASELINE_FLOW} trials for flow-only baseline.")

    study_baseline_flow = optuna.create_study(
        study_name=study_name_baseline_flow,
        storage=storage_name_baseline_flow,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    )

    # Enqueue the default baseline_flow_only parameters from config.json
    baseline_flow_cfg = MAIN_CONFIG['training_params']['baseline_flow_only']
    baseline_transformer_cfg = MAIN_CONFIG['model_architecture']['transformer_body']  # These are for packet pretrain
    # but used as defaults for flow baseline arch

    # For the baseline flow model, the "d_model", "num_heads", "num_layers", "dropout" for its *own*
    # transformer body are typically defined by D_MODEL_COMPARABLE, etc. in train_baseline_flow.py,
    # which come from model_architecture.transformer_body.
    # So we use those as the defaults to enqueue.
    # The classifier_dropout for FlowFineTuningModel comes from model_architecture.classifier_dropout_flow

    enqueued_params = {
        'lr_baseline_flow': baseline_flow_cfg['lr'],  # 5e-5
        'batch_size_baseline_flow': baseline_flow_cfg['batch_size'],  # 64
        'd_model': baseline_transformer_cfg['d_model'],  # 64
        'num_heads': baseline_transformer_cfg['num_heads'],  # 4
        'num_layers': baseline_transformer_cfg['num_layers'],  # 2
        'dropout_transformer_body': baseline_transformer_cfg['dropout'],  # 0.1
        'classifier_dropout_flow': MAIN_CONFIG['model_architecture'].get('classifier_dropout_flow',
                                                                         baseline_transformer_cfg['dropout'])
        # default from config
    }
    try:
        study_baseline_flow.enqueue_trial(enqueued_params)
        script_logger.info(f"Enqueued default baseline flow parameters for an early trial: {enqueued_params}")
    except Exception as e:
        script_logger.error(
            f"Could not enqueue trial with params {enqueued_params}. Error: {e}. Maybe this combination was already run and failed/completed.")

    study_baseline_flow.optimize(objective_baseline_flow, n_trials=N_TRIALS_BASELINE_FLOW)

    script_logger.info("\n--- Optuna Flow-Only Baseline Study Complete ---")
    if study_baseline_flow.best_trial:  # Check if any trial completed successfully
        script_logger.info(f"Best trial number: {study_baseline_flow.best_trial.number}")
        script_logger.info(f"Best validation macro F1-score (baseline flow): {study_baseline_flow.best_value:.4f}")
        script_logger.info("Best hyperparameters (baseline flow):")
        for key, value in study_baseline_flow.best_params.items():
            script_logger.info(f"  {key}: {value}")
    else:
        script_logger.info("No successful trials completed in this study session.")

    script_logger.info(
        "\nNext steps: Update config.json with these best baseline flow params if they outperform defaults, "
        "then run train_baseline_flow.py to train and save the best version of this baseline model.")