import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
import json
from pathlib import Path
import random
import numpy as np
import logging  # Standard logging
import os
import sys  # For sys.stdout
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# --- Project-specific Imports ---
from pipeline.iot_packet_dataset import IoTPacketDataset
from models.packet_pretraining_model import PacketPretrainingModel
from utils.category_mapping import load_mappings
from pipeline.config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING
from train.train import _balanced_loss

# --- Configuration & Basic Setup ---
CONFIG_FILE_PATH = Path("config.json")
LOG_FILE_OPTUNA_PRETRAIN = "optuna_pretrain_log.txt"

script_logger = logging.getLogger(__name__)
# BasicConfig for the script's own logger (script_logger)
# Note: Optuna's logging setup below will define how Optuna's specific logger behaves.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("tune_pretrain_script.log"),
        logging.StreamHandler(sys.stdout)  # Script logger also to stdout
    ]
)

# Create a silent logger to pass to imported functions if we want to suppress their logs
silent_logger = logging.getLogger('silent_optuna_worker')
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

PROCESSED_PACKET_DATA_PATH = Path(MAIN_CONFIG['dataset_paths']['processed_packet_pt'])
MAX_SEQ_LEN_PACKET_CONFIG = MAIN_CONFIG['model_architecture']['max_seq_len_packet']
try:
    FULL_PACKET_DATASET = IoTPacketDataset(PROCESSED_PACKET_DATA_PATH, max_seq_len=MAX_SEQ_LEN_PACKET_CONFIG)
    script_logger.info(f"Full packet dataset loaded. Total sequences: {len(FULL_PACKET_DATASET)}")
except Exception as e:
    script_logger.error(f"Error loading full packet dataset: {e}", exc_info=True)
    raise

CAT_MAP_PACKET_PATH = MAIN_CONFIG.get('paths', {}).get('category_mappings_packet', 'category_mappings_packets.json')
try:
    CAT_MAP_PACKETS = load_mappings(path=CAT_MAP_PACKET_PATH, is_flow=False)
    CAT_SIZES_PACKETS = {col: len(CAT_MAP_PACKETS[col]) for col in categorical_columns_packets if
                         col in CAT_MAP_PACKETS}
    CAT_PAD_PACKETS = {c: CAT_MAP_PACKETS[c].get("unknown", 0) for c in categorical_columns_packets if
                       c in CAT_MAP_PACKETS}

    for col in categorical_columns_packets:
        if col not in CAT_SIZES_PACKETS:
            script_logger.warning(f"Packet category '{col}' not in mappings. Defaulting size to 1, padding_idx to 0.")
            CAT_SIZES_PACKETS[col] = 1
            CAT_PAD_PACKETS[col] = 0
except Exception as e:
    script_logger.error(f"Error loading packet category mappings: {e}", exc_info=True)
    raise

INPUT_DIM_NUMERICAL_PACKET_CONFIG = MAIN_CONFIG['model_architecture'].get('input_dim_packet_numerical',
                                                                          len(numerical_columns_packets))
NUM_CLASSES_PACKET_CONFIG = MAIN_CONFIG['model_architecture']['num_classes_packet']


# --- Optuna Objective Function for Pre-training ---
def objective_pretrain(trial: optuna.trial.Trial):
    # This logger is for messages within the objective function, separate from Optuna's own logger
    objective_logger = logging.getLogger(f"optuna_trial_{trial.number}")
    objective_logger.info(f"--- Starting Optuna Pre-training Trial: {trial.number} ---")

    lr = trial.suggest_float("lr_pretrain", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size_pretrain", [32, 64, 128])

    base_transformer_body_cfg = MAIN_CONFIG['model_architecture']['transformer_body']
    d_model = trial.suggest_categorical("d_model",
                                        [base_transformer_body_cfg['d_model'], 32, 128])
    num_heads = trial.suggest_categorical("num_heads",
                                          [base_transformer_body_cfg['num_heads'], 2, 8])
    if d_model % num_heads != 0:
        objective_logger.warning(f"d_model ({d_model}) not divisible by num_heads ({num_heads}). Pruning trial.")
        raise optuna.TrialPruned(f"d_model ({d_model}) not divisible by num_heads ({num_heads}).")
    num_layers = trial.suggest_int("num_layers", 1,
                                   base_transformer_body_cfg['num_layers'] + 1)
    dropout_transformer_body = trial.suggest_float("dropout_transformer_body", 0.05, 0.3)

    epochs_pretrain = MAIN_CONFIG['training_params']['pre_training_packet']['epochs']

    objective_logger.info(f"Trial {trial.number} Hyperparameters: LR={lr:.2e}, BatchSize={batch_size}, "
                          f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, "
                          f"dropout_body={dropout_transformer_body:.2f}, Epochs={epochs_pretrain}")

    val_ratio = MAIN_CONFIG['dataset_params'].get('val_ratio_packet', 0.1)
    total_size = len(FULL_PACKET_DATASET)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - int(
        total_size * MAIN_CONFIG['dataset_params'].get('test_ratio_packet', 0.1))

    current_test_ratio_dummy = MAIN_CONFIG['dataset_params'].get('test_ratio_packet', 0.1)
    if train_size + val_size > total_size or train_size <= 0 or val_size <= 0:
        objective_logger.warning("Problem with dataset split sizes for pretrain, adjusting.")
        train_size = int(total_size * (1 - val_ratio - 0.01))
        val_size = int(total_size * val_ratio)
        dummy_test_size = total_size - train_size - val_size
    else:
        dummy_test_size = total_size - train_size - val_size

    if train_size <= 0 or val_size <= 0 or dummy_test_size < 0:
        objective_logger.error("Cannot create valid train/val splits for pretrain. Aborting trial.")
        return -float('inf')

    generator = torch.Generator().manual_seed(MAIN_CONFIG['general_settings']['random_seed'])
    train_dataset, val_dataset, _ = random_split(FULL_PACKET_DATASET, [train_size, val_size, dummy_test_size],
                                                 generator=generator)
    num_dataloader_workers = MAIN_CONFIG['general_settings'].get('num_workers_loader', 0) 
    pin_memory_flag = (DEVICE == "cuda" and num_dataloader_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers, # This will now be 12
        pin_memory=pin_memory_flag
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers, # This will now be 12
        pin_memory=pin_memory_flag
    )


    model = PacketPretrainingModel(
        input_dim=INPUT_DIM_NUMERICAL_PACKET_CONFIG,
        cat_sizes=CAT_SIZES_PACKETS,
        cat_padding_idx=CAT_PAD_PACKETS,
        embed_dim=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout_transformer_body,
        num_classes=NUM_CLASSES_PACKET_CONFIG,
        max_seq_len=MAX_SEQ_LEN_PACKET_CONFIG
    )
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = _balanced_loss(train_dataset, DEVICE, logger=silent_logger)

    best_val_macro_f1_for_trial = -1.0

    for epoch in range(1, epochs_pretrain + 1):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):  # Added batch_idx for more detailed logging if needed
            packet_seq = batch["packet_seq"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            cat_feats = {c: batch[c].to(DEVICE) for c in categorical_columns_packets if c in batch}

            optimizer.zero_grad()
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                objective_logger.warning(
                    f"Trial {trial.number}, Epoch {epoch}, Batch {batch_idx}: NaN/Inf training loss. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        val_preds_list, val_labels_list = [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx_val, batch in enumerate(val_loader):  # Added batch_idx_val
                packet_seq = batch["packet_seq"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                cat_feats = {c: batch[c].to(DEVICE) for c in categorical_columns_packets if c in batch}

                logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                if not torch.isfinite(loss):
                    objective_logger.warning(
                        f"Trial {trial.number}, Epoch {epoch}, Val Batch {batch_idx_val}: NaN/Inf validation loss. Skipping batch.")
                    continue
                total_val_loss += loss.item()

                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        if not val_labels_list:
            objective_logger.warning(
                f"Trial {trial.number}, Epoch {epoch}: No validation predictions. Setting F1 to 0.")
            current_val_macro_f1 = 0.0
        else:
            current_val_macro_f1 = f1_score(val_labels_list, val_preds_list, average="macro", zero_division=0)

        # Log to Optuna's logger (which goes to console and optuna_pretrain_log.txt)
        # and also to the objective_logger (which goes to tune_pretrain_script.log and console)
        log_message = (f"Trial {trial.number} (Pretrain) Epoch {epoch}/{epochs_pretrain} - "
                       f"TrainLoss: {avg_train_loss:.4f}, ValLoss: {avg_val_loss:.4f}, "
                       f"ValMacroF1: {current_val_macro_f1:.4f}")
        optuna.logging.get_logger("optuna").info(log_message)  # Optuna's logger
        objective_logger.info(log_message)  # Trial-specific logger (part of script_logger lineage)

        if current_val_macro_f1 > best_val_macro_f1_for_trial:
            best_val_macro_f1_for_trial = current_val_macro_f1

        trial.report(current_val_macro_f1, epoch)
        if trial.should_prune():
            objective_logger.info(f"Trial {trial.number} (Pretrain) pruned at epoch {epoch}.")
            raise optuna.TrialPruned()

    objective_logger.info(
        f"Trial {trial.number} (Pretrain) Finished. Best Validation Macro F1: {best_val_macro_f1_for_trial:.4f}")
    return best_val_macro_f1_for_trial


if __name__ == "__main__":
    study_name_pretrain = "pretrain-packet-transformer-study"
    storage_name_pretrain = f"sqlite:///{study_name_pretrain}.db"

    if os.path.exists(LOG_FILE_OPTUNA_PRETRAIN):
        pass

    # Configure Optuna's own logging
    # Use standard logging handlers, not optuna.logging.StreamHandler
    optuna_stream_handler = logging.StreamHandler(sys.stdout)  # CORRECTED
    optuna_file_handler = logging.FileHandler(LOG_FILE_OPTUNA_PRETRAIN, mode="a")  # CORRECTED

    # You can set a specific format for Optuna's handlers if you want it different
    # from the basicConfig, or let them use the default format.
    # Example:
    # optuna_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # optuna_stream_handler.setFormatter(optuna_formatter)
    # optuna_file_handler.setFormatter(optuna_formatter)

    optuna.logging.enable_propagation()  # Propagate messages to the root logger initially
    optuna.logging.disable_default_handler()  # Disable Optuna's own default stderr handler

    optuna_logger_instance = optuna.logging.get_logger("optuna")  # Get Optuna's root logger
    optuna_logger_instance.addHandler(optuna_stream_handler)
    optuna_logger_instance.addHandler(optuna_file_handler)
    optuna_logger_instance.setLevel(logging.INFO)  # Set level for Optuna's logger

    script_logger.info(
        f"Starting Optuna study for Pre-training: {study_name_pretrain}. Results: {storage_name_pretrain}")
    script_logger.info(f"Optuna's own logs for pre-training will be in: {LOG_FILE_OPTUNA_PRETRAIN}")
    script_logger.info(f"This script's general logs will be in: tune_pretrain_script.log")

    N_TRIALS_PRETRAIN = MAIN_CONFIG.get("optuna_params", {}).get("n_trials_pretrain", 50)
    script_logger.info(f"Optuna will run for {N_TRIALS_PRETRAIN} trials for pre-training.")

    study_pretrain = optuna.create_study(
        study_name=study_name_pretrain,
        storage=storage_name_pretrain,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    )

    study_pretrain.optimize(objective_pretrain, n_trials=N_TRIALS_PRETRAIN)

    script_logger.info("\n--- Optuna Pre-training Study Complete ---")
    script_logger.info(f"Best trial number: {study_pretrain.best_trial.number}")
    script_logger.info(f"Best validation macro F1-score (pretrain): {study_pretrain.best_value:.4f}")
    script_logger.info("Best hyperparameters (pretrain):")
    for key, value in study_pretrain.best_params.items():
        script_logger.info(f"  {key}: {value}")

    script_logger.info("\nNext steps: Update config.json with these best pre-training params, "
                       "run pretrain.py to save the best packet model, "
                       "then proceed to tune the fine-tuning stage.")