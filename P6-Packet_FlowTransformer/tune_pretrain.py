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
import logging
import os
import sys  # For Optuna logging to stdout
from collections import Counter

# --- Project-specific Imports ---
# Adjust these paths if your script is not in the same directory as pretrain.py
from pipeline.iot_dataset import IoTSequenceDataset
from models.transformer import IoTTransformer
from utils.category_mapping import load_mappings
from pipeline.config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING

# --- Configuration & Basic Setup ---
CONFIG_FILE_PATH = Path("config.json")
LOG_FILE_OPTUNA_PRETRAIN = "optuna_pretrain_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("tune_pretrain_script.log"),  # Script's own log
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Helper: Load Main Config ---
def load_main_config(config_path):
    if not config_path.is_file():
        logger.error(f"CRITICAL: Main configuration file not found at {config_path}")
        raise FileNotFoundError(f"Main configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Main configuration loaded from {config_path}")
    return config


# --- Helper: Balanced Loss (Simplified from train.py) ---
def get_balanced_loss_pretrain(train_ds_subset, device):
    if len(train_ds_subset) == 0:
        return nn.CrossEntropyLoss()
    try:
        labels = [train_ds_subset[i]["label"].item() for i in range(len(train_ds_subset))]
    except Exception as e:
        logger.warning(f"Could not extract labels for balanced loss: {e}. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    if not labels or len(np.unique(labels)) <= 1:
        logger.warning("Not enough labels or classes for balanced loss. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    try:
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        if not torch.isfinite(weights_t).all():
            logger.warning("Non-finite class weight detected. Using unweighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss()
        logger.debug(f"Using class-balanced weights for pretrain: {weights_t.cpu().numpy()}")
        return nn.CrossEntropyLoss(weight=weights_t)
    except ValueError as e:
        logger.warning(f"Could not compute class weights for pretrain ({e}). Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


# --- Global Variables (Loaded Once) ---
MAIN_CONFIG = load_main_config(CONFIG_FILE_PATH)
DEVICE = MAIN_CONFIG['general_settings']['device']
if DEVICE == "cuda" and not torch.cuda.is_available():
    logger.warning("CUDA specified but not available. Falling back to CPU.")
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# Load packet dataset (once)
PROCESSED_PACKET_DATA_PATH = Path(MAIN_CONFIG['dataset_paths']['processed_packet_pt'])
MAX_SEQ_LEN_PACKET_CONFIG = MAIN_CONFIG['model_architecture']['max_seq_len_packet']
try:
    FULL_PACKET_DATASET = IoTSequenceDataset(PROCESSED_PACKET_DATA_PATH, max_seq_len=MAX_SEQ_LEN_PACKET_CONFIG)
    logger.info(f"Full packet dataset loaded. Total sequences: {len(FULL_PACKET_DATASET)}")
except Exception as e:
    logger.error(f"Error loading full packet dataset: {e}", exc_info=True)
    raise

# Load packet category mappings (once)
CAT_MAP_PACKET_PATH = MAIN_CONFIG.get('paths', {}).get('category_mappings_packet', 'category_mappings_packets.json')
try:
    CAT_MAP_PACKETS = load_mappings(path=CAT_MAP_PACKET_PATH, is_flow=False)
    CAT_SIZES_PACKETS = {col: len(CAT_MAP_PACKETS[col]) for col in categorical_columns_packets if
                         col in CAT_MAP_PACKETS}
    CAT_PAD_PACKETS = {c: CAT_MAP_PACKETS[c].get("unknown", 0) for c in categorical_columns_packets if
                       c in CAT_MAP_PACKETS}  # Ensure "unknown" exists or default

    for col in categorical_columns_packets:  # Ensure all necessary columns have entries
        if col not in CAT_SIZES_PACKETS:
            logger.warning(f"Packet category '{col}' not in mappings. Defaulting size to 1, padding_idx to 0.")
            CAT_SIZES_PACKETS[col] = 1
            CAT_PAD_PACKETS[col] = 0
except Exception as e:
    logger.error(f"Error loading packet category mappings: {e}", exc_info=True)
    raise

INPUT_DIM_NUMERICAL_PACKET_CONFIG = MAIN_CONFIG['model_architecture'].get('input_dim_packet_numerical',
                                                                          len(numerical_columns_packets))
NUM_CLASSES_PACKET_CONFIG = MAIN_CONFIG['model_architecture']['num_classes_packet']


# --- Optuna Objective Function for Pre-training ---
def objective_pretrain(trial: optuna.trial.Trial):
    logger.info(f"\n--- Starting Optuna Pre-training Trial: {trial.number} ---")

    # --- 1. Suggest Hyperparameters ---
    lr = trial.suggest_float("lr_pretrain", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size_pretrain", [32, 64, 128])

    # Transformer body parameters from config.json as base
    base_transformer_body_cfg = MAIN_CONFIG['model_architecture']['transformer_body']
    d_model = trial.suggest_categorical("d_model",
                                        [base_transformer_body_cfg['d_model'], 32, 128])  # Example: tune around default
    num_heads = trial.suggest_categorical("num_heads",
                                          [base_transformer_body_cfg['num_heads'], 2, 8])  # Must be divisor of d_model
    if d_model % num_heads != 0:  # Optuna constraint
        logger.warning(f"d_model ({d_model}) not divisible by num_heads ({num_heads}). Pruning trial.")
        raise optuna.TrialPruned(f"d_model ({d_model}) not divisible by num_heads ({num_heads}).")
    num_layers = trial.suggest_int("num_layers", 1,
                                   base_transformer_body_cfg['num_layers'] + 1)  # Example: 1 to default+1
    dropout_transformer_body = trial.suggest_float("dropout_transformer_body", 0.05, 0.3)

    epochs_pretrain = MAIN_CONFIG['training_params']['pre_training_packet']['epochs']  # Fixed for now for faster HPO

    logger.info(f"Trial {trial.number} Hyperparameters: LR={lr:.2e}, BatchSize={batch_size}, "
                f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, "
                f"dropout_body={dropout_transformer_body:.2f}, Epochs={epochs_pretrain}")

    # --- 2. Prepare Data (Split for this trial) ---
    # It's better to split FULL_PACKET_DATASET once outside and pass subsets or use fixed seed for splitting
    val_ratio = MAIN_CONFIG['dataset_params'].get('val_ratio_packet', 0.1)
    # The "test" split here is just to make random_split work, we only use train and val
    total_size = len(FULL_PACKET_DATASET)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - int(
        total_size * MAIN_CONFIG['dataset_params'].get('test_ratio_packet', 0.1))  # Keep consistent with pretrain.py

    # Adjust if sizes don't sum up or are too small
    current_test_ratio_dummy = MAIN_CONFIG['dataset_params'].get('test_ratio_packet', 0.1)
    if train_size + val_size > total_size or train_size <= 0 or val_size <= 0:
        logger.warning("Problem with dataset split sizes for pretrain, adjusting.")
        train_size = int(total_size * (1 - val_ratio - 0.01))  # Reserve 1% for dummy test
        val_size = int(total_size * val_ratio)
        dummy_test_size = total_size - train_size - val_size
    else:
        dummy_test_size = total_size - train_size - val_size

    if train_size <= 0 or val_size <= 0 or dummy_test_size < 0:
        logger.error("Cannot create valid train/val splits for pretrain. Aborting trial.")
        return -float('inf')

    generator = torch.Generator().manual_seed(MAIN_CONFIG['general_settings']['random_seed'])
    train_dataset, val_dataset, _ = random_split(FULL_PACKET_DATASET, [train_size, val_size, dummy_test_size],
                                                 generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- 3. Setup Model (IoTTransformer) ---
    model = IoTTransformer(
        input_dim=INPUT_DIM_NUMERICAL_PACKET_CONFIG,
        cat_sizes=CAT_SIZES_PACKETS,
        cat_padding_idx=CAT_PAD_PACKETS,
        embed_dim=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout_transformer_body,  # This is dropout for encoder layers AND final classifier in IoTTransformer
        num_classes=NUM_CLASSES_PACKET_CONFIG,
        max_seq_len=MAX_SEQ_LEN_PACKET_CONFIG
    )
    model.to(DEVICE)

    # --- 4. Setup Optimizer and Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = get_balanced_loss_pretrain(train_dataset, DEVICE)

    # --- 5. Training and Validation Loop (Simplified from train.py's train_model) ---
    best_val_macro_f1_for_trial = -1.0

    for epoch in range(1, epochs_pretrain + 1):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            packet_seq = batch["packet_seq"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            cat_feats = {c: batch[c].to(DEVICE) for c in categorical_columns_packets if c in batch}

            optimizer.zero_grad()
            logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                logger.warning(
                    f"Trial {trial.number}, Epoch {epoch}: NaN/Inf training loss (pretrain). Skipping batch.")
                continue

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), MAIN_CONFIG['training_params']['pre_training_packet'].get('clip_grad', 1.0))
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # Validation
        model.eval()
        val_preds_list, val_labels_list = [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                packet_seq = batch["packet_seq"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                cat_feats = {c: batch[c].to(DEVICE) for c in categorical_columns_packets if c in batch}

                logits = model(packet_seq, cat_feats, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                if not torch.isfinite(loss):
                    logger.warning(
                        f"Trial {trial.number}, Epoch {epoch}: NaN/Inf validation loss (pretrain). Skipping batch.")
                    continue
                total_val_loss += loss.item()

                val_preds_list.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        if not val_labels_list:
            logger.warning(
                f"Trial {trial.number}, Epoch {epoch}: No validation predictions (pretrain). Setting F1 to 0.")
            current_val_macro_f1 = 0.0
        else:
            current_val_macro_f1 = f1_score(val_labels_list, val_preds_list, average="macro", zero_division=0)

        logger.info(
            f"Trial {trial.number} (Pretrain) Epoch {epoch}/{epochs_pretrain} - TrainLoss: {avg_train_loss:.4f}, ValLoss: {avg_val_loss:.4f}, ValMacroF1: {current_val_macro_f1:.4f}")

        if current_val_macro_f1 > best_val_macro_f1_for_trial:
            best_val_macro_f1_for_trial = current_val_macro_f1

        # Optuna Pruning
        trial.report(current_val_macro_f1, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} (Pretrain) pruned at epoch {epoch}.")
            raise optuna.TrialPruned()

    logger.info(
        f"Trial {trial.number} (Pretrain) Finished. Best Validation Macro F1: {best_val_macro_f1_for_trial:.4f}")
    return best_val_macro_f1_for_trial


if __name__ == "__main__":
    study_name_pretrain = "pretrain-packet-transformer-study"
    storage_name_pretrain = f"sqlite:///{study_name_pretrain}.db"

    if os.path.exists(LOG_FILE_OPTUNA_PRETRAIN):
        pass  # Append

    optuna_stream_handler = optuna.logging.StreamHandler(sys.stdout)
    optuna_file_handler = optuna.logging.FileHandler(LOG_FILE_OPTUNA_PRETRAIN, mode="a")
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.addHandler(optuna_stream_handler)
    optuna_logger.addHandler(optuna_file_handler)
    optuna_logger.setLevel(logging.INFO)

    logger.info(f"Starting Optuna study for Pre-training: {study_name_pretrain}. Results: {storage_name_pretrain}")
    logger.info(f"Optuna's own logs for pre-training will be in: {LOG_FILE_OPTUNA_PRETRAIN}")

    # Number of trials for Optuna to run
    N_TRIALS_PRETRAIN = MAIN_CONFIG.get("optuna_params", {}).get("n_trials_pretrain",
                                                                 50)  # Add to config.json if desired
    logger.info(f"Optuna will run for {N_TRIALS_PRETRAIN} trials for pre-training.")

    study_pretrain = optuna.create_study(
        study_name=study_name_pretrain,
        storage=storage_name_pretrain,
        direction="maximize",
        load_if_exists=True,  # Resume if study already exists
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)  # Example pruner
    )

    study_pretrain.optimize(objective_pretrain, n_trials=N_TRIALS_PRETRAIN)

    logger.info("\n--- Optuna Pre-training Study Complete ---")
    logger.info(f"Best trial number: {study_pretrain.best_trial.number}")
    logger.info(f"Best validation macro F1-score (pretrain): {study_pretrain.best_value:.4f}")
    logger.info("Best hyperparameters (pretrain):")
    for key, value in study_pretrain.best_params.items():
        logger.info(f"  {key}: {value}")

    # After this, you would typically:
    # 1. Take the best_params found by study_pretrain.
    # 2. Update your main config.json with these best pre-training hyperparameters.
    # 3. Run pretrain.py one final time with these optimal settings to generate the definitive "model_best.pt" for the packet model.
    # 4. Then, proceed to hyperparameter tuning for the fine-tuning stage, ensuring it loads this newly optimized pre-trained model.
    logger.info("\nNext steps: Update config.json with these best pre-training params, "
                "run pretrain.py to save the best packet model, "
                "then proceed to tune the fine-tuning stage.")

