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
from models.packet_pretraining_model import PacketPretrainingModel  # To reconstruct pre-trained body arch
from models.flow_finetuning_model import FlowFineTuningModel
from utils.category_mapping import load_mappings  # For packet cat_sizes if needed
from pipeline.config import categorical_columns_packets, numerical_columns_packets  # For packet model structure

# --- Configuration & Basic Setup ---
CONFIG_FILE_PATH = Path("config.json")
LOG_FILE_OPTUNA_FINETUNE = "optuna_finetune_log.txt"  # New log file

script_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("tune_finetune_script.log"),  # New script log
        logging.StreamHandler(sys.stdout)
    ]
)

silent_logger = logging.getLogger('silent_optuna_worker_finetune')
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


# --- Helper: Balanced Loss ---
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
PRETRAINED_PACKET_MODEL_SAVE_DIR = Path(MAIN_CONFIG['paths']['packet_model_save_dir'])
PRETRAINED_PACKET_MODEL_WEIGHTS_PATH = PRETRAINED_PACKET_MODEL_SAVE_DIR / "model_best.pt"

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

CAT_MAP_PACKET_PATH = MAIN_CONFIG.get('paths', {}).get('category_mappings_packet', 'category_mappings_packets.json')
try:
    CAT_MAP_PACKETS = load_mappings(path=CAT_MAP_PACKET_PATH, is_flow=False)
    CAT_SIZES_PACKETS = {col: len(CAT_MAP_PACKETS[col]) for col in categorical_columns_packets if
                         col in CAT_MAP_PACKETS}
    CAT_PAD_PACKETS = {c: CAT_MAP_PACKETS[c].get("unknown", 0) for c in categorical_columns_packets if
                       c in CAT_MAP_PACKETS}
    for col in categorical_columns_packets:
        if col not in CAT_SIZES_PACKETS:
            CAT_SIZES_PACKETS[col] = 1
            CAT_PAD_PACKETS[col] = 0
except Exception as e:
    script_logger.error(f"Error loading packet category mappings for temp packet model: {e}", exc_info=True)
    raise

PACKET_MODEL_ARCH_CFG = MAIN_CONFIG['model_architecture']
PACKET_TRANSFORMER_BODY_CFG = PACKET_MODEL_ARCH_CFG['transformer_body']
INPUT_DIM_NUM_PACKET = PACKET_MODEL_ARCH_CFG.get('input_dim_packet_numerical', len(numerical_columns_packets))
MAX_SEQ_LEN_PACKET = PACKET_MODEL_ARCH_CFG['max_seq_len_packet']
NUM_CLASSES_PACKET_PRETRAIN = PACKET_MODEL_ARCH_CFG['num_classes_packet']
NUM_CLASSES_FLOW = PACKET_MODEL_ARCH_CFG['num_classes_flow']
D_MODEL_PRETRAINED_BODY = PACKET_TRANSFORMER_BODY_CFG['d_model']  # d_model of the actual pre-trained encoder


# --- Optuna Objective Function for Fine-tuning ---
def objective_finetune(trial: optuna.trial.Trial):
    objective_logger = logging.getLogger(f"optuna_trial_finetune_{trial.number}")
    objective_logger.info(f"--- Starting Optuna Fine-tuning Trial: {trial.number} ---")

    lr_finetune = trial.suggest_float("lr_finetune", 1e-6, 5e-4, log=True)
    batch_size_flow = trial.suggest_categorical("batch_size_flow", [32, 64, 128])
    classifier_dropout_flow = trial.suggest_float("classifier_dropout_flow", 0.05, 0.5)

    epochs_finetune = MAIN_CONFIG['training_params']['fine_tuning_flow']['epochs']
    clip_grad_finetune = MAIN_CONFIG['training_params']['fine_tuning_flow'].get('clip_grad', 1.0)

    objective_logger.info(f"Trial {trial.number} Hyperparameters: LR={lr_finetune:.2e}, BatchSize={batch_size_flow}, "
                          f"ClassifierDropout={classifier_dropout_flow:.2f}, Epochs={epochs_finetune}")

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
    train_dataset, val_dataset, _ = random_split(FULL_FLOW_DATASET, [train_size, val_size, dummy_test_size],
                                                 generator=generator)

    num_dataloader_workers = MAIN_CONFIG['general_settings'].get('num_workers_loader', 0)
    objective_logger.info(f"Trial {trial.number} DataLoaders using num_workers={num_dataloader_workers}")
    pin_memory_flag = (DEVICE == "cuda" and num_dataloader_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_flow, shuffle=True,
                              num_workers=num_dataloader_workers, pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_flow, shuffle=False, num_workers=num_dataloader_workers,
                            pin_memory=pin_memory_flag)

    # --- Model Setup (as in finetune.py) ---
    temp_packet_model_args = {
        'input_dim': INPUT_DIM_NUM_PACKET, 'cat_sizes': CAT_SIZES_PACKETS,
        'cat_padding_idx': CAT_PAD_PACKETS, 'embed_dim': PACKET_TRANSFORMER_BODY_CFG['d_model'],
        'num_heads': PACKET_TRANSFORMER_BODY_CFG['num_heads'], 'num_layers': PACKET_TRANSFORMER_BODY_CFG['num_layers'],
        'dropout': PACKET_TRANSFORMER_BODY_CFG['dropout'], 'num_classes': NUM_CLASSES_PACKET_PRETRAIN,
        'max_seq_len': MAX_SEQ_LEN_PACKET
    }
    original_packet_model_for_body = PacketPretrainingModel(**temp_packet_model_args)
    pretrained_transformer_body = original_packet_model_for_body.transformer

    model = FlowFineTuningModel(
        num_flow_numerical_features=NUM_FLOW_NUMERICAL_F,
        flow_cat_cardinalities=FLOW_CAT_CARDINALITIES,
        d_model=D_MODEL_PRETRAINED_BODY,  # Use d_model of the actual pre-trained encoder
        pretrained_transformer_encoder_body=pretrained_transformer_body,
        num_classes=NUM_CLASSES_FLOW,
        classifier_dropout=classifier_dropout_flow
    )
    model.to(DEVICE)

    if not PRETRAINED_PACKET_MODEL_WEIGHTS_PATH.is_file():
        objective_logger.error(
            f"Pretrained packet model weights not found at {PRETRAINED_PACKET_MODEL_WEIGHTS_PATH}. Trial {trial.number} cannot proceed.")
        return -float('inf')
    try:
        pretrained_state_dict = torch.load(PRETRAINED_PACKET_MODEL_WEIGHTS_PATH, map_location=DEVICE)
        PRETRAINED_BODY_PREFIX = "transformer."
        TARGET_BODY_PREFIX = "transformer_encoder_body."
        body_weights_to_load = {TARGET_BODY_PREFIX + k[len(PRETRAINED_BODY_PREFIX):]: v
                                for k, v in pretrained_state_dict.items() if k.startswith(PRETRAINED_BODY_PREFIX)}
        if not body_weights_to_load:
            objective_logger.error(
                f"No weights with prefix '{PRETRAINED_BODY_PREFIX}' in {PRETRAINED_PACKET_MODEL_WEIGHTS_PATH}. Trial {trial.number} cannot load.")
            return -float('inf')
        model.load_state_dict(body_weights_to_load, strict=False)
        objective_logger.info(f"Loaded pre-trained transformer body weights for trial {trial.number}.")
    except Exception as e:
        objective_logger.error(f"Error loading pre-trained weights for trial {trial.number}: {e}", exc_info=True)
        return -float('inf')

    for name, param in model.named_parameters():
        if name.startswith(TARGET_BODY_PREFIX):
            param.requires_grad = False
    # --- End Model Setup ---

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_finetune)
    criterion = _balanced_loss_flow(train_dataset, DEVICE, logger_instance=objective_logger)
    best_val_macro_f1_for_trial = -1.0

    for epoch in range(1, epochs_finetune + 1):
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
            if cat_flow is not None and not torch.isfinite(cat_flow.float()).all():
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
            if clip_grad_finetune > 0: nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                                clip_grad_finetune)
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
                if not torch.isfinite(logits).all():
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

        log_message = (f"Trial {trial.number} (Finetune) Epoch {epoch}/{epochs_finetune} - "
                       f"TrainLoss: {avg_train_loss:.4f}, ValLoss: {avg_val_loss:.4f}, ValMacroF1: {current_val_macro_f1:.4f}")
        optuna.logging.get_logger("optuna").info(log_message)
        objective_logger.info(log_message)

        if current_val_macro_f1 > best_val_macro_f1_for_trial: best_val_macro_f1_for_trial = current_val_macro_f1
        trial.report(current_val_macro_f1, epoch)
        if trial.should_prune():
            objective_logger.info(f"Trial {trial.number} (Finetune) pruned at epoch {epoch}.")
            if DEVICE == "cuda": torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    if DEVICE == "cuda": torch.cuda.empty_cache()
    objective_logger.info(
        f"Trial {trial.number} (Finetune) Finished. Best Val Macro F1: {best_val_macro_f1_for_trial:.4f}")
    return best_val_macro_f1_for_trial


if __name__ == "__main__":
    study_name_finetune = "finetune-flow-study"
    storage_name_finetune = f"sqlite:///{study_name_finetune}.db"

    optuna_stream_handler = logging.StreamHandler(sys.stdout)
    optuna_file_handler = logging.FileHandler(LOG_FILE_OPTUNA_FINETUNE, mode="a")
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna_logger_instance = optuna.logging.get_logger("optuna")
    optuna_logger_instance.addHandler(optuna_stream_handler)
    optuna_logger_instance.addHandler(optuna_file_handler)
    optuna_logger_instance.setLevel(logging.INFO)

    script_logger.info(
        f"Starting Optuna study for Fine-tuning: {study_name_finetune}. Results: {storage_name_finetune}")
    script_logger.info(f"Optuna's own logs for fine-tuning will be in: {LOG_FILE_OPTUNA_FINETUNE}")
    script_logger.info(f"This script's general logs will be in: tune_finetune_script.log")

    N_TRIALS_FINETUNE = MAIN_CONFIG.get("optuna_params", {}).get("n_trials_finetune", 30)
    script_logger.info(f"Optuna will run for {N_TRIALS_FINETUNE} trials for fine-tuning.")

    study_finetune = optuna.create_study(
        study_name=study_name_finetune, storage=storage_name_finetune,
        direction="maximize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    )

    # Enqueue default fine-tuning parameters from config.json
    finetune_cfg = MAIN_CONFIG['training_params']['fine_tuning_flow']
    default_classifier_dropout = MAIN_CONFIG['model_architecture'].get('classifier_dropout_flow', 0.1)
    enqueued_ft_params = {
        'lr_finetune': finetune_cfg['lr'],
        'batch_size_flow': finetune_cfg['batch_size'],
        'classifier_dropout_flow': default_classifier_dropout
    }
    try:
        study_finetune.enqueue_trial(enqueued_ft_params)
        script_logger.info(f"Enqueued default fine-tuning parameters: {enqueued_ft_params}")
    except Exception as e:
        script_logger.error(f"Could not enqueue trial with fine-tuning params {enqueued_ft_params}. Error: {e}.")

    study_finetune.optimize(objective_finetune, n_trials=N_TRIALS_FINETUNE)

    script_logger.info("\n--- Optuna Fine-tuning Study Complete ---")
    if study_finetune.best_trial:
        script_logger.info(f"Best trial number: {study_finetune.best_trial.number}")
        script_logger.info(f"Best validation macro F1-score (finetune): {study_finetune.best_value:.4f}")
        script_logger.info("Best hyperparameters (finetune):")
        for key, value in study_finetune.best_params.items(): script_logger.info(f"  {key}: {value}")
    else:
        script_logger.info("No successful trials completed in this fine-tuning study session.")
    script_logger.info("\nNext steps: Update config.json with these best fine-tuning params, "
                       "then run finetune.py to train and save the final best fine-tuned model.")