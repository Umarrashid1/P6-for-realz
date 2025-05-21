import os
import torch
from torch.utils.data import random_split
import json
from pathlib import Path
import random
import numpy as np
import logging

# Project-specific imports
from pipeline.iot_dataset import IoTSequenceDataset
from models.transformer import IoTTransformer
from train.train import train_model, test_model # Assuming these will be updated
from utils.category_mapping import load_mappings
from pipeline.config import categorical_columns_packets, numerical_columns_packets, LABEL_MAPPING

# --- 0. Load Central Configuration, Set Up Logger, and Set Seeds ---
CONFIG_FILE_PATH = Path("config.json") # Adjust path if your config is elsewhere

if not CONFIG_FILE_PATH.is_file():
    # Use print here as logger might not be configured if config is missing
    print(f"❌ CRITICAL: Configuration file not found at {CONFIG_FILE_PATH}")
    exit(1)

with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

# --- Logging Setup ---
log_paths_config = config.get('paths', {}) # Use .get for safety
LOG_FILE_NAME = log_paths_config.get('packet_pretraining_log_file', 'training_packet_pretraining.log')
LOG_DIR = Path(log_paths_config.get('log_dir', 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"✅ Central configuration loaded from {CONFIG_FILE_PATH}")

# Apply Random Seed
SEED = config['general_settings']['random_seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
logger.info(f"🌱 Random seed set to: {SEED}")


# --- Helper Function for Data Splitting (with logging) ---
def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1, logger_instance=None):
    if logger_instance is None:
        logger_instance = logger

    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    if train_size < 0:
        logger_instance.warning(f"Dataset too small for specified ratios (Total: {total_size}, Val: {val_size}, Test: {test_size}). Adjusting splits.")
        if total_size * (1 - val_ratio - test_ratio) <=0:
             train_size = max(1, int(total_size * 0.7))
             val_size = max(1, int(total_size * 0.15))
             test_size = total_size - train_size - val_size
             if test_size < 0 : test_size = 0
        else:
            train_size = total_size - val_size - test_size
    logger_instance.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    return random_split(dataset, [train_size, val_size, test_size])

# --- Configuration & Parameters from Loaded Config ---
logger.info("--- Initializing Packet Pre-training (Using Central Config) ---")

# General Settings
DEVICE_str = config['general_settings']['device']
if DEVICE_str == "cuda" and not torch.cuda.is_available():
    logger.warning("⚠️ CUDA specified in config but not available. Falling back to CPU.")
    DEVICE = "cpu"
else:
    DEVICE = DEVICE_str
NUM_WORKERS_LOADER = config['general_settings']['num_workers_loader']
logger.info(f"Device: {DEVICE}, Num Workers for DataLoader: {NUM_WORKERS_LOADER}")

# Dataset Paths
dataset_cfg = config['dataset_paths']
PROCESSED_PACKET_DATA_PATH = Path(dataset_cfg['processed_packet_pt'])
logger.info(f"Processed Packet Dataset Path: {PROCESSED_PACKET_DATA_PATH}")

# Model Architecture
model_arch_cfg = config['model_architecture']
transformer_body_cfg = model_arch_cfg['transformer_body']
MAX_SEQ_LEN_PACKET = model_arch_cfg['max_seq_len_packet']
INPUT_DIM_NUMERICAL_PACKET = len(numerical_columns_packets) # Or from config: model_arch_cfg.get('input_dim_packet_numerical', len(numerical_columns_packets))
NUM_CLASSES_PACKET = model_arch_cfg['num_classes_packet']

logger.info(f"Key Model Arch Params: embed_dim/d_model={transformer_body_cfg['d_model']}, heads={transformer_body_cfg['num_heads']}, layers={transformer_body_cfg['num_layers']}, dropout={transformer_body_cfg['dropout']}")
logger.info(f"Packet Specific Arch Params: max_seq_len={MAX_SEQ_LEN_PACKET}, num_numerical_features={INPUT_DIM_NUMERICAL_PACKET}, num_classes={NUM_CLASSES_PACKET}")


# Training Hyperparameters
pretrain_params_cfg = config['training_params']['pre_training_packet']
EPOCHS_PRETRAIN = pretrain_params_cfg['epochs']
LR_PRETRAIN = pretrain_params_cfg['lr']
BATCH_SIZE_PRETRAIN = pretrain_params_cfg['batch_size']
CLIP_GRAD_PRETRAIN = pretrain_params_cfg.get('clip_grad', 1.0)
USE_WEIGHTED_SAMPLER_PRETRAIN = pretrain_params_cfg.get('use_weighted_sampler', False)
logger.info(f"Pre-training Params: Epochs={EPOCHS_PRETRAIN}, LR={LR_PRETRAIN}, BatchSize={BATCH_SIZE_PRETRAIN}")


# Output Paths
paths_cfg = config['paths']
PACKET_MODEL_SAVE_DIR = Path(paths_cfg['packet_model_save_dir'])
PACKET_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Packet model checkpoints will be saved to: {PACKET_MODEL_SAVE_DIR}")
# REMOVED: PACKET_MODEL_CONFIG_SAVE_PATH related logging and saving


# --- 1. Load Sequence Dataset ---
logger.info(f"1. Loading packet sequence dataset from: {PROCESSED_PACKET_DATA_PATH} with max_seq_len={MAX_SEQ_LEN_PACKET}")
try:
    full_dataset = IoTSequenceDataset(PROCESSED_PACKET_DATA_PATH, max_seq_len=MAX_SEQ_LEN_PACKET)
    logger.info(f"   Dataset loaded. Total sequences: {len(full_dataset)}")
except Exception as e:
    logger.error(f"❌ Error loading packet dataset: {e}", exc_info=True)
    exit(1)

# --- 2. Initialize Transformer Model ---
logger.info("2. Initializing IoTTransformer model...")
try:
    # Assuming category_mappings.json is in a standard location or its path is also configured
    # For simplicity, keeping load_mappings as is, but its path might need to be configurable too.
    cat_map = load_mappings(is_flow=False)
    cat_sizes = {col: len(cat_map[col]) for col in categorical_columns_packets if col in cat_map}
    cat_pad = {c: cat_map[c]["unknown"] for c in categorical_columns_packets if c in cat_map and "unknown" in cat_map[c]}

    # Check if all categorical_columns_packets are in cat_map and have "unknown"
    for col in categorical_columns_packets:
        if col not in cat_sizes:
            logger.warning(f"Column {col} from categorical_columns_packets not found in loaded category mappings. Check mapping file or column list.")
            # Decide how to handle: error, skip, or provide default size/padding
            cat_sizes[col] = 1 # Default to size 1 if missing
            cat_pad[col] = 0   # Default to padding_idx 0 if missing
        elif "unknown" not in cat_map[col]:
            logger.warning(f"'unknown' category not found for column {col} in mappings. Using padding_idx 0 as default.")
            cat_pad[col] = 0 # Default if "unknown" key is missing

    model_arguments_for_instance = {
        "input_dim": INPUT_DIM_NUMERICAL_PACKET,
        "cat_sizes": cat_sizes,
        "cat_padding_idx": cat_pad,
        "embed_dim": transformer_body_cfg['d_model'],
        "num_heads": transformer_body_cfg['num_heads'],
        "num_layers": transformer_body_cfg['num_layers'],
        "dropout": transformer_body_cfg['dropout'],
        "num_classes": NUM_CLASSES_PACKET,
        "max_seq_len": MAX_SEQ_LEN_PACKET,
    }
    # REMOVED: Detailed logging of model_arguments_for_instance as JSON, individual params logged above
    # REMOVED: Saving of model_arguments_for_instance to PACKET_MODEL_CONFIG_SAVE_PATH

    model = IoTTransformer(**model_arguments_for_instance)
    model.to(DEVICE)
    logger.info("   ✅ IoTTransformer model instantiated and moved to device.")
except Exception as e:
    logger.error(f"❌ Error initializing model or loading category mappings: {e}", exc_info=True)
    exit(1)


# --- 3. Split Dataset ---
logger.info("3. Splitting dataset...")
dataset_params_cfg = config.get('dataset_params', {}) # Safe access
val_ratio = dataset_params_cfg.get('val_ratio_packet', 0.1)
test_ratio = dataset_params_cfg.get('test_ratio_packet', 0.1)

train_dataset, val_dataset, test_dataset = split_dataset_three_ways(
    full_dataset,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    logger_instance=logger
)

# --- 4. Train Model ---
logger.info("4. Starting model training...")
try:
    train_model( # Assuming train.py is updated for logger and num_workers
        model,
        train_dataset,
        val_dataset,
        epochs=EPOCHS_PRETRAIN,
        batch_size=BATCH_SIZE_PRETRAIN,
        lr=LR_PRETRAIN,
        device=DEVICE,
        clip_grad=CLIP_GRAD_PRETRAIN,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER_PRETRAIN,
        save_dir=str(PACKET_MODEL_SAVE_DIR),
        logger=logger,                 # Pass the logger
        num_workers=NUM_WORKERS_LOADER # Pass num_workers
    )
    logger.info("   ✅ Model training completed.")
except Exception as e:
    logger.error(f"❌ Error during model training: {e}", exc_info=True)
    exit(1)

# --- 5. Test Model ---
logger.info("5. Starting model testing...")
try:
    best_model_path_pretrain = PACKET_MODEL_SAVE_DIR / "model_best.pt"
    if not best_model_path_pretrain.exists():
        logger.warning(f"Best model checkpoint not found at {best_model_path_pretrain}. Testing with current model state.")
        model_path_to_test = None
    else:
        model_path_to_test = str(best_model_path_pretrain)

    test_model( # Assuming train.py is updated for logger
        model,
        test_dataset,
        batch_size=BATCH_SIZE_PRETRAIN,
        device=DEVICE,
        model_path=model_path_to_test,
        logger=logger                  # Pass the logger
    )
    logger.info("   ✅ Model testing completed.")
except Exception as e:
    logger.error(f"❌ Error during model testing: {e}", exc_info=True)
    exit(1)

logger.info("🏁 Packet pre-training script (with central config & logging) finished. 🏁")