# train_baseline_flow_only.py
import os
import torch
import torch.nn as nn
from torch.utils.data import random_split
import json
from pathlib import Path
import random # For random.seed
import numpy as np # For np.random.seed
import logging # Import the logging module

# Your existing imports
from pipeline.iot_flow_dataset import IoTFlowDataset
from models.flow_finetuning_model import FlowFineTuningModel # Re-using this class
from train.train_flows import fine_tune_flow_model, test_flow_model
# from pipeline.config import LABEL_MAPPING # If NUM_CLASSES_FLOW isn't in config

# --- 0. Load Central Configuration, Set Up Logger, and Set Seeds ---
CONFIG_FILE_PATH = Path("config.json") # Adjust path if your config is elsewhere

if not CONFIG_FILE_PATH.is_file():
    # Logging might not be set up yet if config fails to load, so print and exit
    print(f"❌ CRITICAL: Configuration file not found at {CONFIG_FILE_PATH}")
    exit(1)

with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)
# Initial print, will be replaced by logger shortly
print(f"✅ Central configuration loaded from {CONFIG_FILE_PATH}")



# Logging Setup
log_paths_config = config.get('paths', {})
LOG_FILE_NAME = log_paths_config.get('baseline_flow_only_log_file', 'training_baseline_flow_only.log')
LOG_DIR = Path(log_paths_config.get('log_dir', 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler() # Log to console
    ]
)
logger = logging.getLogger(__name__) # Get a logger for this script
# --- >>>> END OF NEW LOGGING SETUP <<<< ---

logger.info(f"✅ Central configuration loaded from {CONFIG_FILE_PATH}") # Logged properly now

# Apply Random Seed
SEED = config['general_settings']['random_seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
logger.info(f" Random seed set to: {SEED}")


# --- Helper Functions ---
# Modified to accept and use a logger
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
# General Settings
DEVICE_str = config['general_settings']['device']
if DEVICE_str == "cuda" and not torch.cuda.is_available():
    logger.warning("⚠️ CUDA specified in config but not available. Falling back to CPU.")
    DEVICE = "cpu"
else:
    DEVICE = DEVICE_str

NUM_WORKERS_LOADER = config['general_settings']['num_workers_loader']

# Paths
FLOW_DATASET_PATH = Path(config['dataset_paths']['processed_flow_pt'])
BASELINE_MODEL_SAVE_DIR = Path(config['paths']['flow_only_baseline_save_dir'])
BASELINE_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Model Architecture
transformer_body_config = config['model_architecture']['transformer_body']
D_MODEL_COMPARABLE = transformer_body_config['d_model']
NUM_LAYERS_COMPARABLE = transformer_body_config['num_layers']
NUM_HEADS_COMPARABLE = transformer_body_config['num_heads']
DROPOUT_COMPARABLE = transformer_body_config['dropout']
CLASSIFIER_DROPOUT = transformer_body_config.get('classifier_dropout', DROPOUT_COMPARABLE)
NUM_CLASSES_FLOW = config['model_architecture']['num_classes_flow']

# Training Hyperparameters
baseline_train_params = config['training_params']['baseline_flow_only']
EPOCHS_BASELINE = baseline_train_params['epochs']
LR_BASELINE = baseline_train_params['lr']
BATCH_SIZE_FLOW = baseline_train_params['batch_size']
CLIP_GRAD_CFG = baseline_train_params.get('clip_grad', 1.0)
USE_WEIGHTED_SAMPLER_CFG = baseline_train_params.get('use_weighted_sampler', False)


logger.info(f"--- Training Flow-Only Transformer Baseline (Using Central Config) ---")
logger.info(f"Device: {DEVICE}")
logger.info(f"Flow Dataset: {FLOW_DATASET_PATH}")
logger.info(f"Saving baseline model checkpoints to: {BASELINE_MODEL_SAVE_DIR}")
logger.info(f"Log file: {LOG_FILE_PATH}")
logger.info(f"Comparable Transformer body params: d_model={D_MODEL_COMPARABLE}, layers={NUM_LAYERS_COMPARABLE}, heads={NUM_HEADS_COMPARABLE}")
logger.info(f"Training params: Epochs={EPOCHS_BASELINE}, LR={LR_BASELINE}, BatchSize={BATCH_SIZE_FLOW}, NumWorkers={NUM_WORKERS_LOADER}")

# --- 1. Load Flow Dataset and Get Its Characteristics ---
logger.info(f"1. Loading flow dataset characteristics from: {FLOW_DATASET_PATH}")
try:
    flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
    num_flow_numerical_f = flow_data_pt['numerical_features'].shape[1]
    flow_cat_tensor = flow_data_pt['categorical_features']
    num_flow_categorical_f = flow_cat_tensor.shape[1]
    flow_cat_cardinalities = []
    if num_flow_categorical_f > 0:
        for i in range(num_flow_categorical_f):
            if flow_cat_tensor[:, i].numel() > 0:
                 flow_cat_cardinalities.append(int(torch.max(flow_cat_tensor[:, i])) + 1)
            else:
                 flow_cat_cardinalities.append(1)
    logger.info(f"   Flow data: NumNumerical={num_flow_numerical_f}, NumCategorical={num_flow_categorical_f}, CatCardinalities={flow_cat_cardinalities}")
    del flow_data_pt
except Exception as e:
    logger.error(f"❌ Error loading flow data characteristics: {e}", exc_info=True)
    exit(1)

full_flow_dataset = IoTFlowDataset(pt_file_path=FLOW_DATASET_PATH)

# --- 2. Prepare the Flow-Only Model (Train from Scratch) ---
logger.info("2. Preparing model for flow-only training (from scratch)...")
flow_only_encoder_layer = nn.TransformerEncoderLayer(
    d_model=D_MODEL_COMPARABLE,
    nhead=NUM_HEADS_COMPARABLE,
    dim_feedforward=D_MODEL_COMPARABLE * 4,
    dropout=DROPOUT_COMPARABLE,
    batch_first=True,
    activation=torch.nn.functional.relu
)
flow_only_transformer_body_scratch = nn.TransformerEncoder(
    encoder_layer=flow_only_encoder_layer,
    num_layers=NUM_LAYERS_COMPARABLE
)
flow_only_baseline_model = FlowFineTuningModel(
    num_flow_numerical_features=num_flow_numerical_f,
    flow_cat_cardinalities=flow_cat_cardinalities,
    d_model=D_MODEL_COMPARABLE,
    pretrained_transformer_encoder_body=flow_only_transformer_body_scratch,
    num_classes=NUM_CLASSES_FLOW,
    classifier_dropout=CLASSIFIER_DROPOUT
)
flow_only_baseline_model.to(DEVICE)
for name, param in flow_only_baseline_model.named_parameters():
    param.requires_grad = True
logger.info("   ✅ Flow-only model instantiated. All parameters are set to trainable.")

# --- 3. Split Flow Dataset ---
logger.info("3. Splitting flow dataset...")
train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(
    full_flow_dataset,
    val_ratio=config.get('dataset_params', {}).get('val_ratio', 0.1), # Example: make ratios configurable
    test_ratio=config.get('dataset_params', {}).get('test_ratio', 0.1),
    logger_instance=logger
)
# Original print now handled by logger inside split_dataset_three_ways

# --- 4. Train the Flow-Only Model ---
logger.info("4. Starting Training for Flow-Only Baseline Model...")
fine_tune_flow_model(
    model=flow_only_baseline_model,
    train_dataset=train_flow_dataset,
    val_dataset=val_flow_dataset,
    epochs=EPOCHS_BASELINE,
    batch_size=BATCH_SIZE_FLOW,
    lr=LR_BASELINE,
    device=DEVICE,
    save_dir=str(BASELINE_MODEL_SAVE_DIR),
    clip_grad=CLIP_GRAD_CFG,
    use_weighted_sampler=USE_WEIGHTED_SAMPLER_CFG,
    num_workers_loader=NUM_WORKERS_LOADER,
    logger=logger  # <<<< PASS THE LOGGER
)

# --- 5. Test the Flow-Only Model ---
logger.info("5. Testing Flow-Only Baseline Model...")
best_model_path = BASELINE_MODEL_SAVE_DIR / "model_flow_best.pt"
test_flow_model(
    model=flow_only_baseline_model,
    test_dataset=test_flow_dataset,
    batch_size=BATCH_SIZE_FLOW,
    device=DEVICE,
    model_path=str(best_model_path) if best_model_path.exists() else None,
    logger=logger
)

logger.info("🏁 Flow-only baseline training script (with central config & logging) finished. 🏁")