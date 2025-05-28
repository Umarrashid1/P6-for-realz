# train_baseline_flow_resume_hardcoded.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import json  # Will not be used, but keep for potential future reference of structure
from pathlib import Path
import random
import numpy as np
import logging
# argparse will not be used

# Project-specific imports
from pipeline.iot_flow_dataset import IoTFlowDataset
from models.flow_finetuning_model import FlowFineTuningModel
# Assuming train_flows_resumable.py contains fine_tune_flow_model_resumable
from train.train_flows_resumable import fine_tune_flow_model_resumable, test_flow_model

# --- 0. Hardcoded Configuration & Parameters ---

# General Settings
SEED = 42
DEVICE = "cuda"  # From log
NUM_WORKERS_LOADER = 12  # From log

# Paths
# Using paths relative to where this script might be run, similar to config
# Ensure these paths are correct for your execution environment.
BASE_PROJECT_PATH = Path(__file__).resolve().parent  # Assumes script is in P6-Packet_FlowTransformer
DATASET_BASE_PATH = BASE_PROJECT_PATH.parent.parent.parent / "dataset"  # Adjust if structure is different

FLOW_DATASET_PATH = DATASET_BASE_PATH / "processed_flows_subset_25pct.pt"  # From log
BASELINE_MODEL_SAVE_DIR = BASE_PROJECT_PATH / "checkpoints_flow_only_baseline"  # From log
LOG_DIR = BASE_PROJECT_PATH / "logs"
LOG_FILE_NAME = "training_baseline_flow_only_hardcoded_resume.log"  # New log file for this version

# Model Architecture (for FlowFineTuningModel baseline)
D_MODEL_COMPARABLE = 128  # From log
NUM_LAYERS_COMPARABLE = 4  # From log
NUM_HEADS_COMPARABLE = 4  # From log
DROPOUT_COMPARABLE = 0.1  # From original config.json (model_architecture.transformer_body.dropout)
CLASSIFIER_DROPOUT = 0.110841  # From original config.json (model_architecture.classifier_dropout_flow)
NUM_CLASSES_FLOW = 8  # From original config.json (model_architecture.num_classes_flow)

# Training Hyperparameters (for baseline)
EPOCHS_BASELINE = 10  # From log (total epochs for the run)
LR_BASELINE = 0.000101  # From log
BATCH_SIZE_FLOW = 64  # From log
CLIP_GRAD_CFG = 1.0  # From original config.json (training_params.baseline_flow_only.clip_grad)
USE_WEIGHTED_SAMPLER_CFG = False  # From original config.json

# Dataset Params (for splitting)
VAL_RATIO_FLOW = 0.1  # From original config.json
TEST_RATIO_FLOW = 0.1  # From original config.json

# Resume Parameters (Hardcoded for the specific scenario from the log)
ALLOW_RESUME = True  # We want to resume
# This is the 'latest' style checkpoint your resumable trainer saves.
# The 'best' model weights are in "model_baseline_flow_best.pt"
# For resuming optimizer state, "latest" is usually better if saved correctly.
# Your log shows 'model_flow_best.pt' being saved when F1 improved. Assuming it contains the full dict.
RESUME_CHECKPOINT_NAME = "model_baseline_flow_best.pt"  # To match the one saved in the log for epoch 7.
# The resumable trainer saves best model weights here.
# For full resume, `baseline_flow_checkpoint_latest.pt` is better if it has optimizer state.
# Given the log, it was saving `model_flow_best.pt` when F1 improved.
# Let's assume this best checkpoint HAS optimizer state.

# --- Logging Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'),  # Append to log
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"--- train_baseline_flow_resume_hardcoded.py ---")
logger.info(f"✅ Parameters are HARDCODED for resuming run similar to baseline_flow_only_138836.err")

# Apply Random Seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
logger.info(f"🌱 Random seed set to: {SEED}")


# --- Helper Functions ---
def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1, logger_instance=None):
    if logger_instance is None:
        logger_instance = logger
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    if train_size < 0:
        logger_instance.warning(
            f"Dataset too small for specified ratios (Total: {total_size}, Val: {val_size}, Test: {test_size}). Adjusting splits.")
        if total_size * (1 - val_ratio - test_ratio) <= 0:
            train_size = max(1, int(total_size * 0.7))
            val_size = max(1, int(total_size * 0.15))
            test_size = total_size - train_size - val_size
            if test_size < 0: test_size = 0
        else:
            train_size = total_size - val_size - test_size
    logger_instance.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    return random_split(dataset, [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(SEED))


# --- Main Script Logic ---
logger.info(f"Device: {DEVICE}")
logger.info(f"Flow Dataset: {FLOW_DATASET_PATH}")
logger.info(f"Saving baseline model checkpoints to: {BASELINE_MODEL_SAVE_DIR}")
logger.info(f"Log file: {LOG_FILE_PATH}")
logger.info(
    f"Comparable Transformer body params: d_model={D_MODEL_COMPARABLE}, layers={NUM_LAYERS_COMPARABLE}, heads={NUM_HEADS_COMPARABLE}")
logger.info(
    f"Training params: Epochs={EPOCHS_BASELINE}, LR={LR_BASELINE}, BatchSize={BATCH_SIZE_FLOW}, NumWorkers={NUM_WORKERS_LOADER}")

# --- 1. Load Flow Dataset and Get Its Characteristics ---
logger.info(f"1. Loading flow dataset characteristics from: {FLOW_DATASET_PATH}")
if not FLOW_DATASET_PATH.exists():
    logger.error(f"❌ FATAL: Flow dataset not found at {FLOW_DATASET_PATH}. Please check the path.")
    exit(1)
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
    logger.info(
        f"   Flow data: NumNumerical={num_flow_numerical_f}, NumCategorical={num_flow_categorical_f}, CatCardinalities={flow_cat_cardinalities}")
    del flow_data_pt
except Exception as e:
    logger.error(f"❌ Error loading flow data characteristics: {e}", exc_info=True)
    exit(1)

full_flow_dataset = IoTFlowDataset(pt_file_path=FLOW_DATASET_PATH)

# --- 2. Prepare the Flow-Only Model (Train from Scratch or Resume) ---
logger.info("2. Preparing model for flow-only training...")
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

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, flow_only_baseline_model.parameters()), lr=LR_BASELINE)
logger.info(f"Optimizer AdamW created with LR: {LR_BASELINE}")

start_epoch = 1
best_val_f1_so_far = -1.0
resumable_checkpoint_file_path = BASELINE_MODEL_SAVE_DIR / RESUME_CHECKPOINT_NAME

if ALLOW_RESUME and resumable_checkpoint_file_path.is_file():
    logger.info(f"Attempting to resume training from checkpoint: {resumable_checkpoint_file_path}")
    try:
        checkpoint = torch.load(resumable_checkpoint_file_path, map_location=DEVICE)
        flow_only_baseline_model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully.")
        else:
            logger.warning(
                f"Optimizer state not found in checkpoint {resumable_checkpoint_file_path}. Optimizer will start fresh.")

        # The log showed epoch 7 completed and was the best.
        # If resuming from model_baseline_flow_best.pt saved at epoch 7:
        start_epoch = checkpoint.get('epoch', 0) + 1  # Your resumable saver stores the completed epoch
        best_val_f1_so_far = checkpoint.get('best_val_f1', -1.0)

        logger.info(
            f"Resuming from epoch {start_epoch}. Previous best F1: {best_val_f1_so_far:.4f} (from checkpoint epoch {checkpoint.get('epoch', 'N/A')})")

        if optimizer.param_groups[0]['lr'] != LR_BASELINE:
            logger.info(
                f"Updating optimizer LR from {optimizer.param_groups[0]['lr']:.6f} (checkpoint) to {LR_BASELINE:.6f} (hardcoded).")
            optimizer.param_groups[0]['lr'] = LR_BASELINE
    except Exception as e:
        logger.error(f"Could not load checkpoint from {resumable_checkpoint_file_path}: {e}. Starting fresh.",
                     exc_info=True)
        start_epoch = 1
        best_val_f1_so_far = -1.0
else:
    if ALLOW_RESUME:
        logger.info(f"No checkpoint found at {resumable_checkpoint_file_path} or resuming not allowed. Starting fresh.")
    else:  # This case won't be hit with ALLOW_RESUME = True hardcoded
        logger.info("Starting fresh training (ALLOW_RESUME is False).")
    start_epoch = 1
    best_val_f1_so_far = -1.0

if start_epoch > EPOCHS_BASELINE:
    logger.info(
        f"Start epoch ({start_epoch}) is greater than total epochs ({EPOCHS_BASELINE}). No further training needed.")
else:
    logger.info(f"   Model configured. Training will run from epoch {start_epoch} to {EPOCHS_BASELINE}.")

# --- 3. Split Flow Dataset ---
logger.info("3. Splitting flow dataset...")
train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(
    full_flow_dataset,
    val_ratio=VAL_RATIO_FLOW,
    test_ratio=TEST_RATIO_FLOW,
    logger_instance=logger
)

# --- 4. Train the Flow-Only Model using the Resumable Function ---
if start_epoch <= EPOCHS_BASELINE:
    logger.info(f"4. Starting/Resuming Training for Flow-Only Baseline Model from epoch {start_epoch}...")
    fine_tune_flow_model_resumable(
        model=flow_only_baseline_model,
        train_dataset=train_flow_dataset,
        val_dataset=val_flow_dataset,
        optimizer=optimizer,
        start_epoch=start_epoch,
        best_val_f1_so_far=best_val_f1_so_far,
        epochs=EPOCHS_BASELINE,
        batch_size=BATCH_SIZE_FLOW,
        device=DEVICE,
        clip_grad=CLIP_GRAD_CFG,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER_CFG,
        save_dir=str(BASELINE_MODEL_SAVE_DIR),
        resumable_checkpoint_name="baseline_flow_checkpoint_latest.pt",  # Saves latest state here
        best_model_weights_name="model_baseline_flow_best.pt",  # Saves best model weights here
        num_workers_loader=NUM_WORKERS_LOADER,
        logger=logger
    )
else:
    logger.info("Skipping training as start_epoch > total_epochs.")

# --- 5. Test the Flow-Only Model ---
logger.info("5. Testing Flow-Only Baseline Model...")
best_model_weights_path_for_test = BASELINE_MODEL_SAVE_DIR / "model_baseline_flow_best.pt"

if not best_model_weights_path_for_test.exists():
    logger.warning(
        f"Best model weights file {best_model_weights_path_for_test} not found. Testing might use the last state of the model in memory.")

test_flow_model(
    model=flow_only_baseline_model,
    test_dataset=test_flow_dataset,
    batch_size=BATCH_SIZE_FLOW,
    device=DEVICE,
    model_path=str(best_model_weights_path_for_test) if best_model_weights_path_for_test.exists() else None,
    logger=logger
)

logger.info("🏁 Flow-only baseline training script (hardcoded, with resume) finished. 🏁")