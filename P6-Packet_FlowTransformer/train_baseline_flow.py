# train_baseline_flow_only.py
import os
import torch
import torch.nn as nn
from torch.utils.data import random_split
import json
from pathlib import Path
import random
import numpy as np
import logging
import argparse # New: Import argparse

from pipeline.iot_flow_dataset import IoTFlowDataset
from models.flow_finetuning_model import FlowFineTuningModel
from train.train_flows import fine_tune_flow_model, test_flow_model

CONFIG_FILE_PATH = Path("config.json")

if not CONFIG_FILE_PATH.is_file():
    print(f"❌ CRITICAL: Configuration file not found at {CONFIG_FILE_PATH}")
    exit(1)

with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"✅ Central configuration loaded from {CONFIG_FILE_PATH}")

SEED = config['general_settings']['random_seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
logger.info(f"🌱 Random seed set to: {SEED}")

def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1, logger_instance=None):
    if logger_instance is None:
        logger_instance = logger
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    if train_size < 0:
        logger_instance.warning(f"Dataset too small for specified ratios. Adjusting splits.")
        if total_size * (1 - val_ratio - test_ratio) <= 0:
            train_size = max(1, int(total_size * 0.7))
            val_size = max(1, int(total_size * 0.15))
            test_size = total_size - train_size - val_size
            if test_size < 0: test_size = 0
        else:
            train_size = total_size - val_size - test_size # Should be okay if first condition failed
    logger_instance.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    return random_split(dataset, [train_size, val_size, test_size])

def main(args): # New: main function to take parsed args
    logger.info(f"--- Training Flow-Only Transformer Baseline (Using Central Config) ---")
    if args.resume_checkpoint:
        logger.info(f"Attempting to resume from checkpoint: {args.resume_checkpoint}")

    DEVICE_str = config['general_settings']['device']
    if DEVICE_str == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA specified in config but not available. Falling back to CPU.")
        DEVICE = "cpu"
    else:
        DEVICE = DEVICE_str
    NUM_WORKERS_LOADER = config['general_settings']['num_workers_loader']
    FLOW_DATASET_PATH = Path(config['dataset_paths']['processed_flow_pt'])
    BASELINE_MODEL_SAVE_DIR = Path(config['paths']['flow_only_baseline_save_dir'])
    BASELINE_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    transformer_body_config = config['model_architecture']['transformer_body']
    D_MODEL_COMPARABLE = transformer_body_config['d_model']
    NUM_LAYERS_COMPARABLE = transformer_body_config['num_layers']
    NUM_HEADS_COMPARABLE = transformer_body_config['num_heads']
    DROPOUT_COMPARABLE = transformer_body_config['dropout']
    CLASSIFIER_DROPOUT = config['model_architecture'].get('classifier_dropout_flow', DROPOUT_COMPARABLE) # Use flow specific if present
    NUM_CLASSES_FLOW = config['model_architecture']['num_classes_flow']

    baseline_train_params = config['training_params']['baseline_flow_only']
    EPOCHS_BASELINE = baseline_train_params['epochs']
    LR_BASELINE = baseline_train_params['lr']
    BATCH_SIZE_FLOW = baseline_train_params['batch_size']
    CLIP_GRAD_CFG = baseline_train_params.get('clip_grad', 1.0)
    USE_WEIGHTED_SAMPLER_CFG = baseline_train_params.get('use_weighted_sampler', False)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Flow Dataset: {FLOW_DATASET_PATH}")
    logger.info(f"Saving baseline model checkpoints to: {BASELINE_MODEL_SAVE_DIR}")
    logger.info(f"Log file: {LOG_FILE_PATH}")
    logger.info(f"Comparable Transformer body params: d_model={D_MODEL_COMPARABLE}, layers={NUM_LAYERS_COMPARABLE}, heads={NUM_HEADS_COMPARABLE}")
    logger.info(f"Training params: Epochs={EPOCHS_BASELINE}, LR={LR_BASELINE}, BatchSize={BATCH_SIZE_FLOW}, NumWorkers={NUM_WORKERS_LOADER}")

    logger.info(f"1. Loading flow dataset characteristics from: {FLOW_DATASET_PATH}")
    try:
        flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
        num_flow_numerical_f = flow_data_pt['numerical_features'].shape[1]
        flow_cat_tensor = flow_data_pt['categorical_features']
        num_flow_categorical_f = flow_cat_tensor.shape[1] if flow_cat_tensor.numel() > 0 else 0 # Handle empty tensor
        flow_cat_cardinalities = []
        if num_flow_categorical_f > 0:
            for i in range(num_flow_categorical_f):
                if flow_cat_tensor[:, i].numel() > 0:
                    flow_cat_cardinalities.append(int(torch.max(flow_cat_tensor[:, i])) + 1)
                else:
                    flow_cat_cardinalities.append(1) # Should not happen if num_flow_cat_f > 0 and correctly shaped
        logger.info(f"   Flow data: NumNumerical={num_flow_numerical_f}, NumCategorical={num_flow_categorical_f}, CatCardinalities={flow_cat_cardinalities}")
        del flow_data_pt
    except Exception as e:
        logger.error(f"❌ Error loading flow data characteristics: {e}", exc_info=True)
        exit(1)

    full_flow_dataset = IoTFlowDataset(pt_file_path=FLOW_DATASET_PATH)

    logger.info("2. Preparing model for flow-only training (from scratch if not resuming)...")
    flow_only_encoder_layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL_COMPARABLE, nhead=NUM_HEADS_COMPARABLE,
        dim_feedforward=D_MODEL_COMPARABLE * 4, dropout=DROPOUT_COMPARABLE,
        batch_first=True, activation=torch.nn.functional.relu
    )
    flow_only_transformer_body_scratch = nn.TransformerEncoder(
        encoder_layer=flow_only_encoder_layer, num_layers=NUM_LAYERS_COMPARABLE
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
        param.requires_grad = True # Ensure all params are trainable initially
    logger.info("   ✅ Flow-only model instantiated. All parameters are set to trainable (will be overridden by checkpoint if resuming).")

    logger.info("3. Splitting flow dataset...")
    dataset_params_cfg = config.get('dataset_params', {})
    val_ratio = dataset_params_cfg.get('val_ratio_flow', 0.1)
    test_ratio = dataset_params_cfg.get('test_ratio_flow', 0.1)
    train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(
        full_flow_dataset, val_ratio=val_ratio, test_ratio=test_ratio, logger_instance=logger
    )

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
        logger=logger,
        resume_checkpoint_path=args.resume_checkpoint # Pass the argument here
    )

    logger.info("5. Testing Flow-Only Baseline Model...")
    best_model_path_to_test = BASELINE_MODEL_SAVE_DIR / "model_flow_best.pt"
    test_flow_model(
        model=flow_only_baseline_model, # Model will have best weights loaded if training just finished and saved
        test_dataset=test_flow_dataset,
        batch_size=BATCH_SIZE_FLOW,
        device=DEVICE,
        model_path=str(best_model_path_to_test) if best_model_path_to_test.exists() else None, # Explicitly pass path to load
        logger=logger
    )

    logger.info("🏁 Flow-only baseline training script finished. 🏁")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline flow model, with an option to resume from a checkpoint.")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from (e.g., checkpoints_flow_only_baseline/model_flow_best.pt)."
    )
    args = parser.parse_args()
    main(args)