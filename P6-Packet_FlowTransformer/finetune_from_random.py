# finetune_from_random.py
import os
import torch
from torch.utils.data import random_split
import json
from pathlib import Path
import random
import numpy as np
import logging
import argparse

from pipeline.iot_flow_dataset import IoTFlowDataset
from models.packet_pretraining_model import PacketPretrainingModel
from models.flow_finetuning_model import FlowFineTuningModel
from train.train_flows import fine_tune_flow_model, test_flow_model
from utils.category_mapping import load_mappings
from pipeline.config import categorical_columns_packets, numerical_columns_packets

CONFIG_FILE_PATH = Path("config.json")

if not CONFIG_FILE_PATH.is_file():
    print(f"❌ CRITICAL: Configuration file not found at {CONFIG_FILE_PATH}")
    exit(1)

with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

log_paths_config = config.get('paths', {})
# New log file for this experiment
LOG_FILE_NAME = log_paths_config.get('finetuned_model_log_file', 'training_finetuned_from_random.log')
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
            train_size = total_size - val_size - test_size
    logger_instance.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    return random_split(dataset, [train_size, val_size, test_size])


def freeze_transformer_body(model: FlowFineTuningModel, logger_instance=None):
    if logger_instance is None:
        logger_instance = logger
    TARGET_BODY_PREFIX = "transformer_encoder_body."
    frz_c = 0
    total_params_body = 0
    for name, param in model.named_parameters():
        if name.startswith(TARGET_BODY_PREFIX):
            param.requires_grad = False
            frz_c += 1
            total_params_body += param.numel()
    logger_instance.info(f"🧊 Frozen {frz_c} parameter groups ({total_params_body:,} params) in '{TARGET_BODY_PREFIX}'.")


def main(args):
    logger.info(f"--- Initializing Fine-tuning from RANDOM WEIGHTS ---")

    # --- Load Configuration ---
    DEVICE_str = config['general_settings']['device']
    if DEVICE_str == "cuda" and not torch.cuda.is_available():
        DEVICE = "cpu"
    else:
        DEVICE = DEVICE_str
    NUM_WORKERS_LOADER = config['general_settings']['num_workers_loader']

    paths_cfg = config['paths']
    FLOW_DATASET_PATH = Path(config['dataset_paths']['processed_flow_pt'])
    # New save directory for this experiment
    FINETUNED_MODEL_SAVE_DIR = Path(paths_cfg.get('finetuned_model_save_dir', 'checkpoints_flow_finetuned_from_random'))
    FINETUNED_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


    model_arch_cfg = config['model_architecture']
    transformer_body_cfg = model_arch_cfg['transformer_body']
    NUM_LAYERS = transformer_body_cfg['num_layers']
    D_MODEL_PRETRAINED = transformer_body_cfg['d_model']
    CLASSIFIER_DROPOUT_CFG = model_arch_cfg.get('classifier_dropout_flow', transformer_body_cfg['dropout'])
    NUM_CLASSES_FLOW = model_arch_cfg['num_classes_flow']

    finetune_params_cfg = config['training_params']['fine_tuning_flow']
    BATCH_SIZE_FLOW = finetune_params_cfg['batch_size']
    CLIP_GRAD_CFG = finetune_params_cfg.get('clip_grad', 1.0)

    EPOCHS_HEAD_ONLY = 2
    EPOCHS_PER_UNFREEZE_STEP = 2
    LR_HEAD = finetune_params_cfg['lr']
    LR_FINETUNE_BODY = LR_HEAD / 10

    logger.info(f"Device: {DEVICE}, Num Workers: {NUM_WORKERS_LOADER}")
    logger.info(
        f"Gradual Unfreezing Strategy: Head Only Epochs={EPOCHS_HEAD_ONLY}, Epochs per Unfreeze Step={EPOCHS_PER_UNFREEZE_STEP}")
    logger.info(f"Learning Rates: Head={LR_HEAD}, Body={LR_FINETUNE_BODY}")

    # --- 1. Load Data ---
    logger.info(f"1. Loading flow dataset characteristics from: {FLOW_DATASET_PATH}")
    try:
        flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
        num_flow_numerical_f = flow_data_pt['numerical_features'].shape[1]
        flow_cat_tensor = flow_data_pt['categorical_features']
        num_flow_categorical_f = flow_cat_tensor.shape[1] if flow_cat_tensor.numel() > 0 else 0
        flow_cat_cardinalities = [int(torch.max(flow_cat_tensor[:, i])) + 1 for i in
                                  range(num_flow_categorical_f)] if num_flow_categorical_f > 0 else []
        del flow_data_pt
    except Exception as e:
        logger.error(f"❌ Error loading flow data characteristics: {e}", exc_info=True)
        exit(1)

    full_flow_dataset = IoTFlowDataset(pt_file_path=FLOW_DATASET_PATH)

    logger.info("2. Preparing model for fine-tuning...")
    try:
        dummy_packet_model = PacketPretrainingModel(
            input_dim=len(numerical_columns_packets), cat_sizes={}, cat_padding_idx={},
            embed_dim=D_MODEL_PRETRAINED, num_heads=transformer_body_cfg['num_heads'],
            num_layers=NUM_LAYERS, dropout=transformer_body_cfg['dropout'],
            num_classes=model_arch_cfg['num_classes_packet'], max_seq_len=model_arch_cfg['max_seq_len_packet']
        )
        pretrained_transformer_body = dummy_packet_model.transformer

        flow_finetuning_model_instance = FlowFineTuningModel(
            num_flow_numerical_features=num_flow_numerical_f,
            flow_cat_cardinalities=flow_cat_cardinalities,
            d_model=D_MODEL_PRETRAINED,
            pretrained_transformer_encoder_body=pretrained_transformer_body,
            num_classes=NUM_CLASSES_FLOW,
            classifier_dropout=CLASSIFIER_DROPOUT_CFG
        )
        flow_finetuning_model_instance.to(DEVICE)
        logger.info("   ✅ FlowFineTuningModel instantiated.")

        # MODIFICATION: Skip loading pre-trained weights
        logger.info("SKIPPING loading of pre-trained weights. Using randomly initialized transformer body.")


    except Exception as e:
        logger.error(f"❌ Error during model preparation: {e}", exc_info=True)
        exit(1)

    logger.info("3. Splitting flow dataset...")
    val_ratio = config.get('dataset_params', {}).get('val_ratio_flow', 0.1)
    test_ratio = config.get('dataset_params', {}).get('test_ratio_flow', 0.1)
    train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(full_flow_dataset, val_ratio,
                                                                                       test_ratio, logger)

    # --- 4. Gradual Unfreezing and Staged Training ---
    try:
        # --- Stage 1: Freeze Body, Train Head ---
        logger.info("\n" + "=" * 25 + " Stage 1: Training Classifier Head Only " + "=" * 25)
        freeze_transformer_body(flow_finetuning_model_instance, logger)
        fine_tune_flow_model(
            model=flow_finetuning_model_instance, train_dataset=train_flow_dataset, val_dataset=val_flow_dataset,
            epochs=EPOCHS_HEAD_ONLY, batch_size=BATCH_SIZE_FLOW, lr=LR_HEAD, device=DEVICE,
            save_dir=str(FINETUNED_MODEL_SAVE_DIR), clip_grad=CLIP_GRAD_CFG,
            num_workers_loader=NUM_WORKERS_LOADER, logger=logger,
            patience=3, use_lr_scheduler=True
        )

        # --- Stage 2: Unfreeze layers one by one ---
        for i in range(NUM_LAYERS - 1, -1, -1):
            logger.info("\n" + f" =" * 25 + f" Stage 2.{NUM_LAYERS - i}: Unfreezing Layer {i} " + f"=" * 25)

            for param in flow_finetuning_model_instance.transformer_encoder_body.layers[i].parameters():
                param.requires_grad = True

            trainable_params_count = sum(
                p.numel() for p in flow_finetuning_model_instance.parameters() if p.requires_grad)
            logger.info(f"   🧊 Unfroze layer {i}. Total trainable parameters: {trainable_params_count:,}")

            fine_tune_flow_model(
                model=flow_finetuning_model_instance, train_dataset=train_flow_dataset, val_dataset=val_flow_dataset,
                epochs=EPOCHS_PER_UNFREEZE_STEP, batch_size=BATCH_SIZE_FLOW, lr=LR_FINETUNE_BODY, device=DEVICE,
                save_dir=str(FINETUNED_MODEL_SAVE_DIR), clip_grad=CLIP_GRAD_CFG,
                num_workers_loader=NUM_WORKERS_LOADER, logger=logger,
                patience=3, use_lr_scheduler=True,
                resume_checkpoint_path=str(FINETUNED_MODEL_SAVE_DIR / "model_flow_best.pt"),
                load_optimizer_state=False
            )

        logger.info("\n" + "=" * 30 + " Gradual Unfreezing Complete " + "=" * 30)

    except Exception as e:
        logger.error(f"❌ Error during multi-stage fine-tuning: {e}", exc_info=True)
        exit(1)

    # --- 5. Final Testing ---
    logger.info("5. Testing Final Fine-tuned Model...")
    try:
        best_model_path_to_test = FINETUNED_MODEL_SAVE_DIR / "model_flow_best.pt"
        test_flow_model(
            model=flow_finetuning_model_instance,
            test_dataset=test_flow_dataset, batch_size=BATCH_SIZE_FLOW, device=DEVICE,
            model_path=str(best_model_path_to_test) if best_model_path_to_test.exists() else None,
            logger=logger
        )
        logger.info("   ✅ Fine-tuned model testing completed.")
    except Exception as e:
        logger.error(f"❌ Error during fine-tuned model testing: {e}", exc_info=True)
        exit(1)

    logger.info("🏁 Fine-tuning from random weights script finished. 🏁")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a flow model from random weights with gradual unfreezing.")
    args = parser.parse_args()
    main(args)