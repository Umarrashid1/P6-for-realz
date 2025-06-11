# finetune.py
import os
import torch
from torch.utils.data import random_split
import json
from pathlib import Path
import random
import numpy as np
import logging
import argparse  # New: Import argparse

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
LOG_FILE_NAME = log_paths_config.get('finetuned_model_log_file', 'training_finetuned_model.log')
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


def load_transformer_body_weights(target_model: FlowFineTuningModel, pretrained_checkpoint_path: str, device: str,
                                  logger_instance=None):
    if logger_instance is None:
        logger_instance = logger
    logger_instance.info(f"🔄 Loading Transformer Body weights from: {pretrained_checkpoint_path}")
    try:
        # Load the full checkpoint first to inspect its keys
        full_checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)

        # Determine if it's an old state_dict or new checkpoint format
        if 'model_state_dict' in full_checkpoint:
            pretrained_state_dict = full_checkpoint['model_state_dict']
            logger_instance.info("   Loaded 'model_state_dict' from new checkpoint format.")
        else:
            # Assume it's an old format (just the state_dict)
            pretrained_state_dict = full_checkpoint
            logger_instance.info("   Loaded state_dict directly (assuming old checkpoint format).")

    except FileNotFoundError:
        logger_instance.error(f"❌ Error: Pretrained model file not found: {pretrained_checkpoint_path}")
        return False
    except Exception as e:
        logger_instance.error(f"❌ Error loading pretrained model checkpoint: {e}", exc_info=True)
        return False

    PRETRAINED_BODY_PREFIX = "transformer."
    TARGET_BODY_PREFIX = "transformer_encoder_body."
    body_weights = {
        TARGET_BODY_PREFIX + k[len(PRETRAINED_BODY_PREFIX):]: v
        for k, v in pretrained_state_dict.items() if k.startswith(PRETRAINED_BODY_PREFIX)
    }
    if not body_weights:
        logger_instance.error(
            f"❌ No weights found with prefix '{PRETRAINED_BODY_PREFIX}' in checkpoint {pretrained_checkpoint_path}.")
        return False

    missing, unexpected = target_model.load_state_dict(body_weights, strict=False)
    logger_instance.info(f"✅ Loaded {len(body_weights)} parameter groups into '{TARGET_BODY_PREFIX}'.")
    if missing:
        genuinely_missing_in_body = [k for k in missing if k.startswith(TARGET_BODY_PREFIX)]
        if genuinely_missing_in_body:
            logger_instance.warning(
                f"   ⚠️ Missing in target transformer_encoder_body: {genuinely_missing_in_body[:5]}...")
    if unexpected:
        logger_instance.error(f"   ❌ Unexpected keys in source checkpoint: {unexpected[:5]}...")
        # return False # Commenting out, as fine-tuning might still proceed if only a few unexpected keys
    return True


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
    if frz_c == 0 and sum(p.numel() for name, p in model.named_parameters() if name.startswith(TARGET_BODY_PREFIX)) > 0:
        logger_instance.warning(
            f"   ⚠️ No parameters were actually frozen with prefix '{TARGET_BODY_PREFIX}', but body has params.")


def main(args):  # New: main function
    logger.info(f"--- Initializing Fine-tuning Script (Using Central Config) ---")
    if args.resume_checkpoint:
        logger.info(f"Attempting to resume fine-tuning from checkpoint: {args.resume_checkpoint}")

    DEVICE_str = config['general_settings']['device']
    if DEVICE_str == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA specified in config but not available. Falling back to CPU.")
        DEVICE = "cpu"
    else:
        DEVICE = DEVICE_str
    NUM_WORKERS_LOADER = config['general_settings']['num_workers_loader']

    paths_cfg = config['paths']
    FLOW_DATASET_PATH = Path(config['dataset_paths']['processed_flow_pt'])
    PRETRAINED_PACKET_MODEL_WEIGHTS = Path(paths_cfg['packet_model_save_dir']) / "model_best.pt"
    FINETUNED_MODEL_SAVE_DIR = Path(paths_cfg['finetuned_model_save_dir'])
    FINETUNED_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {DEVICE}, Num Workers: {NUM_WORKERS_LOADER}")
    logger.info(f"Flow Dataset: {FLOW_DATASET_PATH}")
    logger.info(f"Pretrained Packet Model Weights: {PRETRAINED_PACKET_MODEL_WEIGHTS}")
    logger.info(f"Fine-tuned Model Save Dir: {FINETUNED_MODEL_SAVE_DIR}")

    model_arch_cfg = config['model_architecture']
    transformer_body_cfg = model_arch_cfg['transformer_body']
    D_MODEL_PRETRAINED = transformer_body_cfg['d_model']
    CLASSIFIER_DROPOUT_CFG = model_arch_cfg.get('classifier_dropout_flow', transformer_body_cfg['dropout'])
    INPUT_DIM_NUMERICAL_PACKET = model_arch_cfg.get('input_dim_packet_numerical', len(numerical_columns_packets))
    MAX_SEQ_LEN_PACKET = model_arch_cfg['max_seq_len_packet']
    NUM_CLASSES_PACKET_PRETRAIN = model_arch_cfg['num_classes_packet']
    NUM_CLASSES_FLOW = model_arch_cfg['num_classes_flow']

    finetune_params_cfg = config['training_params']['fine_tuning_flow']
    EPOCHS_FINETUNE = finetune_params_cfg['epochs']
    LR_FINETUNE = finetune_params_cfg['lr']
    BATCH_SIZE_FLOW = finetune_params_cfg['batch_size']
    CLIP_GRAD_CFG = finetune_params_cfg.get('clip_grad', 1.0)
    USE_WEIGHTED_SAMPLER_CFG = finetune_params_cfg.get('use_weighted_sampler', False)

    logger.info(
        f"Pretrained Transformer Body: d_model={D_MODEL_PRETRAINED}, layers={transformer_body_cfg['num_layers']}, heads={transformer_body_cfg['num_heads']}")
    logger.info(f"Fine-tuning Params: Epochs={EPOCHS_FINETUNE}, LR={LR_FINETUNE}, BatchSize={BATCH_SIZE_FLOW}")

    logger.info(f"1. Loading flow dataset characteristics from: {FLOW_DATASET_PATH}")
    try:
        flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
        num_flow_numerical_f = flow_data_pt['numerical_features'].shape[1]
        flow_cat_tensor = flow_data_pt['categorical_features']
        num_flow_categorical_f = flow_cat_tensor.shape[1] if flow_cat_tensor.numel() > 0 else 0
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

    logger.info("2. Preparing model for fine-tuning...")
    try:
        cat_map_packet_path = config.get('paths', {}).get('category_mappings_packet', 'category_mappings_packets.json')
        cat_map_packets = load_mappings(path=cat_map_packet_path, is_flow=False)
        cat_sizes_packets = {col: len(cat_map_packets[col]) for col in categorical_columns_packets if
                             col in cat_map_packets}
        cat_pad_packets = {c: cat_map_packets[c].get("unknown", 0) for c in categorical_columns_packets if
                           c in cat_map_packets}
        for col in categorical_columns_packets:  # Ensure all packet categoricals have a default
            if col not in cat_sizes_packets: cat_sizes_packets[col] = 1
            if col not in cat_pad_packets: cat_pad_packets[col] = 0

        temp_packet_model_args = {
            'input_dim': INPUT_DIM_NUMERICAL_PACKET, 'cat_sizes': cat_sizes_packets,
            'cat_padding_idx': cat_pad_packets, 'embed_dim': D_MODEL_PRETRAINED,
            'num_heads': transformer_body_cfg['num_heads'], 'num_layers': transformer_body_cfg['num_layers'],
            'dropout': transformer_body_cfg['dropout'], 'num_classes': NUM_CLASSES_PACKET_PRETRAIN,
            'max_seq_len': MAX_SEQ_LEN_PACKET
        }
        original_packet_model_for_body = PacketPretrainingModel(**temp_packet_model_args)
        if not hasattr(original_packet_model_for_body, 'transformer'):
            logger.error("PacketPretrainingModel needs 'transformer' attribute.");
            exit(1)
        pretrained_transformer_body = original_packet_model_for_body.transformer

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

        if not PRETRAINED_PACKET_MODEL_WEIGHTS.is_file():
            logger.error(
                f"❌ Pretrained weights {PRETRAINED_PACKET_MODEL_WEIGHTS} not found. Fine-tuning from scratch body (if not resuming).")
            # If not resuming, the body will be randomly initialized. If resuming, checkpoint will overwrite.
        else:
            if not load_transformer_body_weights(flow_finetuning_model_instance, str(PRETRAINED_PACKET_MODEL_WEIGHTS),
                                                 DEVICE, logger):
                logger.warning(
                    "⚠️ Failed to load pre-trained weights into body. Body might be randomly initialized unless resuming.")
            else:
                freeze_transformer_body(flow_finetuning_model_instance, logger)

    except Exception as e:
        logger.error(f"❌ Error during model preparation: {e}", exc_info=True)
        exit(1)

    logger.info("3. Splitting flow dataset...")
    dataset_params_cfg = config.get('dataset_params', {})
    val_ratio = dataset_params_cfg.get('val_ratio_flow', 0.1)
    test_ratio = dataset_params_cfg.get('test_ratio_flow', 0.1)
    train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(
        full_flow_dataset, val_ratio=val_ratio, test_ratio=test_ratio, logger_instance=logger
    )

    # logger.info("4. Starting Fine-tuning on Flow Data...")
    # try:
    #     fine_tune_flow_model(
    #         model=flow_finetuning_model_instance,
    #         train_dataset=train_flow_dataset,
    #         val_dataset=val_flow_dataset,
    #         epochs=EPOCHS_FINETUNE, batch_size=BATCH_SIZE_FLOW, lr=LR_FINETUNE, device=DEVICE,
    #         save_dir=str(FINETUNED_MODEL_SAVE_DIR), clip_grad=CLIP_GRAD_CFG,
    #         use_weighted_sampler=USE_WEIGHTED_SAMPLER_CFG, num_workers_loader=NUM_WORKERS_LOADER,
    #         logger=logger,
    #         resume_checkpoint_path=args.resume_checkpoint  # Pass the argument here
    #     )
    #     logger.info("   ✅ Model fine-tuning completed.")
    # except Exception as e:
    #     logger.error(f"❌ Error during model fine-tuning: {e}", exc_info=True)
    #     exit(1)

    logger.info("5. Testing Fine-tuned Flow Model...")
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

    logger.info("🏁 Fine-tuning script finished. 🏁")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a flow model, with an option to resume from a checkpoint.")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume fine-tuning from (e.g., checkpoints_flow_finetuned_optuna_best_s/model_flow_best.pt)."
    )
    args = parser.parse_args()
    main(args)