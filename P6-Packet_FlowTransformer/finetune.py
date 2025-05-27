# finetune.py
import os
import torch
from torch.utils.data import random_split
import json
from pathlib import Path
import random
import numpy as np
import logging

# Project-specific imports
from pipeline.iot_flow_dataset import IoTFlowDataset
from models.packet_pretraining_model import PacketPretrainingModel  # Needed to reconstruct pre-trained model arch
from models.flow_finetuning_model import FlowFineTuningModel
from train.train_flows import fine_tune_flow_model, test_flow_model  # Assumes this is updated for logger
from utils.category_mapping import load_mappings  # For packet cat_sizes if needed for IoTTransformer
from pipeline.config import categorical_columns_packets, numerical_columns_packets  # For IoTTransformer instantiation

# Load Central Configuration, Set Up Logger, and Set Seeds
CONFIG_FILE_PATH = Path("config.json")

if not CONFIG_FILE_PATH.is_file():
    print(f"❌ CRITICAL: Configuration file not found at {CONFIG_FILE_PATH}")
    exit(1)

with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

# --- Logging Setup ---
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

# Apply Random Seed
SEED = config['general_settings']['random_seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
logger.info(f"🌱 Random seed set to: {SEED}")


# --- Helper Functions (Modified to use logger) ---
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
    return random_split(dataset, [train_size, val_size, test_size])


def load_transformer_body_weights(target_model: FlowFineTuningModel, pretrained_checkpoint_path: str, device: str,
                                  logger_instance=None):
    if logger_instance is None:
        logger_instance = logger
    logger_instance.info(f"🔄 Loading Transformer Body weights from: {pretrained_checkpoint_path}")
    try:
        pretrained_state_dict = torch.load(pretrained_checkpoint_path, map_location=device)
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
        # Filter out missing keys that are NOT part of the target_body_prefix (e.g. classifier of FlowFineTuningModel)
        genuinely_missing_in_body = [k for k in missing if k.startswith(TARGET_BODY_PREFIX)]
        if genuinely_missing_in_body:
            logger_instance.warning(
                f"   ⚠️ Missing in target transformer_encoder_body: {len(genuinely_missing_in_body)} keys (e.g., {genuinely_missing_in_body[:3]}...). This might be an issue.")
        # Log other missing keys if necessary, but they are expected (new flow layers, new classifier)
        # logger.info(f"   ℹ️ Other missing keys (expected for new layers): {[k for k in missing if not k.startswith(TARGET_BODY_PREFIX)][:3]}...")
    if unexpected:
        logger_instance.error(
            f"   ❌ Unexpected keys in source checkpoint (should be 0 if loading only body): {len(unexpected)} (e.g., {unexpected[:3]}...).")
        return False  # This typically indicates an issue with the source checkpoint or prefixes
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
    logger_instance.info(
        f"🧊 Frozen {frz_c} parameter groups ({total_params_body:,} params) in '{TARGET_BODY_PREFIX}'. Other layers are trainable.")
    if frz_c == 0 and sum(
            p.numel() for p in model.parameters() if p.requires_grad and name.startswith(TARGET_BODY_PREFIX)) > 0:
        logger_instance.warning(
            f"   ⚠️ No parameters were actually frozen with prefix '{TARGET_BODY_PREFIX}', but body has params. Check model structure or prefix.")
    elif total_params_body == 0:
        logger_instance.warning(f"   ⚠️ Transformer body '{TARGET_BODY_PREFIX}' seems to have no parameters to freeze.")


# --- Configuration & Parameters from Loaded Config ---
logger.info(f"--- Initializing Fine-tuning Script (Using Central Config) ---")

# General Settings
DEVICE_str = config['general_settings']['device']
if DEVICE_str == "cuda" and not torch.cuda.is_available():
    logger.warning("⚠️ CUDA specified in config but not available. Falling back to CPU.")
    DEVICE = "cpu"
else:
    DEVICE = DEVICE_str
NUM_WORKERS_LOADER = config['general_settings']['num_workers_loader']

# Paths
paths_cfg = config['paths']
FLOW_DATASET_PATH = Path(config['dataset_paths']['processed_flow_pt'])
# Construct path to the best pre-trained packet model
PRETRAINED_PACKET_MODEL_WEIGHTS = Path(paths_cfg['packet_model_save_dir']) / "model_best.pt"
FINETUNED_MODEL_SAVE_DIR = Path(paths_cfg['finetuned_model_save_dir'])
FINETUNED_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Device: {DEVICE}, Num Workers: {NUM_WORKERS_LOADER}")
logger.info(f"Flow Dataset: {FLOW_DATASET_PATH}")
logger.info(f"Pretrained Packet Model Weights: {PRETRAINED_PACKET_MODEL_WEIGHTS}")
logger.info(f"Fine-tuned Model Save Dir: {FINETUNED_MODEL_SAVE_DIR}")
logger.info(f"Log file: {LOG_FILE_PATH}")

# Model Architecture Parameters (from central config)
model_arch_cfg = config['model_architecture']
transformer_body_cfg = model_arch_cfg['transformer_body']
D_MODEL_PRETRAINED = transformer_body_cfg['d_model']
NUM_LAYERS_PRETRAINED = transformer_body_cfg['num_layers']
NUM_HEADS_PRETRAINED = transformer_body_cfg['num_heads']
DROPOUT_PRETRAINED = transformer_body_cfg['dropout']
CLASSIFIER_DROPOUT_CFG = model_arch_cfg.get('classifier_dropout_flow',
                                            DROPOUT_PRETRAINED)  # Flow specific classifier dropout

# Packet model specific architecture details needed for reconstructing IoTTransformer
INPUT_DIM_NUMERICAL_PACKET = model_arch_cfg.get('input_dim_packet_numerical', len(numerical_columns_packets))
MAX_SEQ_LEN_PACKET = model_arch_cfg['max_seq_len_packet']
NUM_CLASSES_PACKET_PRETRAIN = model_arch_cfg['num_classes_packet']  # Num classes of pretrain task

# Flow specific parameters
NUM_CLASSES_FLOW = model_arch_cfg['num_classes_flow']

# Fine-tuning Hyperparameters
finetune_params_cfg = config['training_params']['fine_tuning_flow']
EPOCHS_FINETUNE = finetune_params_cfg['epochs']
LR_FINETUNE = finetune_params_cfg['lr']
BATCH_SIZE_FLOW = finetune_params_cfg['batch_size']
CLIP_GRAD_CFG = finetune_params_cfg.get('clip_grad', 1.0)
USE_WEIGHTED_SAMPLER_CFG = finetune_params_cfg.get('use_weighted_sampler', False)

logger.info(
    f"Pretrained Transformer Body Params: d_model={D_MODEL_PRETRAINED}, layers={NUM_LAYERS_PRETRAINED}, heads={NUM_HEADS_PRETRAINED}")
logger.info(f"Fine-tuning Params: Epochs={EPOCHS_FINETUNE}, LR={LR_FINETUNE}, BatchSize={BATCH_SIZE_FLOW}")

# --- 1. Load Flow Dataset and its characteristics ---
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
                flow_cat_cardinalities.append(1)  # Default for empty categorical column
    logger.info(
        f"   Flow data: NumNumerical={num_flow_numerical_f}, NumCategorical={num_flow_categorical_f}, CatCardinalities={flow_cat_cardinalities}")
    del flow_data_pt
except Exception as e:
    logger.error(f"❌ Error loading flow data characteristics: {e}", exc_info=True)
    exit(1)

full_flow_dataset = IoTFlowDataset(pt_file_path=FLOW_DATASET_PATH)

# --- 2. Prepare the FlowFineTuningModel ---
logger.info("2. Preparing model for fine-tuning...")
try:
    # Step 2a: Instantiate original packet model structure to extract its encoder body
    # This needs the categorical sizes for packet data
    cat_map_packet_path = config.get('paths', {}).get('category_mappings_packet', 'category_mappings_packets.json')
    logger.info(f"   Loading packet category mappings from: {cat_map_packet_path} for temp IoTTransformer")
    cat_map_packets = load_mappings(path=cat_map_packet_path, is_flow=False)
    cat_sizes_packets = {col: len(cat_map_packets[col]) for col in categorical_columns_packets if
                         col in cat_map_packets}
    cat_pad_packets = {c: cat_map_packets[c]["unknown"] for c in categorical_columns_packets if
                       c in cat_map_packets and "unknown" in cat_map_packets[c]}
    # Ensure all necessary packet categorical columns are present
    for col in categorical_columns_packets:
        if col not in cat_sizes_packets:
            logger.warning(
                f"Packet category '{col}' not in mappings. Defaulting size to 1, padding_idx to 0 for temp IoTTransformer.")
            cat_sizes_packets[col] = 1
            cat_pad_packets[col] = 0
        elif "unknown" not in cat_map_packets[col] and col in cat_pad_packets:  # Check if unknown was added
            logger.warning(
                f"Unknown category not found for packet col '{col}'. Defaulting padding_idx to 0 for temp IoTTransformer.")
            cat_pad_packets[col] = 0

    temp_packet_model_args = {
        'input_dim': INPUT_DIM_NUMERICAL_PACKET,  # Num numerical features for packets
        'cat_sizes': cat_sizes_packets,  # Cat sizes for packet features
        'cat_padding_idx': cat_pad_packets,  # Cat padding for packet features
        'embed_dim': D_MODEL_PRETRAINED,
        'num_heads': NUM_HEADS_PRETRAINED,
        'num_layers': NUM_LAYERS_PRETRAINED,
        'dropout': DROPOUT_PRETRAINED,
        'num_classes': NUM_CLASSES_PACKET_PRETRAIN,  # From pretrain task
        'max_seq_len': MAX_SEQ_LEN_PACKET
    }
    original_packet_model_for_body = PacketPretrainingModel(**temp_packet_model_args)

    if not hasattr(original_packet_model_for_body, 'transformer'):
        logger.error("Original IoTTransformer class needs 'transformer' attribute (the nn.TransformerEncoder).")
        raise AttributeError(
            "Original IoTTransformer class needs 'transformer' attribute for its nn.TransformerEncoder.")

    pretrained_transformer_body = original_packet_model_for_body.transformer
    logger.info("   ✅ Pretrained Transformer encoder body structure extracted from IoTTransformer.")

    # Step 2b: Instantiate the FlowFineTuningModel
    flow_finetuning_model_instance = FlowFineTuningModel(
        num_flow_numerical_features=num_flow_numerical_f,
        flow_cat_cardinalities=flow_cat_cardinalities,
        d_model=D_MODEL_PRETRAINED,  # This d_model is for flow feature projections AND the encoder
        pretrained_transformer_encoder_body=pretrained_transformer_body,
        num_classes=NUM_CLASSES_FLOW,
        classifier_dropout=CLASSIFIER_DROPOUT_CFG
    )
    flow_finetuning_model_instance.to(DEVICE)
    logger.info("   ✅ FlowFineTuningModel instantiated.")

    # Load pre-trained weights into the body and freeze it
    if not PRETRAINED_PACKET_MODEL_WEIGHTS.is_file():
        logger.error(
            f"❌ Pretrained packet model weights not found at {PRETRAINED_PACKET_MODEL_WEIGHTS}. Cannot proceed with fine-tuning.")
        exit(1)

    if not load_transformer_body_weights(flow_finetuning_model_instance, str(PRETRAINED_PACKET_MODEL_WEIGHTS), DEVICE,
                                         logger_instance=logger):
        logger.error("❌ Failed to load pre-trained weights into the transformer body.")
        exit(1)
    freeze_transformer_body(flow_finetuning_model_instance, logger_instance=logger)

except Exception as e:
    logger.error(f"❌ Error during model preparation: {e}", exc_info=True)
    exit(1)

# --- 3. Split flow dataset ---
logger.info("3. Splitting flow dataset...")
dataset_params_cfg = config.get('dataset_params', {})
val_ratio = dataset_params_cfg.get('val_ratio_flow', 0.1)  # Flow specific ratios
test_ratio = dataset_params_cfg.get('test_ratio_flow', 0.1)

train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(
    full_flow_dataset,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    logger_instance=logger
)

# --- 4. Fine-tune the model ---
logger.info("4. Starting Fine-tuning on Flow Data...")
try:
    fine_tune_flow_model(  # Assuming train_flows.py is updated for logger
        model=flow_finetuning_model_instance,
        train_dataset=train_flow_dataset,
        val_dataset=val_flow_dataset,
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE_FLOW,
        lr=LR_FINETUNE,
        device=DEVICE,
        save_dir=str(FINETUNED_MODEL_SAVE_DIR),
        clip_grad=CLIP_GRAD_CFG,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER_CFG,
        num_workers_loader=NUM_WORKERS_LOADER,
        logger=logger  # Pass the logger
    )
    logger.info("   ✅ Model fine-tuning completed.")
except Exception as e:
    logger.error(f"❌ Error during model fine-tuning: {e}", exc_info=True)
    exit(1)

# --- 5. Final Test ---
logger.info("5. Testing Fine-tuned Flow Model...")
try:
    best_model_path = FINETUNED_MODEL_SAVE_DIR / "model_flow_best.pt"
    if not best_model_path.exists():
        logger.warning(
            f"Best fine-tuned model checkpoint not found at {best_model_path}. Testing with current model state.")
        model_path_to_test = None
    else:
        model_path_to_test = str(best_model_path)

    test_flow_model(  # Assuming train_flows.py is updated for logger
        model=flow_finetuning_model_instance,
        test_dataset=test_flow_dataset,
        batch_size=BATCH_SIZE_FLOW,  # Or a specific test_batch_size from config
        device=DEVICE,
        model_path=model_path_to_test,
        logger=logger  # Pass the logger
    )
    logger.info("   ✅ Fine-tuned model testing completed.")
except Exception as e:
    logger.error(f"❌ Error during fine-tuned model testing: {e}", exc_info=True)
    exit(1)

logger.info("🏁 Fine-tuning script (with central config & logging) finished. 🏁")