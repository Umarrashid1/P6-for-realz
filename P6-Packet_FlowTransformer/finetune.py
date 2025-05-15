import os
import torch
from torch.utils.data import random_split
import json  # For loading the configuration file
from pathlib import Path  # For path handling
from pipeline.iot_flowdataset import IoTFlowDataset  # Assuming this path is correct
from models.transformer import IoTTransformer  # Assuming this path is correct
from models.flow_finetuning_model import FlowFineTuningModel  # Assuming this path is correct
from train.train_flows import fine_tune_flow_model, test_flow_model  # Assuming this path is correct
import logging  # Import logging module
import sys  # For sys.exit

# --- Setup Logging ---
# Determine the base directory of the script (P6-Packet_FlowTransformer)
# This assumes the script is run from its location within P6-Packet_FlowTransformer
# or that the CWD is P6-Packet_FlowTransformer as in the SLURM script.
log_file_path = Path(os.getcwd()) / 'finetune_flow_model.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='w'  # Overwrite log file each time
)


# If you also want to see logs on console (stderr) during interactive runs, uncomment next line:
# logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))


# --- Helper Functions ---
def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])


def load_transformer_body_weights(target_model: FlowFineTuningModel, pretrained_checkpoint_path: str, device: str):
    logging.info(f"🔄 Loading Transformer Body weights from: {pretrained_checkpoint_path}")  # Changed from print
    try:
        pretrained_state_dict = torch.load(pretrained_checkpoint_path, map_location=device)
    except FileNotFoundError:
        logging.error(f"❌ Error: Pretrained model file not found: {pretrained_checkpoint_path}")  # Changed from print
        return False
    except Exception as e:  # Catch other potential torch.load errors
        logging.error(f"❌ Error loading pretrained model file {pretrained_checkpoint_path}: {e}")
        return False

    PRETRAINED_BODY_PREFIX = "transformer."  # In IoTTransformer state_dict
    TARGET_BODY_PREFIX = "transformer_encoder_body."  # In FlowFineTuningModel

    body_weights = {TARGET_BODY_PREFIX + k[len(PRETRAINED_BODY_PREFIX):]: v
                    for k, v in pretrained_state_dict.items() if k.startswith(PRETRAINED_BODY_PREFIX)}

    if not body_weights:
        logging.warning(
            f"⚠️ No weights found with prefix '{PRETRAINED_BODY_PREFIX}' in checkpoint {pretrained_checkpoint_path}.")  # Changed from print to warning
        # Depending on strictness, you might want to return False or allow continuation
        # For now, assume it's a warning and continue to see load_state_dict output
        # return False # Uncomment if this should be a fatal error

    missing, unexpected = target_model.load_state_dict(body_weights, strict=False)
    logging.info(f"✅ Loaded {len(body_weights)} layers into '{TARGET_BODY_PREFIX}'.")  # Changed from print
    if missing: logging.info(
        f"   ℹ️ Missing in target: {len(missing)} (expected for new layers, e.g.: {missing[:3]}...)")  # Changed from print
    if unexpected:
        logging.error(
            f"   ❌ Unexpected in source checkpoint: {len(unexpected)} (e.g.: {unexpected[:3]}...) Check prefixes and model structure.")  # Changed from print
        return False  # Treat unexpected weights in the source as an error
    return True


def freeze_transformer_body(model: FlowFineTuningModel):
    TARGET_BODY_PREFIX = "transformer_encoder_body."
    frz_c = 0
    total_params_in_body = 0
    for name, param in model.named_parameters():
        if name.startswith(TARGET_BODY_PREFIX):
            param.requires_grad = False
            frz_c += 1
            total_params_in_body += param.numel()
        else:
            param.requires_grad = True  # Ensure other parts are trainable

    if total_params_in_body > 0 and frz_c > 0:  # Check if the body actually had parameters
        logging.info(
            f"🧊 Frozen {frz_c} param groups in '{TARGET_BODY_PREFIX}'. Others trainable.")  # Changed from print
    elif total_params_in_body == 0:
        logging.warning(
            f"   ⚠️ No parameters found with prefix '{TARGET_BODY_PREFIX}'. Nothing was frozen. Check model structure.")
    else:  # frz_c == 0 but total_params_in_body > 0 (should not happen if prefix is correct)
        logging.warning(
            f"   ⚠️ No parameters were frozen with prefix '{TARGET_BODY_PREFIX}', though parameters exist there. All are trainable. Check freezing logic.")


# --- Configuration & Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths - Using Path objects for robustness
# Assuming the script is run from P6-Packet_FlowTransformer directory
BASE_DIR = Path(
    os.getcwd())  # Or explicitly Path(__file__).resolve().parent if script location is fixed relative to data
FLOW_DATASET_PATH = BASE_DIR / "../../../dataset/processed_flows.pt"  # Adjust if CWD is different
PACKET_MODEL_CHECKPOINT_DIR = BASE_DIR / "checkpoints_packet_model"
PRETRAINED_PACKET_MODEL_WEIGHTS = PACKET_MODEL_CHECKPOINT_DIR / "model_best.pt"
PACKET_MODEL_CONFIG_PATH = PACKET_MODEL_CHECKPOINT_DIR / "packet_config.json"
FINETUNED_MODEL_SAVE_DIR = BASE_DIR / "checkpoints_flow_finetuned_vMain"  # Will be created by train_flows

# Fine-tuning Hyperparameters
NUM_CLASSES_FLOW = 8
EPOCHS_FINETUNE = 10
LR_FINETUNE = 5e-5
BATCH_SIZE_FLOW = 64

logging.info("--- Fine-tuning Setup ---")  # Changed from print
logging.info(f"Device: {DEVICE}")  # Changed from print
logging.info(f"Flow Dataset: {FLOW_DATASET_PATH.resolve()}")  # Changed from print, resolve for absolute path
logging.info(f"Pretrained Packet Model Weights: {PRETRAINED_PACKET_MODEL_WEIGHTS.resolve()}")  # Changed from print
logging.info(f"Pretrained Packet Model Config: {PACKET_MODEL_CONFIG_PATH.resolve()}")  # Changed from print

# --- 1. Load Packet Model Configuration ---
logging.info(
    f"\n1. Loading original packet model configuration from: {PACKET_MODEL_CONFIG_PATH.resolve()}")  # Changed from print
try:
    with open(PACKET_MODEL_CONFIG_PATH, 'r') as f:
        packet_model_arch_params = json.load(f)
    logging.info("   ✅ Packet model configuration loaded.")  # Changed from print
    D_MODEL_PRETRAINED = packet_model_arch_params['embed_dim']
    NUM_LAYERS_PRETRAINED = packet_model_arch_params['num_layers']
    NUM_HEADS_PRETRAINED = packet_model_arch_params['num_heads']
    DROPOUT_PRETRAINED = packet_model_arch_params.get('dropout', 0.1)
except FileNotFoundError:
    logging.error(
        f"❌ Error: Packet model config file not found: {PACKET_MODEL_CONFIG_PATH.resolve()}")  # Changed from print
    sys.exit(1)  # Changed from exit(1)
except KeyError as e:
    logging.error(
        f"❌ Error: Missing key {e} in packet model config file {PACKET_MODEL_CONFIG_PATH.resolve()}.")  # Changed from print
    sys.exit(1)  # Changed from exit(1)
except Exception as e:  # Catch other potential JSON errors
    logging.error(f"❌ Error loading packet model config {PACKET_MODEL_CONFIG_PATH.resolve()}: {e}")
    sys.exit(1)

# --- 2. Load Flow Dataset and its characteristics ---
logging.info(f"\n2. Loading flow dataset characteristics from: {FLOW_DATASET_PATH.resolve()}")  # Changed from print
try:
    if not FLOW_DATASET_PATH.exists():
        logging.error(f"❌ Error: Flow data file not found: {FLOW_DATASET_PATH.resolve()}")
        sys.exit(1)
    flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
    num_flow_numerical_f = flow_data_pt['numerical_features'].shape[1]
    flow_cat_tensor = flow_data_pt['categorical_features']
    num_flow_categorical_f = flow_cat_tensor.shape[1]
    flow_cat_cardinalities = []
    if num_flow_categorical_f > 0:
        for i in range(num_flow_categorical_f):
            flow_cat_cardinalities.append(int(torch.max(flow_cat_tensor[:, i])) + 1)
    logging.info(
        f"   Flow data: NumNumerical={num_flow_numerical_f}, NumCategorical={num_flow_categorical_f}, CatCardinalities={flow_cat_cardinalities}")  # Changed from print
except KeyError as e:
    logging.error(
        f"❌ Error: Missing key {e} in flow data .pt file ({FLOW_DATASET_PATH.resolve()}).")  # Changed from print
    sys.exit(1)  # Changed from exit(1)
except Exception as e:  # Catch other potential torch.load or access errors
    logging.error(f"❌ Error loading flow data {FLOW_DATASET_PATH.resolve()}: {e}")
    sys.exit(1)

full_flow_dataset = IoTFlowDataset(pt_file_path=str(FLOW_DATASET_PATH))  # IoTFlowDataset might expect string path

# --- 3. Prepare the FlowFineTuningModel ---
logging.info("\n3. Preparing model for fine-tuning...")  # Changed from print
try:
    temp_packet_model_args = {
        'input_dim': packet_model_arch_params['input_dim'],
        'cat_sizes': packet_model_arch_params['cat_sizes'],
        'cat_padding_idx': packet_model_arch_params['cat_padding_idx'],
        'embed_dim': D_MODEL_PRETRAINED,
        'num_heads': NUM_HEADS_PRETRAINED,
        'num_layers': NUM_LAYERS_PRETRAINED,
        'dropout': DROPOUT_PRETRAINED,
        'num_classes': packet_model_arch_params['num_classes'],
        'max_seq_len': packet_model_arch_params['max_seq_len']
    }
    original_packet_model_for_body = IoTTransformer(**temp_packet_model_args)

    if not hasattr(original_packet_model_for_body, 'transformer'):
        logging.error(
            "Original IoTTransformer class needs 'transformer' attribute for its nn.TransformerEncoder.")  # Changed from print + raise
        sys.exit(1)  # Changed from raise
    pretrained_transformer_body = original_packet_model_for_body.transformer
    logging.info("   ✅ Transformer body extracted from packet model structure.")  # Changed from print
except Exception as e:
    logging.error(
        f"❌ Error instantiating temporary packet model from loaded config: {type(e).__name__} - {e}")  # Changed from print
    import traceback

    logging.error(traceback.format_exc())  # Log full traceback for this complex step
    sys.exit(1)  # Changed from exit(1)

flow_finetuning_model_instance = FlowFineTuningModel(
    num_flow_numerical_features=num_flow_numerical_f,
    flow_cat_cardinalities=flow_cat_cardinalities,
    d_model=D_MODEL_PRETRAINED,
    num_classes=NUM_CLASSES_FLOW,
    classifier_dropout=0.1,
    pretrained_transformer_encoder_body=pretrained_transformer_body
)
flow_finetuning_model_instance.to(DEVICE)
logging.info("   ✅ FlowFineTuningModel instantiated.")  # Changed from print

if not load_transformer_body_weights(flow_finetuning_model_instance, str(PRETRAINED_PACKET_MODEL_WEIGHTS), DEVICE):
    logging.error("Halting due to issues in loading transformer body weights.")
    sys.exit(1)  # Changed from exit(1)
freeze_transformer_body(flow_finetuning_model_instance)

# --- 4. Split flow dataset ---
logging.info("\n4. Splitting flow dataset...")  # Changed from print
try:
    train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(full_flow_dataset)
    if len(train_flow_dataset) == 0 or len(val_flow_dataset) == 0:
        logging.error("Training or Validation dataset is empty after split.")  # Changed from print + raise
        sys.exit(1)  # Changed from raise
    logging.info(
        f"   Train: {len(train_flow_dataset)}, Val: {len(val_flow_dataset)}, Test: {len(test_flow_dataset)}")  # Changed from print
except ValueError as e:
    logging.error(f"Error during dataset split: {e}")  # Changed from print
    sys.exit(1)  # Changed from exit(1)
except Exception as e:  # Catch any other split errors
    logging.error(f"Unexpected error during dataset split: {e}")
    sys.exit(1)

# --- 5. Fine-tune the model ---
logging.info("\n5. Starting Fine-tuning on Flow Data...")  # Changed from print
try:
    fine_tune_flow_model(
        model=flow_finetuning_model_instance,
        train_dataset=train_flow_dataset,
        val_dataset=val_flow_dataset,
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE_FLOW,
        lr=LR_FINETUNE,
        device=DEVICE,
        save_dir=str(FINETUNED_MODEL_SAVE_DIR),  # Ensure save_dir is a string
        clip_grad=1.0,
        use_weighted_sampler=False
    )
except Exception as e:
    logging.error(f"❌ Error during fine_tune_flow_model: {type(e).__name__} - {e}")
    import traceback

    logging.error(traceback.format_exc())
    sys.exit(1)

# --- 6. Final Test ---
logging.info("\n6. Testing Fine-tuned Flow Model...")  # Changed from print
best_model_path = FINETUNED_MODEL_SAVE_DIR / "model_flow_best.pt"  # Path object
try:
    test_flow_model(
        model=flow_finetuning_model_instance,
        test_dataset=test_flow_dataset,
        batch_size=BATCH_SIZE_FLOW,
        device=DEVICE,
        model_path=str(best_model_path) if best_model_path.exists() else None  # Ensure model_path is string or None
    )
except Exception as e:
    logging.error(f"❌ Error during test_flow_model: {type(e).__name__} - {e}")
    import traceback

    logging.error(traceback.format_exc())
    sys.exit(1)

logging.info("\n🏁 main.py fine-tuning script finished. 🏁")  # Changed from print