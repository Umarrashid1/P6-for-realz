import os
import torch
from torch.utils.data import random_split
import json  # For loading the configuration file
from pathlib import Path  # For path handling
from pipeline.iot_flowdataset import IoTFlowDataset
from models.transformer import IoTTransformer
from models.flow_finetuning_model import FlowFineTuningModel
from train.train_flows import fine_tune_flow_model, test_flow_model


# --- Helper Functions ---
def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])

def load_transformer_body_weights(target_model: FlowFineTuningModel, pretrained_checkpoint_path: str, device: str):
    print(f"🔄 Loading Transformer Body weights from: {pretrained_checkpoint_path}")
    try:
        pretrained_state_dict = torch.load(pretrained_checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"❌ Error: Pretrained model file not found: {pretrained_checkpoint_path}")
        return False
    PRETRAINED_BODY_PREFIX = "transformer_encoder."  # In IoTTransformer state_dict
    TARGET_BODY_PREFIX = "transformer_encoder_body."  # In FlowFineTuningModel
    body_weights = {TARGET_BODY_PREFIX + k[len(PRETRAINED_BODY_PREFIX):]: v
                    for k, v in pretrained_state_dict.items() if k.startswith(PRETRAINED_BODY_PREFIX)}
    if not body_weights:
        print(f"❌ No weights found with prefix '{PRETRAINED_BODY_PREFIX}' in checkpoint.")
        return False
    missing, unexpected = target_model.load_state_dict(body_weights, strict=False)
    print(f"✅ Loaded {len(body_weights)} layers into '{TARGET_BODY_PREFIX}'.")
    if missing: print(f"   ℹ️ Missing in target: {len(missing)} (expected for new layers: {missing[:3]}...)")
    if unexpected: print(
        f"   ❌ Unexpected in source: {len(unexpected)} (should be 0: {unexpected[:3]}...)"); return False
    return True


def freeze_transformer_body(model: FlowFineTuningModel):
    TARGET_BODY_PREFIX = "transformer_encoder_body."
    frz_c = 0
    for name, param in model.named_parameters():
        if name.startswith(TARGET_BODY_PREFIX):
            param.requires_grad = False; frz_c += 1
        else:
            param.requires_grad = True
    print(f"🧊 Frozen {frz_c} param groups in '{TARGET_BODY_PREFIX}'. Others trainable.")
    if frz_c == 0 and sum(p.numel() for p in model.parameters() if p.requires_grad) > 0:  # Check if model has params
        print(f"   ⚠️ No params frozen with prefix '{TARGET_BODY_PREFIX}'. All are trainable.")


# --- Configuration & Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
FLOW_DATASET_PATH = "../../../dataset/processed_flows.pt"  # Your preprocessed "pure flow" .pt file
PACKET_MODEL_CHECKPOINT_DIR = Path("checkpoints_packet_model")  # Where packet model & config were saved
PRETRAINED_PACKET_MODEL_WEIGHTS = PACKET_MODEL_CHECKPOINT_DIR / "model_best.pt"  # Or "model_best.pt"
PACKET_MODEL_CONFIG_PATH = PACKET_MODEL_CHECKPOINT_DIR / "packet_config.json"
FINETUNED_MODEL_SAVE_DIR = "checkpoints_flow_finetuned_vMain"

# Fine-tuning Hyperparameters
NUM_CLASSES_FLOW = 8
EPOCHS_FINETUNE = 10
LR_FINETUNE = 5e-5
BATCH_SIZE_FLOW = 64

print(f"--- Fine-tuning Setup ---")
print(f"Device: {DEVICE}")
print(f"Flow Dataset: {FLOW_DATASET_PATH}")
print(f"Pretrained Packet Model Weights: {PRETRAINED_PACKET_MODEL_WEIGHTS}")
print(f"Pretrained Packet Model Config: {PACKET_MODEL_CONFIG_PATH}")

# --- 1. Load Packet Model Configuration ---
print(f"\n1. Loading original packet model configuration from: {PACKET_MODEL_CONFIG_PATH}")
try:
    with open(PACKET_MODEL_CONFIG_PATH, 'r') as f:
        packet_model_arch_params = json.load(f)
    print(f"   ✅ Packet model configuration loaded.")
    # Extract key architectural params for the transformer body
    D_MODEL_PRETRAINED = packet_model_arch_params['embed_dim']
    NUM_LAYERS_PRETRAINED = packet_model_arch_params['num_layers']
    NUM_HEADS_PRETRAINED = packet_model_arch_params['num_heads']
    DROPOUT_PRETRAINED = packet_model_arch_params.get('dropout', 0.1)  # .get for backward compatibility
except FileNotFoundError:
    print(f"❌ Error: Packet model config file not found: {PACKET_MODEL_CONFIG_PATH}")
    exit(1)
except KeyError as e:
    print(f"❌ Error: Missing key {e} in packet model config file {PACKET_MODEL_CONFIG_PATH}.")
    exit(1)

# --- 2. Load Flow Dataset and its characteristics ---
print(f"\n2. Loading flow dataset characteristics from: {FLOW_DATASET_PATH}")
try:
    flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
    num_flow_numerical_f = flow_data_pt['numerical_features'].shape[1]
    flow_cat_tensor = flow_data_pt['categorical_features']
    num_flow_categorical_f = flow_cat_tensor.shape[1]
    flow_cat_cardinalities = []
    if num_flow_categorical_f > 0:
        for i in range(num_flow_categorical_f):
            flow_cat_cardinalities.append(int(torch.max(flow_cat_tensor[:, i])) + 1)
    print(
        f"   Flow data: NumNumerical={num_flow_numerical_f}, NumCategorical={num_flow_categorical_f}, CatCardinalities={flow_cat_cardinalities}")
except FileNotFoundError:  # ... (error handling)
    print(f"❌ Error: Flow data file not found: {FLOW_DATASET_PATH}")
    exit(1)
except KeyError as e:
    print(f"❌ Error: Missing key {e} in flow data .pt file.")
    exit(1)

full_flow_dataset = IoTFlowDataset(pt_file_path=FLOW_DATASET_PATH)

# --- 3. Prepare the FlowFineTuningModel ---
print("\n3. Preparing model for fine-tuning...")
# Instantiate original packet model structure using loaded config to get its body
try:
    # Use all params from the loaded config for IoTTransformer instantiation
    # Ensure your IoTTransformer __init__ matches these keys
    temp_packet_model_args = {
        'input_dim': packet_model_arch_params['input_dim'],
        'cat_sizes': packet_model_arch_params['cat_sizes'],
        'cat_padding_idx': packet_model_arch_params['cat_padding_idx'],
        'embed_dim': D_MODEL_PRETRAINED,  # from loaded config
        'num_heads': NUM_HEADS_PRETRAINED,  # from loaded config
        'num_layers': NUM_LAYERS_PRETRAINED,  # from loaded config
        'dropout': DROPOUT_PRETRAINED,  # from loaded config
        'num_classes': packet_model_arch_params['num_classes'],  # Original num_classes
        'max_seq_len': packet_model_arch_params['max_seq_len']
    }
    original_packet_model_for_body = IoTTransformer(**temp_packet_model_args)

    if not hasattr(original_packet_model_for_body, 'transformer'):
        raise AttributeError(
            "Original IoTTransformer class needs 'transformer' attribute for its nn.TransformerEncoder.")
    pretrained_transformer_body = original_packet_model_for_body.transformer
    print("   ✅ Transformer body extracted from packet model structure.")
except Exception as e:
    print(f"❌ Error instantiating temporary packet model from loaded config: {type(e).__name__} - {e}")
    exit(1)

# Instantiate the FlowFineTuningModel
flow_finetuning_model_instance = FlowFineTuningModel(
    num_flow_numerical_features=num_flow_numerical_f,
    flow_cat_cardinalities=flow_cat_cardinalities,
    d_model=D_MODEL_PRETRAINED,
    num_classes=NUM_CLASSES_FLOW,
    classifier_dropout=0.1,  # Example
    pretrained_transformer_encoder_body=pretrained_transformer_body
)
flow_finetuning_model_instance.to(DEVICE)
print("   ✅ FlowFineTuningModel instantiated.")

# Load pre-trained weights into the body and freeze it
if not load_transformer_body_weights(flow_finetuning_model_instance, PRETRAINED_PACKET_MODEL_WEIGHTS, DEVICE):
    exit(1)
freeze_transformer_body(flow_finetuning_model_instance)

# --- 4. Split flow dataset ---
print("\n4. Splitting flow dataset...")
try:
    train_flow_dataset, val_flow_dataset, test_flow_dataset = split_dataset_three_ways(full_flow_dataset)
    print(f"   Train: {len(train_flow_dataset)}, Val: {len(val_flow_dataset)}, Test: {len(test_flow_dataset)}")
    if len(train_flow_dataset) == 0 or len(val_flow_dataset) == 0:
        raise ValueError("Training or Validation dataset is empty after split.")
except ValueError as e:  # ... (error handling) ...
    print(f"Error during dataset split: {e}")
    exit(1)

# --- 5. Fine-tune the model ---
print("\n5. Starting Fine-tuning on Flow Data...")
fine_tune_flow_model(
    model=flow_finetuning_model_instance,
    train_dataset=train_flow_dataset,
    val_dataset=val_flow_dataset,
    epochs=EPOCHS_FINETUNE,
    batch_size=BATCH_SIZE_FLOW,
    lr=LR_FINETUNE,
    device=DEVICE,
    save_dir=FINETUNED_MODEL_SAVE_DIR,
    clip_grad=1.0,
    use_weighted_sampler=False
)

# --- 6. Final Test ---
print("\n6. Testing Fine-tuned Flow Model...")
best_model_path = Path(FINETUNED_MODEL_SAVE_DIR) / "model_flow_best.pt"
test_flow_model(
    model=flow_finetuning_model_instance,
    test_dataset=test_flow_dataset,
    batch_size=BATCH_SIZE_FLOW,
    device=DEVICE,
    model_path=str(best_model_path) if best_model_path.exists() else None
)

print("\n🏁 main.py fine-tuning script finished. 🏁")

