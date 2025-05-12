import os
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Assuming these imports exist and are correct
# from pipeline.iot_dataset import IoTDataset # You'll use this
# from models.transformer import IoTTransformer # Definition of original packet transformer
# from train.train import train_model, test_model # train_model needs slight modification for optimizer
# import pipeline.config as config # For flow numerical/categorical column names

# --- Placeholder for your actual imports ---
# Make sure these paths are correct for your project structure
# This is a common way to handle relative imports if 'pipeline' is a sibling to 'scripts' or similar
import sys
# Add the parent directory of 'pipeline' to sys.path if needed
# Example: sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.iot_dataset import IoTDataset
from models.transformer import IoTTransformer  # Original packet transformer definition
from train.train import train_model, test_model  # train_model needs slight modification for optimizer
import pipeline.config as config

# --- Configuration ---
FLOW_DATASET_PATH = "../../dataset/mini_flow.pt"  # Your "pure flow" dataset
PRETRAINED_PACKET_MODEL_PATH = "iot_transformer_pretrained_small.pt"  # Your packet model checkpoint

NUM_CLASSES = 8  # From your script; number of output classes for the flow task
D_MODEL = 128  # **** IMPORTANT: Set this to the d_model of your PRETRAINED packet transformer ****
NUM_LAYERS_PRETRAINED = 2  # **** IMPORTANT: Set this to num_layers of PRETRAINED packet transformer ****
N_HEAD_PRETRAINED = 4  # **** IMPORTANT: Set this to nhead of PRETRAINED packet transformer ****
# Add any other essential architectural parameters of the original IoTTransformer if they affect the body structure

VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE_FINETUNE = 1e-4  # Fine-tuning LR, typically smaller
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR_FINETUNED = "checkpoints_finetuned_pure_flows"


# --- Model Adaptation: FlowFineTuningModel ---
class FlowFineTuningModel(nn.Module):
    """
    Wrapper model for fine-tuning. Uses the core transformer blocks
    from a pre-trained packet model but adds new input layers for flows
    and a new classification head.
    """

    def __init__(self,
                 pretrained_transformer_body: nn.Module,  # The nn.TransformerEncoder (or equivalent)
                 d_model: int,  # d_model of the transformer body
                 num_flow_numerical_features: int,
                 flow_categorical_feature_cardinalities: List[int],
                 num_classes: int):
        super().__init__()
        self.d_model = d_model

        # --- New Input Layers for Flow Features ---
        # These layers will be trained from scratch.

        # 1. Numerical Flow Features Projection
        # Project numerical flow features to a part of d_model
        # Example: project to d_model if no categorical, or d_model // 2 if combined
        self.flow_numerical_projection = nn.Linear(num_flow_numerical_features, d_model)
        # If combining with categorical, adjust projection size, e.g.:
        # self.flow_numerical_projection = nn.Linear(num_flow_numerical_features, d_model // 2 if flow_categorical_feature_cardinalities else d_model)

        # 2. Categorical Flow Features Embeddings
        self.flow_categorical_embeddings = nn.ModuleList()
        total_cat_embedding_dim = 0
        if flow_categorical_feature_cardinalities:
            # Simple approach: Embed each flow categorical feature and sum/concatenate
            # For simplicity, let's assume we project numerical to d_model and don't use flow categoricals for now,
            # or that the original IoTTransformer's input handling is more complex.
            # This part needs to align with how you want to feed flow features to the transformer body.
            # A common way: concatenate numerical projection and sum of categorical embeddings.
            # For this example, let's assume the flow data's "categorical" tensor is processed
            # by a set of new embedding layers.
            # Each embedding could be, for example, d_model / num_cat_features or a fixed small dim.
            # Let's make a simple choice: sum embeddings if they exist, and numerical projection handles d_model.
            # This part is highly dependent on the design of IoTTransformer's input.
            # For now, let's assume combined_input_to_transformer = self.flow_numerical_projection(...)
            # If you have flow categorical features, you'd add:
            # cat_embedding_dim_example = d_model // 2 # if num_proj is also d_model // 2
            # for card in flow_categorical_feature_cardinalities:
            #    self.flow_categorical_embeddings.append(nn.Embedding(card, cat_embedding_dim_example))
            # total_cat_embedding_dim = cat_embedding_dim_example * len(flow_categorical_feature_cardinalities) # if concatenating
            # Or just cat_embedding_dim_example if summing or averaging.
            pass  # Placeholder for flow categorical embedding logic if needed

        # --- Pre-trained Transformer Body ---
        # This is the nn.TransformerEncoder (or equivalent stack of layers)
        # from your original IoTTransformer.
        self.transformer_encoder_body = pretrained_transformer_body

        # --- New Classification Head ---
        # Takes the output of the transformer body (d_model) and maps to num_classes
        self.flow_classification_head = nn.Linear(d_model, num_classes)

        print("\n--- FlowFineTuningModel Architecture Initialized ---")
        print(f"  Flow Numerical Input Dim: {num_flow_numerical_features}")
        print(f"  Flow Categorical Cardinalities: {flow_categorical_feature_cardinalities}")
        print(f"  d_model (Transformer Body): {d_model}")
        print(f"  Output Classes: {num_classes}")
        print("-" * 50)

    def forward(self, flow_numerical_data: torch.Tensor, flow_categorical_data: Optional[torch.Tensor] = None):
        """
        Forward pass for flow data.
        Args:
            flow_numerical_data (torch.Tensor): Shape [batch_size, num_flow_numerical_features]
            flow_categorical_data (torch.Tensor, optional): Shape [batch_size, num_flow_categorical_features]
                                                            Each column contains integer codes.
        Returns:
            torch.Tensor: Logits, shape [batch_size, num_classes]
        """
        # 1. Process flow numerical features
        # Shape: [batch_size, d_model]
        projected_numerical_flow_features = self.flow_numerical_projection(flow_numerical_data)

        # --- Placeholder for processing flow_categorical_data ---
        # if flow_categorical_data is not None and len(self.flow_categorical_embeddings) > 0:
        #     cat_embeddings = []
        #     for i, embedding_layer in enumerate(self.flow_categorical_embeddings):
        #         cat_embeddings.append(embedding_layer(flow_categorical_data[:, i]))
        #     # Combine categorical embeddings (e.g., sum or concatenate)
        #     # combined_cat_embeddings = torch.sum(torch.stack(cat_embeddings), dim=0)
        #     # Then combine with numerical:
        #     # transformer_input_features = torch.cat([projected_numerical_flow_features, combined_cat_embeddings], dim=1)
        #     # Ensure the concatenated dimension matches d_model or adjust projections.
        # else:
        transformer_input_features = projected_numerical_flow_features  # Assuming only numerical for now or combined elsewhere

        # 2. Reshape for Transformer (as a sequence of length 1)
        # Shape: [batch_size, 1, d_model]
        transformer_sequence_input = transformer_input_features.unsqueeze(1)

        # 3. Pass through the pre-trained transformer body
        # The transformer might need a mask, even for length 1.
        # Depending on the implementation, it might handle None or require a dummy mask.
        # src_key_padding_mask (if used by your transformer_encoder_body):
        #   torch.zeros(batch_size, 1, dtype=torch.bool, device=transformer_sequence_input.device)
        # **** ADJUST MASKING BASED ON YOUR IoTTransformer's ENCODER BODY'S EXPECTATIONS ****
        transformer_output = self.transformer_encoder_body(transformer_sequence_input)  # Add mask if needed
        # Output shape likely [batch_size, 1, d_model]

        # 4. Get representation for classification (output of the single 'token')
        # Shape: [batch_size, d_model]
        flow_representation = transformer_output.squeeze(1)

        # 5. Pass through the new classification head
        logits = self.flow_classification_head(flow_representation)

        return logits


# --- Helper Functions ---

def split_dataset_three_ways(dataset, val_ratio=0.1, test_ratio=0.1):
    """Splits a dataset into train, validation, and test sets."""
    # (Your existing robust split function)
    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty, cannot split.")
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)

    # Ensure val_size and test_size are at least 1 if dataset is small but not empty,
    # and train_size is also at least 1 if possible.
    if total_size < 3:  # Handle very small datasets
        print("Warning: Dataset too small for standard train/val/test split. Adjusting...")
        train_size = max(0, total_size - 2)  # try to get at least 1 for val and test if total >=2
        val_size = 1 if total_size >= 2 else 0
        test_size = 1 if total_size >= 1 and val_size == 1 else (1 if total_size >= 1 and val_size == 0 else 0)
        train_size = total_size - val_size - test_size  # Recalculate train
    else:
        train_size = total_size - val_size - test_size
        if train_size < 1:  # Ensure train_size is not zero if val+test is too large
            train_size = 1
            val_size = int((total_size - train_size) * (val_ratio / (val_ratio + test_ratio + 1e-9)))
            test_size = total_size - train_size - val_size

    if train_size + val_size + test_size != total_size:  # Final adjustment due to rounding
        train_size = total_size - val_size - test_size

    print(f"Splitting dataset: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")
    if train_size <= 0 or val_size <= 0 or test_size <= 0:  # Check if any split is zero
        print("Warning: One of the data splits is zero or negative. Check ratios and total dataset size.")
        # Fallback if still problematic: give most to train
        if total_size > 2:
            train_size = total_size - 2
            val_size = 1
            test_size = 1
            print(f"Adjusted split: Train={train_size}, Val={val_size}, Test={test_size}")

    return random_split(dataset, [train_size, val_size, test_size])


def load_transformer_body_weights(target_model: FlowFineTuningModel, pretrained_checkpoint_path: str):
    """
    Loads weights from the 'transformer_encoder_body' (or equivalent) of a pre-trained
    IoTTransformer checkpoint into the 'transformer_encoder_body' of the FlowFineTuningModel.
    """
    print(f"\n🔄 Loading Transformer Body weights from: {pretrained_checkpoint_path}")
    try:
        pretrained_state_dict = torch.load(pretrained_checkpoint_path, map_location=DEVICE)
    except FileNotFoundError:
        print(f"❌ Error: Pretrained model file not found at {pretrained_checkpoint_path}")
        return False
    except Exception as e:
        print(f"❌ Error loading pretrained model file: {e}")
        return False

    # --- Identify layers belonging to the transformer body ---
    # **** This prefix MUST MATCH the attribute name of the transformer body ****
    # **** in BOTH the original IoTTransformer AND FlowFineTuningModel ****
    # **** Common names: 'transformer_encoder', 'encoder_layers', 'body', etc. ****
    # **** In your IoTTransformer, it seems to be 'self.transformer_encoder' based on previous discussions. ****
    # **** So, in FlowFineTuningModel, we named it 'self.transformer_encoder_body' to load into. ****
    PRETRAINED_BODY_PREFIX = "transformer_encoder."  # Prefix in the SAVED checkpoint
    TARGET_BODY_PREFIX = "transformer_encoder_body."  # Prefix in the FlowFineTuningModel

    transformer_body_weights_to_load = {}
    loaded_count = 0
    skipped_count = 0

    for k_pretrained, v_pretrained in pretrained_state_dict.items():
        if k_pretrained.startswith(PRETRAINED_BODY_PREFIX):
            # Construct the corresponding key name in the target model
            k_target = TARGET_BODY_PREFIX + k_pretrained[len(PRETRAINED_BODY_PREFIX):]

            if k_target in target_model.state_dict():
                if target_model.state_dict()[k_target].shape == v_pretrained.shape:
                    transformer_body_weights_to_load[k_target] = v_pretrained
                    loaded_count += 1
                else:
                    print(
                        f"   ⚠️ Skipped (shape mismatch): {k_pretrained} (Pretrained: {v_pretrained.shape}) vs {k_target} (Target: {target_model.state_dict()[k_target].shape})")
                    skipped_count += 1
            else:
                print(f"   ⚠️ Skipped (target key not found): {k_pretrained} (Target key would be: {k_target})")
                skipped_count += 1
        # else: # Skipping other layers (embeddings, classification head of packet model)
        # print(f"   Skipping non-body layer from checkpoint: {k_pretrained}")

    if not transformer_body_weights_to_load:
        print(f"❌ Error: No weights matched for the transformer body.")
        print(
            f"   Checked for keys starting with '{PRETRAINED_BODY_PREFIX}' in checkpoint and map to '{TARGET_BODY_PREFIX}' in target model.")
        print(f"   Available keys in pretrained model (sample): {list(pretrained_state_dict.keys())[:5]}...")
        print(f"   Available keys in target model (sample): {list(target_model.state_dict().keys())[:5]}...")
        return False

    # Load the filtered state dict into the target model
    # `strict=False` is important as we are only loading a part of the model
    incompatible_keys = target_model.load_state_dict(transformer_body_weights_to_load, strict=False)

    print(f"\n✅ Loaded {loaded_count} parameter tensors into '{TARGET_BODY_PREFIX}'.")
    if skipped_count > 0:
        print(
            f"   Skipped {skipped_count} parameter tensors from checkpoint due to mismatch or not being part of the body.")

    if incompatible_keys.missing_keys:
        # These are keys in FlowFineTuningModel that were NOT in transformer_body_weights_to_load
        # This is expected for the new input layers and new classification head.
        print(
            f"   ℹ️ Note: {len(incompatible_keys.missing_keys)} layers in the FlowFineTuningModel were not loaded (expected for new layers):")
        # for key in incompatible_keys.missing_keys[:5]: print(f"     - {key}")
    if incompatible_keys.unexpected_keys:
        # These are keys from transformer_body_weights_to_load that were NOT in FlowFineTuningModel's state_dict
        # This should ideally be empty if prefixes and structure match.
        print(
            f"   ❌ Error: {len(incompatible_keys.unexpected_keys)} loaded keys were not expected by the model. Check model structure and prefixes.")
        # for key in incompatible_keys.unexpected_keys[:5]: print(f"     - {key}")
        return False
    return True


def freeze_transformer_body(model: FlowFineTuningModel):
    """Freezes the parameters of the transformer_encoder_body part of the model."""
    # **** This MUST MATCH the attribute name in FlowFineTuningModel ****
    TARGET_BODY_PREFIX = "transformer_encoder_body."
    frozen_count = 0
    total_params = 0
    print(f"\n🧊 Freezing layers with prefix '{TARGET_BODY_PREFIX}'...")
    for name, param in model.named_parameters():
        total_params += 1
        if name.startswith(TARGET_BODY_PREFIX):
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True  # Ensure other layers (new input, new head) are trainable
            print(f"   ✅ Layer '{name}' is TRAINABLE.")
    print(f"   Frozen {frozen_count} parameter groups out of {total_params} in FlowFineTuningModel.")
    if frozen_count == 0 and total_params > 0:
        print(
            f"   ⚠️ Warning: No parameters matched the prefix '{TARGET_BODY_PREFIX}'. Check attribute name. All params might be trainable.")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    os.makedirs(SAVE_DIR_FINETUNED, exist_ok=True)

    # 1. Load Flow Dataset Characteristics
    print(f"\nLoading flow dataset for metadata: {FLOW_DATASET_PATH}")
    try:
        # Load the .pt file to get dimensions for the FlowFineTuningModel
        flow_data_pt = torch.load(FLOW_DATASET_PATH, map_location='cpu')
        if 'numerical' not in flow_data_pt or 'categorical' not in flow_data_pt:
            raise KeyError("Flow .pt file must contain 'numerical' and 'categorical' keys.")

        num_flow_numerical_features = flow_data_pt['numerical'].shape[1]
        # Assuming flow_data_pt['categorical'] is [num_samples, num_cat_features]
        # And cardinalities were saved or can be derived as in your original script
        # For simplicity, let's use the derivation from your script:
        flow_categorical_tensor = flow_data_pt['categorical']
        flow_cat_cardinalities = [int(torch.max(flow_categorical_tensor[:, i]) + 1) for i in
                                  range(flow_categorical_tensor.shape[1])]
        # If 'metadata' with 'categorical_cardinalities' exists in .pt, use that instead.

        print(
            f"Flow data characteristics: Num Numerical={num_flow_numerical_features}, Cat Cardinalities={flow_cat_cardinalities}")
    except FileNotFoundError:
        print(f"❌ Error: Flow data file not found at {FLOW_DATASET_PATH}")
        exit(1)
    except KeyError as e:
        print(f"❌ Error: Missing key {e} in flow data .pt file. Ensure it's correctly preprocessed.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading flow data for metadata: {e}")
        exit(1)

    # 2. Instantiate the Original Packet Model Structure (to get its body)
    # This is needed to correctly extract the 'transformer_encoder' part.
    # Ensure parameters like d_model, num_layers, nhead match the *saved* pre-trained model.
    print("\nInstantiating structure of the original pre-trained packet model...")
    try:
        # These parameters define the structure of the transformer body we want to extract.
        # They MUST match the parameters used when 'iot_transformer_pretrained_small.pt' was saved.
        original_packet_model_structure = IoTTransformer(
            num_numerical=config.num_numerical_cols_packet,  # Example: from a config for packet model
            cat_cardinalities=config.cat_cardinalities_packet,  # Example: from a config for packet model
            d_model=D_MODEL,
            num_classes=NUM_CLASSES,  # Dummy, not used for body
            num_layers=NUM_LAYERS_PRETRAINED,
            nhead=N_HEAD_PRETRAINED,
            # Add other args like dim_feedforward if your IoTTransformer requires them
            # dim_feedforward = D_MODEL * 4 # A common setting
        )
        # **** ADJUST 'transformer_encoder' IF THE ATTRIBUTE NAME IS DIFFERENT IN IoTTransformer ****
        if not hasattr(original_packet_model_structure, 'transformer_encoder'):
            raise AttributeError(
                "Original IoTTransformer class does not have 'transformer_encoder' attribute. Check attribute name for the transformer body.")
        pretrained_body_module = original_packet_model_structure.transformer_encoder
        print("   Successfully created structure and identified transformer body.")
    except AttributeError as e:
        print(f"❌ {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error instantiating original IoTTransformer structure: {e}")
        print(f"   Ensure D_MODEL, NUM_LAYERS_PRETRAINED, N_HEAD_PRETRAINED and other critical architectural")
        print(f"   parameters match the model saved in '{PRETRAINED_PACKET_MODEL_PATH}'.")
        print(
            f"   Also ensure 'config.num_numerical_cols_packet' and 'config.cat_cardinalities_packet' are defined if used.")
        exit(1)

    # 3. Instantiate the Fine-tuning Model
    fine_tuning_model = FlowFineTuningModel(
        pretrained_transformer_body=pretrained_body_module,
        d_model=D_MODEL,
        num_flow_numerical_features=num_flow_numerical_features,
        flow_categorical_feature_cardinalities=flow_cat_cardinalities,
        num_classes=NUM_CLASSES
    )
    fine_tuning_model.to(DEVICE)

    # 4. Load Pre-trained Weights into the Transformer Body
    if not load_transformer_body_weights(fine_tuning_model, PRETRAINED_PACKET_MODEL_PATH):
        print("❌ Critical error: Failed to load pre-trained weights. Exiting.")
        exit(1)

    # 5. Freeze the Transformer Body
    freeze_transformer_body(fine_tuning_model)

    # 6. Load Full Flow Dataset for Fine-tuning
    print(f"\nLoading full flow dataset for fine-tuning: {FLOW_DATASET_PATH}")
    # Your IoTDataset class should be able to load the .pt file directly
    # and return dicts with 'numerical', 'categorical', 'label' keys
    # matching what FlowFineTuningModel's forward pass expects.
    try:
        full_flow_dataset = IoTDataset(FLOW_DATASET_PATH,
                                       numerical_key='numerical',  # Key for numerical data in .pt
                                       categorical_key='categorical',  # Key for categorical data in .pt
                                       label_key='label')  # Key for labels in .pt
        if len(full_flow_dataset) == 0:
            raise ValueError("Flow dataset is empty after loading.")
        print(f"   Flow dataset loaded. Size: {len(full_flow_dataset)}")
    except Exception as e:
        print(f"❌ Error loading full flow dataset with IoTDataset class: {e}")
        exit(1)

    # 7. Split Dataset
    try:
        train_dataset, val_dataset, test_dataset = split_dataset_three_ways(
            full_flow_dataset, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
        )
        print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("❌ Error: Training or Validation dataset split is empty. Check dataset size and split ratios.")
            exit(1)
    except ValueError as e:
        print(f"❌ Error splitting dataset: {e}")
        exit(1)

    # 8. Fine-tune the Model
    # Ensure train_model's optimizer is like: optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    print("\n--- Starting Fine-tuning ---")
    # You might need to pass a flag to train_model or modify it to handle the optimizer correctly
    train_model(
        model=fine_tuning_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE_FINETUNE,
        device=DEVICE,
        save_dir=SAVE_DIR_FINETUNED,
        use_filtered_optimizer=True  # Add this flag or similar to your train_model
    )
    print("--- Fine-tuning Finished ---")

    # 9. Final Evaluation
    print("\n--- Starting Final Evaluation on Test Set ---")
    # Ensure test_model also correctly unpacks data from the flow dataset
    best_model_path = os.path.join(SAVE_DIR_FINETUNED, "model_best.pt")  # Assuming train_model saves this
    if not os.path.exists(best_model_path):
        print(
            f"Warning: Best model checkpoint not found at {best_model_path}. Testing with last epoch model if available.")
        # Fallback to last epoch model if train_model saves it with a known name
        # For now, test_model will use the model object in its current (last trained) state.

    test_model(
        model=fine_tuning_model,
        # Test the model object directly (it has the loaded best weights if train_model reloads)
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        device=DEVICE
        # model_path=best_model_path # Or load explicitly if train_model doesn't update the passed model object
    )
    print("--- Evaluation Finished ---")

