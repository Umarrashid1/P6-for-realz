# In a utility file (e.g., train_utils.py) or at the top of your Optuna scripts
import torch
import torch.nn as nn
import logging
from pathlib import Path
import json # For loading config if paths are dynamic

# --- Load Config to get base dataset path ---
# This assumes this utility file is in a place where it can find config.json
# or you pass MAIN_CONFIG to it.
# For simplicity, if this is directly in your Optuna scripts, MAIN_CONFIG is already available.
# If it's a separate utility, you might need to load MAIN_CONFIG here or pass parts of it.

# Assuming MAIN_CONFIG is accessible (as it is in your Optuna scripts)
# Or, if this is a standalone utility, you might load it:
CONFIG_FILE_PATH_UTIL = Path("config.json") # Adjust path as needed
with open(CONFIG_FILE_PATH_UTIL, 'r') as f:
    main_config_util = json.load(f)
WEIGHTS_DIR = Path(main_config_util['dataset_paths']['raw_packet_dir']).parent / "weights"



PACKET_WEIGHTS_PATH = WEIGHTS_DIR / "packet_class_weights.pt"
FLOW_WEIGHTS_PATH = WEIGHTS_DIR / "flow_class_weights.pt"

PRECOMPUTED_PACKET_WEIGHTS = None
PRECOMPUTED_FLOW_WEIGHTS = None

try:
    PRECOMPUTED_PACKET_WEIGHTS = torch.load(PACKET_WEIGHTS_PATH)
    print(f"Successfully loaded precomputed packet weights from {PACKET_WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"Warning: {PACKET_WEIGHTS_PATH} not found. Balanced loss for packets will be unweighted if this file is not generated first.")
except Exception as e:
    print(f"Warning: Could not load packet weights from {PACKET_WEIGHTS_PATH}: {e}")


try:
    PRECOMPUTED_FLOW_WEIGHTS = torch.load(FLOW_WEIGHTS_PATH)
    print(f"Successfully loaded precomputed flow weights from {FLOW_WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"Warning: {FLOW_WEIGHTS_PATH} not found. Balanced loss for flows will be unweighted if this file is not generated first.")
except Exception as e:
    print(f"Warning: Could not load flow weights from {FLOW_WEIGHTS_PATH}: {e}")


def get_balanced_loss(device, data_type: str, logger_instance=None) -> nn.CrossEntropyLoss:
    if logger_instance is None:
        # Fallback logger if none provided (e.g. for main training scripts that might not have specific trial loggers)
        logger_instance = logging.getLogger('get_balanced_loss_fallback')
        if not logger_instance.hasHandlers(): # Avoid adding handlers multiple times
            handler = logging.StreamHandler(sys.stdout) # Or logging.NullHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            logger_instance.setLevel(logging.INFO) # Or desired level
            logger_instance.propagate = False


    weights_to_use = None
    if data_type == "packet":
        weights_to_use = PRECOMPUTED_PACKET_WEIGHTS
    elif data_type == "flow":
        weights_to_use = PRECOMPUTED_FLOW_WEIGHTS

    if weights_to_use is not None:
        weights_t = weights_to_use.to(device)
        if torch.isfinite(weights_t).all():
            logger_instance.debug(f"Using precomputed class-balanced weights for {data_type}: {weights_t.cpu().numpy()}")
            return nn.CrossEntropyLoss(weight=weights_t)
        else:
            logger_instance.warning(f"Precomputed weights for {data_type} contain NaN/Inf. Using unweighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss()
    else:
        logger_instance.warning(f"Precomputed weights for {data_type} not available or not loaded. Using unweighted CrossEntropyLoss.")
        return nn.CrossEntropyLoss()