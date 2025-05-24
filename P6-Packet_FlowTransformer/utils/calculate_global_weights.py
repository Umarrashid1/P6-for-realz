import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pipeline.iot_packet_dataset import IoTPacketDataset # Assuming this is your dataset class
from pipeline.iot_flow_dataset import IoTFlowDataset   # Assuming this is your dataset class
from pathlib import Path
import json

CONFIG_FILE_PATH = Path("config.json")
with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

MAX_SEQ_LEN_PACKET_CONFIG = config['model_architecture']['max_seq_len_packet']
WEIGHTS_DIR = Path(config['dataset_paths']['raw_packet_dir']).parent / "weights"


# For Packet Data
#PACKET_DATA_PATH = Path(config['dataset_paths']['processed_packet_pt'])
#packet_dataset = IoTPacketDataset(PACKET_DATA_PATH, max_seq_len=MAX_SEQ_LEN_PACKET_CONFIG)
#packet_labels = [packet_dataset[i]["label"].item() for i in range(len(packet_dataset))]
#packet_classes = np.unique(packet_labels)
#packet_weights_np = compute_class_weight("balanced", classes=packet_classes, y=packet_labels)
#packet_weights_pt = torch.tensor(packet_weights_np, dtype=torch.float32)
#torch.save(packet_weights_pt, WEIGHTS_DIR / "packet_class_weights.pt") # Save in weights folder
#print("Saved packet_class_weights.pt")
#print(f"Packet classes: {packet_classes}, weights: {packet_weights_pt.numpy()}")

# For Flow Data
FLOW_DATA_PATH = Path(config['dataset_paths']['processed_flow_pt'])
flow_dataset = IoTFlowDataset(FLOW_DATA_PATH)
flow_labels = [flow_dataset[i]["label"].item() for i in range(len(flow_dataset))] # This will still be slow here once!
flow_classes = np.unique(flow_labels)
flow_weights_np = compute_class_weight("balanced", classes=flow_classes, y=flow_labels)
flow_weights_pt = torch.tensor(flow_weights_np, dtype=torch.float32)
torch.save(flow_weights_pt, WEIGHTS_DIR / "flow_class_weights.pt") # Save in weights folder
print("Saved flow_class_weights.pt")
print(f"Flow classes: {flow_classes}, weights: {flow_weights_pt.numpy()}")