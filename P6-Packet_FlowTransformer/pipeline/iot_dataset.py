# pipeline/iot_dataset.py
"""PyTorch Dataset for pre‑batched IoT flow sequences.

Returns a dict compatible with the updated training loop:
    {
        'packet_seq':      FloatTensor [T, F_num],
        'attention_mask':  FloatTensor [T],
        'label':           LongTensor,
        'cat_<col1>':      LongTensor [T],
        'cat_<col2>':      LongTensor [T],
        ...
    }

Assumes `preprocess_flows_as_sequences()` saved one tensor per categorical column.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

from .config import categorical_columns_packets  # adjust import path if different

class IoTSequenceDataset(Dataset):
    """Loads the merged .pt file and yields item‑level tensors."""

    def __init__(self, pt_file_path: str | Path, max_seq_len: int = 64):
        data = torch.load(pt_file_path, map_location="cpu")

        # Base tensors
        self.packet_seqs     = data["packet_seq"]       # [N, T, F_num]
        self.labels          = data["label"]            # [N]
        self.attention_masks = data["attention_mask"]   # [N, T]
        self.cat_tensors     = {col: data[col] for col in categorical_columns_packets}

        self.max_seq_len = max_seq_len

        # Sanity checks ----------------------------------------------------
        if self.packet_seqs.shape[1] != max_seq_len:
            raise ValueError(
                f"Expected sequence length {max_seq_len}, "
                f"but got {self.packet_seqs.shape[1]}.")
        for col, tensor in self.cat_tensors.items():
            if tensor.shape[1] != max_seq_len:
                raise ValueError(
                    f"Categorical column '{col}' length mismatch: "
                    f"got {tensor.shape[1]} expected {max_seq_len}.")

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "packet_seq":     self.packet_seqs[idx],
            "attention_mask": self.attention_masks[idx],
            "label":          self.labels[idx],
        }
        # Attach categorical ids
        for col in categorical_columns_packets:
            item[col] = self.cat_tensors[col][idx]
        return item
