# pipeline/iot_dataset.py
import torch
from torch.utils.data import Dataset

class IoTSequenceDataset(Dataset):
    def __init__(self, pt_file_path, max_seq_len=64):
        data = torch.load(pt_file_path)

        self.packet_seqs = data["packet_seq"]         # shape: [N, T, F]
        self.labels = data["label"]                   # shape: [N]
        self.attention_masks = data["attention_mask"] # shape: [N, T]
        self.max_seq_len = max_seq_len

        # Ensure all sequences are of the same length
        if self.packet_seqs.shape[1] != max_seq_len:
            raise ValueError(f"Expected sequence length {max_seq_len}, but got {self.packet_seqs.shape[1]}.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "packet_seq": self.packet_seqs[idx],        # [T, F]
            "label": self.labels[idx],
            "attention_mask": self.attention_masks[idx] # [T]
        }
