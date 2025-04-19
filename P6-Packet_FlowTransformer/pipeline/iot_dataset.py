import torch
from torch.utils.data import Dataset

class IoTSequenceDataset(Dataset):
    def __init__(self, pt_file_path, max_seq_len=64):
        data = torch.load(pt_file_path)

        self.packet_seqs = data["packet_seq"]  # shape: [N_flows, T, F]
        self.labels = data["label"]            # shape: [N_flows]
        self.max_seq_len = max_seq_len

        # Optional: pad/truncate to uniform length
        self.padded_seqs = []
        for seq in self.packet_seqs:
            if len(seq) < max_seq_len:
                pad_len = max_seq_len - len(seq)
                pad = torch.zeros(pad_len, seq.size(1))
                padded_seq = torch.cat([seq, pad], dim=0)
            else:
                padded_seq = seq[:max_seq_len]
            self.padded_seqs.append(padded_seq)

        self.padded_seqs = torch.stack(self.padded_seqs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "packet_seq": self.padded_seqs[idx],  # [T, F]
            "label": self.labels[idx]
        }
