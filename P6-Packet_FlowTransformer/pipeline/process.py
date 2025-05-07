import os
import numpy as np
import pandas as pd
import torch
import pyarrow.dataset as ds
import pyarrow.csv as pv

from .config import categorical_columns, numerical_columns, LABEL_MAPPING

# ── LOAD GLOBAL NUMERIC STATS ─────────────────────────────────────────
STD_STATS_PATH = "standardization_stats.npz"
_std_stats = np.load(STD_STATS_PATH)
STD_COLS = _std_stats["cols"].tolist()
GLOBAL_MEAN = dict(zip(STD_COLS, _std_stats["mean"]))
GLOBAL_STD = dict(zip(STD_COLS, _std_stats["std"]))
EPS = 1e-6

def preprocess_flows_as_sequences(dataset_dir, output_file, test_mode=False, rows_per_file=20000, missing_strategy="zero", max_seq_len=64):
    all_packet_seqs = []
    all_labels = []
    attention_masks = []

    csv_format = ds.CsvFileFormat(read_options=pv.ReadOptions(autogenerate_column_names=False))
    dataset = ds.dataset(dataset_dir, format=csv_format)

    fragments = []
    for fragment in dataset.get_fragments():
        try:
            table = fragment.to_table()
            if test_mode and rows_per_file:
                table = table.slice(0, rows_per_file)
            df = table.to_pandas()
            path = fragment.path
            fragments.append((df, os.path.join(dataset_dir, path)))
        except Exception as e:
            print(f"[SKIP] Could not read {fragment.path}: {e}")

    for df, file_path in fragments:
        label = find_label_from_path(file_path)
        if label == -1:
            print(f"[SKIP] No label for: {file_path}")
            continue

        required_flow_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
        missing_cols = [col for col in numerical_columns + categorical_columns if col not in df.columns]
        if missing_cols or any(c not in df.columns for c in required_flow_cols):
            print(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
            continue

        df["protocol"] = np.select(
            [df["l4_tcp"].eq(1), df["l4_udp"].eq(1)],
            ["TCP", "UDP"],
            default="OTHER"
        )

        df = df[numerical_columns + categorical_columns + ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']].copy()

        if missing_strategy == "mean":
            for col in numerical_columns:
                df[col].fillna(df[col].mean(), inplace=True)
            for col in categorical_columns:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
        elif missing_strategy == "median":
            for col in numerical_columns:
                df[col].fillna(df[col].median(), inplace=True)
            for col in categorical_columns:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
        elif missing_strategy == "zero":
            df[numerical_columns] = df[numerical_columns].fillna(0)
            df[categorical_columns] = df[categorical_columns].fillna("unknown")
        elif missing_strategy == "ffill":
            df.fillna(method='ffill', inplace=True)
        else:
            raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

        for col in numerical_columns:
            df[col] = (df[col] - GLOBAL_MEAN[col]) / (GLOBAL_STD[col] + EPS)

        df[categorical_columns] = df[categorical_columns].astype("category").apply(lambda x: x.cat.codes)

        group_keys = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
        flow_groups = df.groupby(group_keys, sort=False)

        for _, flow_df in flow_groups:
            flow_features = flow_df[numerical_columns + categorical_columns].values
            flow_len = len(flow_features)

            print(f"\n[INFO] Processing flow from {file_path}")
            print(f"       → Flow length: {flow_len}")

            if flow_len < max_seq_len:
                print(f"       → Flow is shorter than max_seq_len ({max_seq_len}) — will pad.")
                pkt_tensor = torch.tensor(flow_features, dtype=torch.float32)
                attention_mask = torch.cat([torch.ones(flow_len), torch.zeros(max_seq_len - flow_len)])
                pad = torch.zeros(max_seq_len - flow_len, pkt_tensor.shape[1])
                pkt_tensor = torch.cat([pkt_tensor, pad], dim=0)

                all_packet_seqs.append(pkt_tensor)
                all_labels.append(label)
                attention_masks.append(attention_mask)

            else:
                stride = max_seq_len // 2
                num_chunks = (flow_len - max_seq_len) // stride + 1
                print(f"       → Flow is long enough. Using stride: {stride}")
                print(f"       → Splitting into {num_chunks} chunks of size {max_seq_len}")

                for i, start_idx in enumerate(range(0, flow_len - max_seq_len + 1, stride)):
                    end_idx = start_idx + max_seq_len
                    chunk = flow_features[start_idx:end_idx]
                    pkt_tensor = torch.tensor(chunk, dtype=torch.float32)
                    attention_mask = torch.ones(max_seq_len)

                    all_packet_seqs.append(pkt_tensor)
                    all_labels.append(label)
                    attention_masks.append(attention_mask)

                remainder = (flow_len - max_seq_len) % stride
                if remainder != 0:
                    final_chunk = flow_features[-max_seq_len:]
                    pkt_tensor = torch.tensor(final_chunk, dtype=torch.float32)
                    attention_mask = torch.ones(max_seq_len)
                    all_packet_seqs.append(pkt_tensor)
                    all_labels.append(label)
                    attention_masks.append(attention_mask)

    if not all_packet_seqs:
        raise RuntimeError("No flows found.")

    packet_tensor = torch.stack(all_packet_seqs)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    attention_mask_tensor = torch.stack(attention_masks)

    torch.save({
        "packet_seq": packet_tensor,
        "label": label_tensor,
        "attention_mask": attention_mask_tensor
    }, output_file)

    print(f"\n[INFO] Preprocessing complete — saved {len(packet_tensor)} flows to {output_file}")
    print(f"[INFO] Shape: packets {packet_tensor.shape}, labels {label_tensor.shape}")

def find_label_from_path(file_path):
    current_path = os.path.dirname(file_path)
    while current_path != os.path.dirname(current_path):
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1
