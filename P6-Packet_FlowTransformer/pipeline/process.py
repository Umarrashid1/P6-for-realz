import os
import pandas as pd
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from .config import categorical_columns, numerical_columns, LABEL_MAPPING


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_label_from_path(file_path):
    current_path = os.path.dirname(file_path)
    while current_path != os.path.dirname(current_path):  # Stop at root
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1


def process_file(file_path, rows_per_file, test_mode, missing_strategy, max_seq_len, global_min, global_max):
    label = find_label_from_path(file_path)
    if label == -1:
        print(f"[SKIP] No label for: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, nrows=rows_per_file if test_mode else None)
    except Exception as e:
        print(f"[ERROR] Couldn't read {file_path}: {e}")
        return None

    missing_cols = [col for col in numerical_columns + categorical_columns if col not in df.columns]
    required_flow_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
    if missing_cols or any(c not in df.columns for c in required_flow_cols):
        print(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
        return None

    if 'l4_tcp' in df.columns and 'l4_udp' in df.columns:
        df['protocol'] = np.where(df['l4_tcp'] == 1, 'TCP',
                                  np.where(df['l4_udp'] == 1, 'UDP', 'OTHER'))
    else:
        print(f"[SKIP] Missing 'l4_tcp' or 'l4_udp' in {file_path}")
        return None

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

    # Transfer numerical data to GPU for normalization
    for col in numerical_columns:
        min_val = global_min[col]
        max_val = global_max[col]
        df[col] = (df[col] - min_val) / (max_val - min_val + 1e-6)

    df[categorical_columns] = df[categorical_columns].astype("category").apply(lambda x: x.cat.codes)

    group_keys = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    flow_groups = df.groupby(group_keys, sort=False)

    local_packet_seqs, local_labels, local_attention_masks = [], [], []

    # Process each flow and transfer the data to GPU
    for _, flow_df in flow_groups:
        flow_features = flow_df[numerical_columns + categorical_columns].values
        flow_len = len(flow_features)

        if flow_len < max_seq_len:
            pkt_tensor = torch.tensor(flow_features, dtype=torch.float32).to(device)
            pad_len = max_seq_len - flow_len
            pad = torch.zeros(pad_len, pkt_tensor.shape[1]).to(device)
            attention_mask = torch.cat([torch.ones(flow_len), torch.zeros(pad_len)]).to(device)
            pkt_tensor = torch.cat([pkt_tensor, pad], dim=0)

            local_packet_seqs.append(pkt_tensor)
            local_labels.append(label)
            local_attention_masks.append(attention_mask)
        else:
            stride = max_seq_len // 2
            for start_idx in range(0, flow_len - max_seq_len + 1, stride):
                chunk = flow_features[start_idx:start_idx + max_seq_len]
                pkt_tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
                attention_mask = torch.ones(max_seq_len).to(device)
                local_packet_seqs.append(pkt_tensor)
                local_labels.append(label)
                local_attention_masks.append(attention_mask)

            remainder = (flow_len - max_seq_len) % stride
            if remainder != 0:
                final_chunk = flow_features[-max_seq_len:]
                pkt_tensor = torch.tensor(final_chunk, dtype=torch.float32).to(device)
                attention_mask = torch.ones(max_seq_len).to(device)
                local_packet_seqs.append(pkt_tensor)
                local_labels.append(label)
                local_attention_masks.append(attention_mask)

    return local_packet_seqs, local_labels, local_attention_masks


def preprocess_flows_as_sequences(dataset_dir, output_file, test_mode=False, rows_per_file=20000,
                                   missing_strategy="zero", max_seq_len=64):

    def get_all_csv_paths(root_dir):
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files if file.endswith(".csv")
        ]

    all_csv_files = get_all_csv_paths(dataset_dir)

    print("[INFO] Scanning files to compute global min/max for normalization...")
    global_min = {col: float('inf') for col in numerical_columns}
    global_max = {col: float('-inf') for col in numerical_columns}

    # Scanning for min/max on GPU
    for file_path in all_csv_files:
        try:
            df = pd.read_csv(file_path, nrows=rows_per_file if test_mode else None)
            for col in numerical_columns:
                if col in df.columns:
                    global_min[col] = min(global_min[col], df[col].min(skipna=True))
                    global_max[col] = max(global_max[col], df[col].max(skipna=True))
        except Exception as e:
            print(f"[WARN] Skipping {file_path} during normalization scan: {e}")

    print(f"[INFO] Starting multiprocessing with {cpu_count()} workers...")

    worker_func = partial(
        process_file,
        rows_per_file=rows_per_file,
        test_mode=test_mode,
        missing_strategy=missing_strategy,
        max_seq_len=max_seq_len,
        global_min=global_min,
        global_max=global_max
    )

    with Pool(cpu_count()) as pool:
        results = pool.map(worker_func, all_csv_files)

    all_packet_seqs, all_labels, attention_masks = [], [], []
    for result in results:
        if result:
            pkt_seqs, lbls, masks = result
            all_packet_seqs.extend(pkt_seqs)
            all_labels.extend(lbls)
            attention_masks.extend(masks)

    if not all_packet_seqs:
        raise RuntimeError("No flows found.")

    # Convert final results to tensors and move them to the appropriate device (GPU or CPU)
    packet_tensor = torch.stack(all_packet_seqs).to(device)
    label_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
    attention_mask_tensor = torch.stack(attention_masks).to(device)

    torch.save({
        "packet_seq": packet_tensor,
        "label": label_tensor,
        "attention_mask": attention_mask_tensor
    }, output_file)

    print(f"\n[INFO] Preprocessing complete — saved {len(packet_tensor)} flows to {output_file}")
    print(f"[INFO] Shape: packets {packet_tensor.shape}, labels {label_tensor.shape}")
