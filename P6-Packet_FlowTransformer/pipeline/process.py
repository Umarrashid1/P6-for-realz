# pipeline/process.py
import os
import pandas as pd
import numpy as np
import torch
from .config import categorical_columns, numerical_columns, LABEL_MAPPING

def preprocess_flows_as_sequences(dataset_dir, output_file, test_mode=False, rows_per_file=20000, missing_strategy="zero", max_seq_len=64):
    all_packet_seqs = []
    all_labels = []
    attention_masks = []


    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            label = find_label_from_path(file_path)
            if label == -1:
                print(f"[SKIP] No label for: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path, nrows=rows_per_file if test_mode else None)
            except Exception as e:
                print(f"[ERROR] Couldn't read {file_path}: {e}")
                continue

            missing_cols = [col for col in numerical_columns + categorical_columns if col not in df.columns]
            required_flow_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port']

            # Check for the presence of required columns and protocol
            if missing_cols or any(c not in df.columns for c in required_flow_cols):
                print(f"[SKIP] Missing columns in {file_path}: {missing_cols}")
                continue

            # Infer protocol based on l4_tcp / l4_udp
            if 'l4_tcp' in df.columns and 'l4_udp' in df.columns:
                def infer_protocol(row):
                    if row['l4_tcp'] == 1:
                        return 'TCP'
                    elif row['l4_udp'] == 1:
                        return 'UDP'
                    else:
                        return 'OTHER'

                # Add the 'protocol' column to the DataFrame
                df['protocol'] = df.apply(infer_protocol, axis=1)
            else:
                print(f"[SKIP] Missing 'l4_tcp' or 'l4_udp' in {file_path}")
                continue

            # Subset dataframe
            df = df[numerical_columns + categorical_columns + ['src_ip', 'dst_ip', 'src_port', 'dst_port',
                                                               'protocol']].copy()

            # Handle missing values
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

            # Normalize numerical
            df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].min()) / (
                df[numerical_columns].max() - df[numerical_columns].min() + 1e-6
            )

            # Encode categorical
            df[categorical_columns] = df[categorical_columns].astype("category").apply(lambda x: x.cat.codes)

            # Group packets into flows (5-tuple)
            group_keys = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
            flow_groups = df.groupby(group_keys)

            for _, flow_df in flow_groups:
                flow_features = flow_df[numerical_columns + categorical_columns].values
                flow_len = len(flow_features)

                # Sliding window to create overlapping chunks
                window_size = max_seq_len
                stride = window_size // 2  #
                num_chunks = (flow_len - window_size) // stride + 1

                print(f"[INFO] Processing flow from file: {file_path} with label: {label}")
                print(f"Flow had {flow_len} packets, split into {num_chunks} chunks using sliding window.")

                # Create chunks of packet sequences
                for i in range(num_chunks):
                    # Calculate start and end indices for the sliding window
                    start_idx = i * stride
                    end_idx = start_idx + window_size
                    chunk = flow_features[start_idx:end_idx]

                    # Skip empty chunks
                    if len(chunk) == 0:
                        continue

                    # Convert chunk to tensor
                    pkt_tensor = torch.tensor(chunk, dtype=torch.float32)

                    # Attention mask creation
                    flow_len = len(chunk)
                    if flow_len < max_seq_len:
                        attention_mask = torch.cat([torch.ones(flow_len), torch.zeros(max_seq_len - flow_len)])
                        pad = torch.zeros(max_seq_len - flow_len, pkt_tensor.shape[1])
                        pkt_tensor = torch.cat([pkt_tensor, pad], dim=0)
                    else:
                        pkt_tensor = pkt_tensor[:max_seq_len]
                        attention_mask = torch.ones(max_seq_len)

                    # Ensure the attention mask is the same length as the padded sequence
                    all_packet_seqs.append(pkt_tensor)
                    all_labels.append(label)
                    attention_masks.append(attention_mask)

    # Check if any flows were processed
    if not all_packet_seqs:
        raise RuntimeError("No flows found.")

    # Stack all packet sequences and convert to tensors
    packet_tensor = torch.stack(all_packet_seqs)  # [N, T, F]
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    attention_mask_tensor = torch.stack(attention_masks)  # [N, T]


    torch.save({
        "packet_seq": packet_tensor,  # [N, T, F]
        "label": label_tensor,  # [N]
        "attention_mask": attention_mask_tensor  # [N, T]
    }, output_file)

    print(f"\n[INFO] Preprocessing complete — saved {len(packet_tensor)} flows to {output_file}")
    print(f"[INFO] Shape: packets {packet_tensor.shape}, labels {label_tensor.shape}")


def find_label_from_path(file_path):
    current_path = os.path.dirname(file_path)
    while current_path != os.path.dirname(current_path):  # Stop at root
        folder_name = os.path.basename(current_path)
        for key in LABEL_MAPPING:
            if key.lower() in folder_name.lower():
                return LABEL_MAPPING[key]
        current_path = os.path.dirname(current_path)
    return -1

