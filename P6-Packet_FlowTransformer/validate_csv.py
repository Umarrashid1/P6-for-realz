import os
import pandas as pd
import numpy as np
from collections import defaultdict
from pipeline.config import categorical_columns, numerical_columns  # adjust if needed

DATASET_DIR = "/ceph/project/P6-iot-flow-ids/dataset/raw_dataset/DatasetAnomaly"

summary = []
cardinalities = defaultdict(int)

for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            info = {
                "file": file_path,
                "rows": len(df),
                "missing_num": [],
                "missing_cat": [],
                "nan_num": 0,
                "nan_cat": 0
            }

            # Check columns
            info["missing_num"] = [col for col in numerical_columns if col not in df.columns]
            info["missing_cat"] = [col for col in categorical_columns if col not in df.columns]

            if not info["missing_num"]:
                num_df = df[numerical_columns].astype(np.float32)
                info["nan_num"] = int(num_df.isna().sum().sum())

            if not info["missing_cat"]:
                cat_df = df[categorical_columns].astype("category")
                info["nan_cat"] = int(cat_df.isna().sum().sum())

                # Track max unique values (cardinalities)
                for col in categorical_columns:
                    count = len(cat_df[col].cat.categories)
                    cardinalities[col] = max(cardinalities[col], count)

            summary.append(info)

        except Exception as e:
            summary.append({"file": file_path, "error": str(e)})

# Print summary
print("\n==== CSV FILE CHECK SUMMARY ====")
for s in summary:
    if "error" in s:
        print(f"[ERROR] {s['file']}: {s['error']}")
    else:
        print(f"{s['file']}: {s['rows']} rows, NaNs(num)={s['nan_num']}, NaNs(cat)={s['nan_cat']}")

# Print cardinalities
print("\n==== MAX CARDINALITIES PER CATEGORICAL COLUMN ====")
for col, val in cardinalities.items():
    print(f"{col}: {val}")

# Optional: write to JSON
import json
with open("cardinalities.json", "w") as f:
    json.dump(cardinalities, f, indent=2)
