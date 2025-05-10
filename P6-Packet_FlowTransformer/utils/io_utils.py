import os
import glob
import pandas as pd

def list_csv_files(dataset_dir: str):
    return [
        f for f in glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        if not f.endswith(":Zone.Identifier")
    ]

def load_csv_file(file_path: str, test_mode: bool = False, rows_per_file: int = 20000):
    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
            nrows=rows_per_file if test_mode and rows_per_file else None,
        )
        return df, file_path
    except Exception as e:
        print(f"[SKIP] Could not read {file_path}: {e}")
        return None
