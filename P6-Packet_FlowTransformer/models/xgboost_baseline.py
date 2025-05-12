import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pipeline.config import LABEL_MAPPING
import xgboost as xgb
import time
import argparse # Import argparse for command-line arguments

# --- Configuration ---


# IMPORTANT: Update this path to the main directory containing the class subdirectories
# It's now also configurable via command-line argument
BASE_DATA_DIR = "../../dataset/roni/DatasetFlow" # Default value

# Reverse mapping for printing results
LABEL_NAMES = {v: k for k, v in LABEL_MAPPING.items()}

# Default number of files to load in test mode
DEFAULT_MAX_FILES_TEST_MODE = 10

# Categorical and Numerical columns for flows
categorical_columns_flows = [
    "Src IP",
    "Dst IP",
    "Protocol",
    "Src Port",
    "Dst Port",
]

numerical_columns_flows = [
    "Flow Duration",
    "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

all_features = categorical_columns_flows + numerical_columns_flows


# --- Helper Functions ---

def find_label_from_path(file_path: str) -> int:
    """
    Derives the integer label from the file path based on parent directory names.
    Searches parent directories until a name matches a key in LABEL_MAPPING.
    """
    current_path = os.path.dirname(file_path)
    # Normalize path separator for consistency
    current_path = os.path.normpath(current_path)

    # Split the path into components
    path_components = current_path.split(os.sep)

    # Iterate backwards through components (closer directories first)
    for folder_name in reversed(path_components):
        if not folder_name:  # Skip empty components (e.g., from root '/')
            continue
        # Exact match first (case-insensitive)
        for key, label_value in LABEL_MAPPING.items():
            if key.lower() == folder_name.lower():
                # print(f"Found exact label '{key}' ({label_value}) in path component '{folder_name}' for file {os.path.basename(file_path)}")
                return label_value
        # If no exact match, check if key is part of the folder name (case-insensitive)
        for key, label_value in LABEL_MAPPING.items():
            if key.lower() in folder_name.lower():
                # print(f"Found partial label '{key}' ({label_value}) in path component '{folder_name}' for file {os.path.basename(file_path)}")
                return label_value


    print(f"Warning: Could not find label for path: {file_path}")
    return -1  # Return -1 if no label found


def load_data(base_dir: str, max_files: int | None = None) -> pd.DataFrame | None:
    """
    Loads CSV files recursively from the base directory,
    assigns labels based on folder names, and concatenates them.
    Optionally limits the number of files loaded via max_files.
    """
    all_data_frames = []
    # Recursively find all csv files
    try:
        csv_files = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True)
    except Exception as e:
        print(f"Error during file search in '{base_dir}': {e}")
        print("Please check if the BASE_DATA_DIR is correct and accessible.")
        return None

    if not csv_files:
        print(f"Error: No CSV files found under directory: {base_dir}")
        print("Please ensure BASE_DATA_DIR is set correctly and contains the dataset.")
        return None

    print(f"Found {len(csv_files)} CSV files in total.")

    files_to_process = csv_files
    if max_files is not None and max_files > 0:
        print(f"Limiting data loading to a maximum of {max_files} files.")
        # Optionally shuffle before taking the first N files for more variety in test mode
        # import random
        # random.shuffle(csv_files)
        files_to_process = csv_files[:max_files] # Take the first 'max_files' files

    if not files_to_process:
         print("Error: No files selected for processing (max_files might be 0 or negative, or no files found).")
         return None

    print(f"Attempting to load and process {len(files_to_process)} files...")

    processed_count = 0
    skipped_count = 0
    for i, file_path in enumerate(files_to_process):
        label = find_label_from_path(file_path)
        if label != -1:  # Only process files where a label was found
            try:
                # Read CSV, handle potential parsing errors if needed
                df_chunk = pd.read_csv(file_path, low_memory=False)

                # Clean column names (remove leading/trailing spaces)
                df_chunk.columns = df_chunk.columns.str.strip()

                # Check if all required features exist
                missing_cols = [col for col in all_features if col not in df_chunk.columns]
                if missing_cols:
                    print(f"Warning: Skipping file {file_path}. Missing columns: {missing_cols}")
                    skipped_count += 1
                    continue

                # Select only required features + add Label
                df_chunk = df_chunk[all_features].copy()  # Select features first
                df_chunk['Label'] = label  # Add label column
                all_data_frames.append(df_chunk)
                processed_count += 1

                # Print progress more frequently or at the end
                # if (processed_count) % 10 == 0 or (i + 1) == len(files_to_process):
                #     print(f"  Processed {processed_count}/{len(files_to_process)} attempted files...")

            except pd.errors.EmptyDataError:
                 print(f"Warning: Skipping empty file {file_path}")
                 skipped_count += 1
            except Exception as e:
                print(f"Error reading or processing file {file_path}: {e}")
                skipped_count += 1
        else:
            print(f"Skipping file due to missing label: {file_path}")
            skipped_count += 1

    print(f"Finished processing files. Successfully processed: {processed_count}, Skipped: {skipped_count}")

    if not all_data_frames:
        print("Error: No data loaded. Check file paths, formats, label extraction, and column names.")
        return None

    print("Concatenating dataframes...")
    full_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"Data loading complete. Total rows in loaded subset: {len(full_df)}")
    return full_df


def preprocess_data(df: pd.DataFrame, cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Preprocesses the data: handles inf/NaN, encodes categoricals, ensures numeric types.
    """
    print("Starting preprocessing...")
    start_preprocess_time = time.time()

    # 1. Select relevant columns (redundant if load_data selects, but safe)
    relevant_cols = cat_cols + num_cols + ['Label']
    # Ensure only existing columns are selected
    cols_to_select = [col for col in relevant_cols if col in df.columns]
    if len(cols_to_select) != len(relevant_cols):
         missing_in_df = set(relevant_cols) - set(df.columns)
         print(f"Warning: Columns missing in DataFrame before preprocessing: {missing_in_df}")
    df = df[cols_to_select].copy()


    # 2. Handle Infinite values (replace with NaN)
    print("  Handling infinite values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Handle Missing values (fill with 0 for baseline)
    # Check which columns have NaNs before filling
    nan_cols_before = df.columns[df.isna().any()].tolist()
    if nan_cols_before:
        print(f"  Handling missing values (filling with 0) in columns: {nan_cols_before}...")
        # Important: Fill NaNs *before* changing types
        df.fillna(0, inplace=True)
    else:
        print("  No missing values (NaN) found before filling.")


    # 4. Encode Categorical Features
    print("  Encoding categorical features...")
    label_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            # Convert to string first to handle potential mixed types gracefully
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoder if needed later
        else:
            print(f"  Warning: Categorical column '{col}' not found in DataFrame for encoding.")

    # 5. Ensure Numerical Columns are Numeric
    print("  Ensuring numerical columns are numeric...")
    nan_created = False
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object: # Only coerce if not already numeric
                 print(f"    Coercing column '{col}' to numeric...")
                 original_nan_count = df[col].isna().sum()
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 new_nan_count = df[col].isna().sum()
                 if new_nan_count > original_nan_count:
                     print(f"    Warning: Coercion created {new_nan_count - original_nan_count} new NaNs in column '{col}'.")
                     nan_created = True
            # Or check if dtype is numeric but contains non-finite values not caught earlier
            # This check might be redundant after replace inf/nan and fillna(0)
            # if not np.all(np.isfinite(df[col])):
            #      print(f"    Warning: Non-finite values found in numeric column '{col}' after initial handling.")

        else:
             print(f"  Warning: Numerical column '{col}' not found in DataFrame for type check.")


    # Fill any NaNs potentially created by pd.to_numeric(errors='coerce')
    if nan_created:
        print("  Re-filling NaNs potentially created by numeric coercion...")
        df.fillna(0, inplace=True)

    end_preprocess_time = time.time()
    print(f"Preprocessing finished in {end_preprocess_time - start_preprocess_time:.2f} seconds.")
    return df


# --- Main Execution Logic ---
def run_pipeline(base_dir: str, max_files: int | None = None):
    """Encapsulates the main workflow."""

    print("--- IoT DIAD 2024 XGBoost Baseline ---")
    if max_files is not None:
        print(f"*** RUNNING WITH MAX_FILES = {max_files} ***")

    # --- 1. Load Data ---
    start_load_time = time.time()
    df_raw = load_data(base_dir, max_files=max_files)
    end_load_time = time.time()

    if df_raw is None or df_raw.empty:
        print("Exiting due to data loading failure.")
        return # Exit function instead of script
    print(f"Data loaded in {end_load_time - start_load_time:.2f} seconds.")

    # Check if 'Label' column exists after loading
    if 'Label' not in df_raw.columns:
        print("Error: 'Label' column not found after loading data. Check load_data function and label assignment.")
        return

    print("\nInitial Data Info:")
    print(f"Shape: {df_raw.shape}")
    print("Label Distribution:")
    # Map integer labels back to names for printing value_counts
    if not df_raw['Label'].empty:
         print(df_raw['Label'].value_counts().sort_index().rename(index=LABEL_NAMES))
    else:
         print("Label column is empty.")
    print("-" * 30)

    # --- 2. Preprocess Data ---
    df_processed = preprocess_data(df_raw, categorical_columns_flows, numerical_columns_flows)
    del df_raw  # Free up memory

    # Check if 'Label' column still exists after preprocessing
    if 'Label' not in df_processed.columns:
        print("Error: 'Label' column lost during preprocessing.")
        return
    if df_processed.empty:
        print("Error: DataFrame became empty during preprocessing.")
        return

    print("\nProcessed Data Info:")
    print(f"Shape: {df_processed.shape}")
    print("Sample of processed data:")
    print(df_processed.head())
    # print("Data types after processing:")
    # print(df_processed.dtypes)
    # Check again for NaNs after processing
    if df_processed.isnull().any().any():
        print("\nWarning: NaNs still present after preprocessing!")
        print(df_processed.isnull().sum()[df_processed.isnull().sum() > 0]) # Show columns with NaNs
    else:
        print("\nNo NaNs detected after preprocessing.")
    print("-" * 30)

    # --- 3. Split Data ---
    print("Splitting data into training and testing sets (80/20 split)...")

    # Ensure Label column is suitable for stratification
    if df_processed['Label'].nunique() < 2:
         print(f"Warning: Only {df_processed['Label'].nunique()} unique labels found. Stratified split might behave unexpectedly.")
         # Decide how to handle: proceed without stratification or raise error
         stratify_param = None # Fallback to non-stratified split
    else:
         stratify_param = df_processed['Label']

    # Define features (X) and target (y) again from processed data
    features_in_processed = [col for col in all_features if col in df_processed.columns]
    if len(features_in_processed) != len(all_features):
         print(f"Warning: Not all expected features are present in the processed DataFrame. Using: {features_in_processed}")
    X = df_processed[features_in_processed]
    y = df_processed['Label']


    if len(X) == 0:
        print("Error: No data available for splitting after preprocessing.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
    except ValueError as e:
         print(f"Error during train_test_split (possibly due to too few samples per class for stratification): {e}")
         print("Attempting split without stratification...")
         try:
             X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=0.2, random_state=42 # Removed stratify
             )
         except ValueError as e2:
             print(f"Error during non-stratified split: {e2}")
             print("Cannot proceed with model training.")
             return


    del df_processed  # Free up memory

    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
    print("-" * 30)

    # Check if training or testing sets are empty
    if X_train.empty or X_test.empty:
        print("Error: Training or testing set is empty after split. Cannot train model.")
        return

    # --- 4. Train XGBoost Model ---
    print("Training XGBoost model...")
    start_train_time = time.time()

    # Determine the number of classes from the training labels
    num_classes = len(np.unique(y_train))
    print(f"Detected {num_classes} classes in the training data.")
    if num_classes <= 1:
        print("Error: Need at least 2 classes to train a classifier.")
        return

    # Baseline XGBoost Classifier for multi-class classification
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # Specify multi-class classification
        num_class=num_classes,      # Number of classes present in training data
        # use_label_encoder=False,  # Deprecated in newer XGBoost, defaults to False
        eval_metric='mlogloss',     # Logloss metric for multi-class
        # Add other parameters for tuning if needed (e.g., n_estimators, max_depth, learning_rate)
        # For baseline, defaults are often sufficient.
        random_state=42
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        # Consider adding more specific error handling if needed
        print("Check data types and values in training data.")
        print("X_train dtypes:\n", X_train.dtypes)
        print("y_train unique values:", np.unique(y_train))
        return


    end_train_time = time.time()
    print(f"Model training finished in {end_train_time - start_train_time:.2f} seconds.")
    print("-" * 30)

    # --- 5. Evaluate Model ---
    print("Evaluating model on the test set...")
    start_eval_time = time.time()

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Check data types and values in testing data.")
        print("X_test dtypes:\n", X_test.dtypes)
        return


    end_eval_time = time.time()
    print(f"Prediction finished in {end_eval_time - start_eval_time:.2f} seconds.")

    # --- Overall Accuracy ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("-" * 30)

    # --- Classification Report (Precision, Recall, F1-Score) ---
    print("Classification Report:")
    # Get target names in the correct order based on unique labels present in y_test/y_pred
    present_labels_in_results = sorted(np.unique(np.concatenate((y_test, y_pred))))
    target_names_ordered = [LABEL_NAMES.get(i, f"Unknown ({i})") for i in present_labels_in_results]
    # Ensure labels argument matches the order of target_names_ordered
    report = classification_report(y_test, y_pred, labels=present_labels_in_results, target_names=target_names_ordered, digits=4, zero_division=0)
    print(report)
    print("-" * 30)

    # --- Confusion Matrix and Class-wise Accuracy ---
    print("Confusion Matrix:")
    # Use the same labels as the classification report for consistency
    cm = confusion_matrix(y_test, y_pred, labels=present_labels_in_results)
    # Display confusion matrix with labels using pandas for clarity
    cm_df = pd.DataFrame(cm, index=target_names_ordered, columns=target_names_ordered)
    print(cm_df)

    print("\nClass-wise Accuracy (Recall):")
    # Calculate accuracy (recall) for each class directly from the CM diagonal
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    # Handle potential division by zero if a class has no true samples in the test set (shouldn't happen with default CM)
    class_accuracy = np.nan_to_num(class_accuracy) # Replace NaN with 0

    if len(class_accuracy) == len(target_names_ordered):
        for i, class_name in enumerate(target_names_ordered):
            print(f"  - {class_name}: {class_accuracy[i]:.4f}")
    else:
        print("Warning: Mismatch between confusion matrix dimensions and target names.")
        print("Raw class accuracy array:", class_accuracy)

    print("-" * 30)


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost model on network flow data.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../dataset/roni/DatasetFlow", # Default value if not provided
        help="Path to the base directory containing class subdirectories with CSV files."
    )
    parser.add_argument(
        "--test",
        action="store_true", # Makes it a flag: presence means True, absence means False
        help=f"Run in test mode, loading a limited number of files (default: {DEFAULT_MAX_FILES_TEST_MODE})."
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Explicitly set the maximum number of CSV files to load. Overrides the default test mode limit if --test is also used."
    )

    args = parser.parse_args()

    # Determine the final max_files value based on arguments
    limit_files = None
    if args.test:
        limit_files = args.max_files if args.max_files is not None else DEFAULT_MAX_FILES_TEST_MODE
    elif args.max_files is not None: # Allow limiting files even if not in test mode
        limit_files = args.max_files

    # Run the main pipeline
    run_pipeline(base_dir=args.data_dir, max_files=limit_files)

    print("--- Script Finished ---")