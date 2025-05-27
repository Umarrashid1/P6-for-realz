import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score  # Using F1 score for optimization
from pipeline.config import LABEL_MAPPING, categorical_columns_flows, numerical_columns_flows
import xgboost as xgb
import time
import argparse
import optuna
import logging
import sys

# --- Configuration (Adapted from xgboost_baseline.py) ---
BASE_DATA_DIR_DEFAULT = "../../dataset/roni/DatasetFlow"  # Default, can be overridden
LABEL_NAMES = {v: k for k, v in LABEL_MAPPING.items()}
all_features = categorical_columns_flows + numerical_columns_flows

# --- Logging Setup (Similar to other tune_*.py scripts) ---
LOG_FILE_OPTUNA_XGBOOST = "optuna_xgboost_baseline_log.txt"
script_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("tune_xgboost_baseline_script.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


# --- Helper Functions (Adapted from xgboost_baseline.py) ---

def find_label_from_path(file_path: str) -> int:
    current_path = os.path.dirname(file_path)
    current_path = os.path.normpath(current_path)
    path_components = current_path.split(os.sep)
    for folder_name in reversed(path_components):
        if not folder_name:
            continue
        for key, label_value in LABEL_MAPPING.items():
            if key.lower() == folder_name.lower():
                return label_value
        for key, label_value in LABEL_MAPPING.items():
            if key.lower() in folder_name.lower():
                return label_value
    # objective_logger.warning(f"Could not find label for path: {file_path}") # Use logger if available
    return -1


def load_data(base_dir: str, max_files: int | None = None, logger_instance=None) -> pd.DataFrame | None:
    if logger_instance is None: logger_instance = script_logger
    all_data_frames = []
    try:
        csv_files = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True)
    except Exception as e:
        logger_instance.error(f"Error during file search in '{base_dir}': {e}")
        return None

    if not csv_files:
        logger_instance.error(f"Error: No CSV files found under directory: {base_dir}")
        return None
    logger_instance.debug(f"Found {len(csv_files)} CSV files in total.")

    files_to_process = csv_files
    if max_files is not None and max_files > 0:
        logger_instance.info(f"Limiting data loading to a maximum of {max_files} files.")
        import random
        random.shuffle(csv_files)
        files_to_process = csv_files[:max_files]

    if not files_to_process:
        logger_instance.error("Error: No files selected for processing.")
        return None
    logger_instance.info(f"Attempting to load and process {len(files_to_process)} files...")

    processed_count = 0
    skipped_count = 0
    for i, file_path in enumerate(files_to_process):
        label = find_label_from_path(file_path)
        if label != -1:
            try:
                df_chunk = pd.read_csv(file_path, low_memory=False)
                df_chunk.columns = df_chunk.columns.str.strip()
                missing_cols = [col for col in all_features if col not in df_chunk.columns]
                if missing_cols:
                    logger_instance.warning(f"Skipping file {file_path}. Missing columns: {missing_cols}")
                    skipped_count += 1
                    continue
                df_chunk = df_chunk[all_features].copy()
                df_chunk['Label'] = label
                all_data_frames.append(df_chunk)
                processed_count += 1
            except pd.errors.EmptyDataError:
                logger_instance.warning(f"Skipping empty file {file_path}")
                skipped_count += 1
            except Exception as e:
                logger_instance.error(f"Error reading or processing file {file_path}: {e}")
                skipped_count += 1
        else:
            # logger_instance.warning(f"Skipping file due to missing label: {file_path}") # Can be too verbose
            skipped_count += 1

    logger_instance.info(f"Finished file loading. Processed: {processed_count}, Skipped: {skipped_count}")
    if not all_data_frames:
        logger_instance.error("No data loaded.")
        return None
    full_df = pd.concat(all_data_frames, ignore_index=True)
    logger_instance.info(f"Data loading complete. Total rows: {len(full_df)}")
    return full_df


def preprocess_data(df: pd.DataFrame, cat_cols: list, num_cols: list, logger_instance=None) -> tuple[
                                                                                                   pd.DataFrame, pd.Series] | None:
    if logger_instance is None: logger_instance = script_logger
    logger_instance.info("Starting preprocessing...")

    relevant_cols = cat_cols + num_cols + ['Label']
    cols_to_select = [col for col in relevant_cols if col in df.columns]
    if len(cols_to_select) != len(relevant_cols):
        missing_in_df = set(relevant_cols) - set(df.columns)
        logger_instance.warning(f"Columns missing in DataFrame before preprocessing: {missing_in_df}")
    df = df[cols_to_select].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)  # Simple fill for baseline

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            logger_instance.warning(f"Categorical column '{col}' not found for encoding.")

    nan_created = False
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object:
                original_nan_count = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                new_nan_count = df[col].isna().sum()
                if new_nan_count > original_nan_count:
                    nan_created = True
        else:
            logger_instance.warning(f"Numerical column '{col}' not found for type check.")
    if nan_created:
        df.fillna(0, inplace=True)

    logger_instance.info("Preprocessing finished.")

    if 'Label' not in df.columns:
        logger_instance.error("Label column not found after preprocessing.")
        return None

    features_in_processed = [col for col in all_features if col in df.columns]
    X = df[features_in_processed]
    y = df['Label']
    return X, y


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial, base_dir: str, max_files_per_trial: int | None):
    objective_logger = logging.getLogger(f"optuna_trial_xgb_{trial.number}")
    objective_logger.info(f"--- Starting Optuna XGBoost Baseline Trial: {trial.number} ---")

    # Load and preprocess data for this trial
    # For efficiency in real scenarios, you might load data once outside the objective
    # and sample/split, or use a smaller, consistent subset for tuning.
    df_raw = load_data(base_dir, max_files=max_files_per_trial, logger_instance=objective_logger)
    if df_raw is None or df_raw.empty:
        objective_logger.error(f"Trial {trial.number}: Data loading failed or returned empty. Pruning.")
        raise optuna.TrialPruned("Data loading failed.")

    preprocess_result = preprocess_data(df_raw, categorical_columns_flows, numerical_columns_flows,
                                        logger_instance=objective_logger)
    del df_raw
    if preprocess_result is None:
        objective_logger.error(f"Trial {trial.number}: Preprocessing failed. Pruning.")
        raise optuna.TrialPruned("Preprocessing failed.")

    X, y = preprocess_result

    if X.empty or y.empty:
        objective_logger.error(f"Trial {trial.number}: No data after preprocessing. Pruning.")
        raise optuna.TrialPruned("No data after preprocessing.")

    # Split into training and validation sets for this trial
    # Using a fixed random_state for the split within a trial ensures that
    # hyperparameter changes are evaluated on the same split.
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    except ValueError:  # Fallback if stratification fails (e.g. too few samples for a class)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    del X, y

    if X_train.empty or X_val.empty:
        objective_logger.error(f"Trial {trial.number}: Train or validation set is empty after split. Pruning.")
        raise optuna.TrialPruned("Train/Validation split resulted in empty set.")

    # Hyperparameters to tune for XGBoost
    params = {
        "objective": "multi:softmax",
        "num_class": len(np.unique(y_train)),  # Determine num_class from training data
        "eval_metric": "mlogloss",
        "verbosity": 0,  # Suppress XGBoost's own prints
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),  # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),  # L1 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        # If using GPU:
        # "tree_method": "gpu_hist",
        # "predictor": "gpu_predictor"
    }

    if params["num_class"] <= 1:
        objective_logger.error(
            f"Trial {trial.number}: Not enough classes in training data ({params['num_class']}). Pruning.")
        raise optuna.TrialPruned("Not enough classes.")

    objective_logger.info(f"Trial {trial.number} Hyperparameters: {params}")

    model = xgb.XGBClassifier(**params)

    try:
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=10,  # Optional: for early stopping
                  verbose=False)  # Suppress XGBoost's own prints during fit
    except Exception as e:
        objective_logger.error(f"Trial {trial.number}: Error during model.fit: {e}", exc_info=True)
        raise optuna.TrialPruned("Error during model training.")

    preds = model.predict(X_val)
    # Using macro F1-score as it's good for imbalanced multi-class problems
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)

    objective_logger.info(f"Trial {trial.number} Validation Macro F1-score: {f1:.4f}")

    # Optuna pruning (optional, if not using XGBoost's early stopping or want additional pruning)
    # trial.report(f1, step=trial.number) # Example for reporting
    # if trial.should_prune():
    #     raise optuna.TrialPruned()

    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost baseline model using Optuna.")
    parser.add_argument(
        "--data_dir", type=str, default=BASE_DATA_DIR_DEFAULT,
        help="Path to the base directory containing CSV files."
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of Optuna trials to run."
    )
    parser.add_argument(
        "--max_files_per_trial", type=int, default=20,  # Limit files for faster tuning trials
        help="Maximum number of CSV files to load per Optuna trial (0 for all in data_dir)."
    )
    args = parser.parse_args()

    # Configure Optuna's logging
    optuna_stream_handler = logging.StreamHandler(sys.stdout)
    optuna_file_handler = logging.FileHandler(LOG_FILE_OPTUNA_XGBOOST, mode="a")
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna_logger_instance = optuna.logging.get_logger("optuna")
    optuna_logger_instance.addHandler(optuna_stream_handler)
    optuna_logger_instance.addHandler(optuna_file_handler)
    optuna_logger_instance.setLevel(logging.INFO)

    script_logger.info(f"Starting Optuna study for XGBoost Baseline.")
    script_logger.info(f"Data directory: {args.data_dir}")
    script_logger.info(f"Number of trials: {args.n_trials}")
    script_logger.info(f"Max files per trial: {'All' if args.max_files_per_trial == 0 else args.max_files_per_trial}")
    script_logger.info(f"Optuna's logs will be in: {LOG_FILE_OPTUNA_XGBOOST}")
    script_logger.info(f"This script's general logs will be in: tune_xgboost_baseline_script.log")

    study_name_xgboost = "xgboost-baseline-study"
    storage_name_xgboost = f"sqlite:///{study_name_xgboost}.db"

    study = optuna.create_study(
        study_name=study_name_xgboost,
        storage=storage_name_xgboost,
        direction="maximize",
        load_if_exists=True,
        # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1) # Example pruner
    )

    # Pass additional arguments to the objective function using a lambda
    effective_max_files = args.max_files_per_trial if args.max_files_per_trial > 0 else None
    study.optimize(lambda trial: objective(trial, args.data_dir, effective_max_files),
                   n_trials=args.n_trials)

    script_logger.info("\n--- Optuna XGBoost Baseline Study Complete ---")
    if study.best_trial:
        script_logger.info(f"Best trial number: {study.best_trial.number}")
        script_logger.info(f"Best validation macro F1-score: {study.best_value:.4f}")
        script_logger.info("Best hyperparameters:")
        for key, value in study.best_params.items():
            script_logger.info(f"  {key}: {value}")
    else:
        script_logger.info("No successful trials completed in this study session.")

    script_logger.info(f"\nStudy statistics: {study.trials_dataframe().shape[0]} trials completed.")
    script_logger.info(f"To visualize: optuna-dashboard {storage_name_xgboost}")