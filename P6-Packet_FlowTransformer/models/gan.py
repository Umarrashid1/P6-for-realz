# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pipeline.config import numerical_columns, categorical_columns



# Ensure there are no overlaps between CATEGORICAL_COLUMNS and NUMERICAL_COLUMNS
# and that they cover all features you intend to use from your CSV.

# --- 1. Configure Logging ---
logging.basicConfig(filename='gan_script.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')
logging.info("Script started. Preprocessing will be applied and then reversed for final output.")

# --- 2. Define GAN Parameters ---
latent_dim = 100
# data_dim will be determined after preprocessing (due to one-hot encoding)
num_samples_to_generate = 1000
epochs = 5000  # Adjust as needed
batch_size = 64
g_lr = 0.0002
d_lr = 0.0002
beta_1 = 0.5


# --- 3. Prepare Real Data ---
def load_and_preprocess_real_data(file_path, numerical_cols, categorical_cols):
    """
    Loads data, separates numerical and categorical features,
    scales numerical, and one-hot encodes categorical.
    Returns the processed data, the preprocessor object, and the new data dimension.
    """
    try:
        logging.info(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data. Original shape: {df.shape}")

        # Select only the columns we're interested in
        all_relevant_columns = numerical_cols + categorical_cols
        # Ensure all listed columns exist in the DataFrame
        missing_cols = [col for col in all_relevant_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"The following specified columns are missing from the CSV: {missing_cols}")
            raise KeyError(f"Missing columns in CSV: {missing_cols}")

        df_relevant = df[all_relevant_columns].copy()  # Use .copy() to avoid SettingWithCopyWarning
        logging.info(f"Selected relevant columns. Shape: {df_relevant.shape}")

        # Handle missing values (important for real-world data)
        # Numerical: fill with median
        for col in numerical_cols:
            if df_relevant[col].isnull().any():
                median_val = df_relevant[col].median()
                df_relevant[col].fillna(median_val, inplace=True)
                logging.info(f"Filled NaNs in numerical column '{col}' with median {median_val}.")
        # Categorical: fill with a placeholder string 'MISSING'
        for col in categorical_cols:
            # Convert to string first to ensure fillna works as expected and to handle mixed types
            df_relevant[col] = df_relevant[col].astype(str)
            if df_relevant[col].isnull().any() or (
                    df_relevant[col] == 'nan').any():  # Pandas might read 'nan' as string
                df_relevant[col].fillna('MISSING', inplace=True)
                df_relevant[col].replace('nan', 'MISSING', inplace=True)  # Explicitly replace string 'nan'
                logging.info(f"Filled NaNs/string 'nan' in categorical column '{col}' with placeholder 'MISSING'.")

        # Define transformers
        # Numerical features: Scale to [-1, 1] for tanh activation in generator
        numerical_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])

        # Categorical features: One-hot encode
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            # sparse_output=False for dense array
        ])

        # Create a preprocessor object using ColumnTransformer
        # This object will be fitted on the training data and used to transform it.
        # It will also be used to inverse_transform the generated data.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'  # Drop any columns not specified in numerical_cols or categorical_cols
        )

        # Fit the preprocessor on the relevant data and transform it
        logging.info("Fitting preprocessor and transforming data...")
        real_data_processed_np = preprocessor.fit_transform(df_relevant)
        logging.info(f"Data processed. Shape after preprocessing: {real_data_processed_np.shape}")

        # The number of features for the GAN is now the width of this processed array
        processed_data_dim = real_data_processed_np.shape[1]
        logging.info(f"Effective data dimension for GAN (after one-hot encoding, etc.): {processed_data_dim}")

        return real_data_processed_np.astype('float32'), preprocessor, processed_data_dim

    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        raise
    except KeyError as e:
        logging.error(
            f"KeyError: {e}. This likely means a column specified in NUMERICAL_COLUMNS or CATEGORICAL_COLUMNS was not found in the CSV file.")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise


file_path = "../../../dataset/raw_dataset"

try:
    real_training_data, data_preprocessor, data_dim = load_and_preprocess_real_data(
        file_path, numerical_columns, categorical_columns
    )
    logging.info(f"Using real training data with shape: {real_training_data.shape}, data_dim for GAN: {data_dim}")
    if data_dim == 0:
        logging.error("Data dimension for GAN is 0. Check column lists and data. Exiting.")
        exit()
except Exception as e:
    logging.error(f"Failed to load or preprocess real data. Exiting. Error: {e}")
    exit()


# --- 4. Build the Generator ---
def build_generator(latent_dim, output_dim):
    model = Sequential(name="Generator")
    # Adjust layer sizes if data_dim becomes very large due to one-hot encoding
    model.add(Dense(128 if output_dim < 256 else 256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256 if output_dim < 512 else 512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512 if output_dim < 1024 else 1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    # Output layer uses tanh because numerical data is scaled to [-1, 1]
    # and one-hot encoded data is also effectively in a similar range (0s and 1s)
    # which tanh can represent.
    model.add(Dense(output_dim, activation='tanh'))
    logging.info("Generator model built with tanh output activation.")
    return model


# --- 5. Build the Discriminator ---
def build_discriminator(input_dim):
    model = Sequential(name="Discriminator")
    # Adjust layer sizes based on input_dim
    model.add(Dense(512 if input_dim < 1024 else 1024, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256 if input_dim < 512 else 512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification (real/fake)
    logging.info("Discriminator model built.")
    return model


# --- 6. Build the GAN (Combined Model) ---
generator = build_generator(latent_dim, data_dim)
discriminator = build_discriminator(data_dim)

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=d_lr, beta_1=beta_1),
                      metrics=['accuracy'])

# For the combined GAN model, we only train the generator
discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
generated_data_point = generator(gan_input)
gan_output = discriminator(generated_data_point)

gan = Model(gan_input, gan_output, name="GAN")
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=g_lr, beta_1=beta_1))
logging.info("GAN model built and compiled.")

# --- 7. Training the GAN ---
logging.info(f"Starting GAN training for {epochs} epochs.")
for epoch in range(epochs):
    # --- Train Discriminator ---
    # Select a random batch of real (processed) samples
    idx = np.random.randint(0, real_training_data.shape[0], batch_size)
    real_samples = real_training_data[idx]

    # Generate a batch of new synthetic (processed) samples
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise, verbose=0)

    # Labels for real and fake samples (with label smoothing for real labels)
    real_labels = np.ones((batch_size, 1)) * 0.9
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # --- Train Generator ---
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # We want the discriminator to classify these as real (label 1)
    valid_labels_for_generator = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels_for_generator)

    if (epoch + 1) % 100 == 0:
        logging.info(
            f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1] * 100:.2f}% | G Loss: {g_loss:.4f}")
logging.info("GAN training finished.")

# --- 8. Generate Synthetic Data (in processed format initially) ---
logging.info(f"Generating {num_samples_to_generate} synthetic samples (in processed format)...")
noise_for_generation = np.random.normal(0, 1, (num_samples_to_generate, latent_dim))
synthetic_data_processed = generator.predict(noise_for_generation, verbose=0)
logging.info(f"Generated processed synthetic data. Shape: {synthetic_data_processed.shape}")

# --- 9. Inverse Transform Synthetic Data and Save to CSV ---
try:
    logging.info("Attempting to inverse transform synthetic data to original format...")
    # The data_preprocessor (ColumnTransformer) expects a 2D array with the same number of columns
    # it was fitted on (i.e., the processed_data_dim).
    # The `inverse_transform` method will then apply the inverse of MinMaxScaler to the numerical parts
    # and the inverse of OneHotEncoder to the categorical parts.

    synthetic_data_reconstructed_np = data_preprocessor.inverse_transform(synthetic_data_processed)

    # Convert the reconstructed numpy array back to a DataFrame with original column names
    # The order of columns in `inverse_transform` output matches the order in `all_relevant_columns`
    # used during fitting the preprocessor.
    original_column_order = numerical_columns + categorical_columns
    synthetic_df_reconstructed = pd.DataFrame(synthetic_data_reconstructed_np, columns=original_column_order)

    logging.info(f"Synthetic data inverse transformed. Shape: {synthetic_df_reconstructed.shape}")

    # Post-processing: Ensure correct data types for numerical columns if needed
    # (MinMaxScaler's inverse_transform usually returns float)
    for col in numerical_columns:
        # If original numerical columns were integers, you might want to round and cast
        # For example, if 'ttl' should be an integer:
        # if col == 'ttl':
        #    synthetic_df_reconstructed[col] = pd.to_numeric(synthetic_df_reconstructed[col], errors='coerce').round().astype('Int64')
        # else:
        synthetic_df_reconstructed[col] = pd.to_numeric(synthetic_df_reconstructed[col], errors='coerce')

    # For categorical columns, OneHotEncoder's inverse_transform returns them as objects (strings)
    # which is usually what's desired.

    csv_filename = 'synthetic_data_resembling_original.csv'
    synthetic_df_reconstructed.to_csv(csv_filename, index=False)
    logging.info(f"Synthetic data resembling original format successfully saved to {csv_filename}")

except Exception as e:
    logging.error(f"Error during inverse transformation or saving CSV: {e}")
    logging.warning(
        "Saving processed data instead (before inverse transform) to 'synthetic_data_processed_fallback.csv'")
    # Fallback: save the processed data if inverse transform fails, so output is not lost
    # Need to get column names for the processed data for this fallback
    try:
        processed_feature_names = data_preprocessor.get_feature_names_out()
    except Exception:  # Older sklearn might not have get_feature_names_out directly on ColumnTransformer
        processed_feature_names = [f"proc_feat_{i}" for i in range(synthetic_data_processed.shape[1])]

    synthetic_df_processed_fallback = pd.DataFrame(synthetic_data_processed, columns=processed_feature_names)
    synthetic_df_processed_fallback.to_csv('synthetic_data_processed_fallback.csv', index=False)

logging.info("Script finished.")
