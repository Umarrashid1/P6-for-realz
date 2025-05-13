# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import logging
from sklearn.preprocessing import MinMaxScaler # Added for data scaling

# --- 1. Configure Logging ---
# Set up logging to a file
logging.basicConfig(filename='gan_script.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w') # 'w' to overwrite the log file each time

logging.info("Script started.")

# --- 2. Define GAN Parameters ---
# These are minimal parameters for demonstration
latent_dim = 100      # Size of the random noise vector (input to generator)
data_dim = 5          # Number of features in the synthetic data -  *** IMPORTANT: UPDATE THIS TO MATCH YOUR DATASET ***
num_samples_to_generate = 1000
epochs = 5000         # Minimal number of epochs for demonstration
batch_size = 64
g_lr = 0.0002         # Generator learning rate
d_lr = 0.0002         # Discriminator learning rate
beta_1 = 0.5          # Adam optimizer beta1

# --- 3. Prepare Real Data ---
# This is where you load and preprocess YOUR dataset.
def load_and_preprocess_real_data(file_path, num_features):
    """
    Loads data from a CSV file, selects relevant features, and scales it.
    """
    try:
        logging.info(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data. Shape: {df.shape}")

        # *** IMPORTANT: Adapt this part to your dataset ***
        # For example, if your CSV has many columns, select the ones you want to model.
        # Ensure you have `num_features` columns selected.
        if df.shape[1] < num_features:
            logging.error(f"Dataset has fewer columns ({df.shape[1]}) than expected `data_dim` ({num_features}). Please check your `data_dim` or dataset.")
            raise ValueError("Not enough columns in the dataset for the specified `data_dim`.")

        # Assuming the first `num_features` columns are the ones you want.
        # If not, select them explicitly, e.g., df[['col1', 'col2', ...]]
        real_data_df = df.iloc[:, :num_features]
        logging.info(f"Selected {num_features} features. Shape after selection: {real_data_df.shape}")

        # Convert to numpy array
        real_data_np = real_data_df.values.astype('float32')

        # Scale data to the range [-1, 1] as the generator uses 'tanh' activation
        scaler = MinMaxScaler(feature_range=(-1, 1))
        real_data_scaled = scaler.fit_transform(real_data_np)
        logging.info(f"Data scaled to [-1, 1]. Shape: {real_data_scaled.shape}")

        return real_data_scaled, scaler # Return scaler to potentially de-scale generated data

    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise

dataset_path = '../../dataset/raw_dataset'

# Load your actual data
try:
    # The `data_dim` parameter (defined earlier) should match the number of features
    # you intend to use from your dataset.
    real_training_data, data_scaler = load_and_preprocess_real_data(dataset_path, data_dim)
    logging.info(f"Using real training data with shape: {real_training_data.shape}")
except Exception as e:
    logging.error(f"Failed to load real data. Exiting. Error: {e}")
    exit() # Exit if data loading fails


# --- 4. Build the Generator ---
def build_generator(latent_dim, output_dim):
    model = Sequential(name="Generator")
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='tanh')) # tanh to keep outputs in [-1, 1] range
    logging.info("Generator model built.")
    return model

# --- 5. Build the Discriminator ---
def build_discriminator(input_dim):
    model = Sequential(name="Discriminator")
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification (real/fake)
    logging.info("Discriminator model built.")
    return model

# --- 6. Build the GAN (Combined Model) ---
generator = build_generator(latent_dim, data_dim)
discriminator = build_discriminator(data_dim)

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=d_lr, beta_1=beta_1),
                      metrics=['accuracy'])

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
    idx = np.random.randint(0, real_training_data.shape[0], batch_size)
    real_samples = real_training_data[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise, verbose=0)

    real_labels = np.ones((batch_size, 1)) * 0.9
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # --- Train Generator ---
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_labels_for_generator = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels_for_generator)

    if (epoch + 1) % 100 == 0:
        logging.info(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.2f}% | G Loss: {g_loss:.4f}")

logging.info("GAN training finished.")

# --- 8. Generate Synthetic Data ---
logging.info(f"Generating {num_samples_to_generate} synthetic samples...")
noise_for_generation = np.random.normal(0, 1, (num_samples_to_generate, latent_dim))
synthetic_data_generated_scaled = generator.predict(noise_for_generation, verbose=0)

# De-normalize the data to its original scale
synthetic_data_generated = data_scaler.inverse_transform(synthetic_data_generated_scaled)
logging.info(f"Generated and de-normalized synthetic data. Shape: {synthetic_data_generated.shape}")


# --- 9. Save Synthetic Data to CSV ---
# Create column names based on the original data or generic ones
# If your original data had column names, you might want to use them.
# For this example, we'll use generic feature names.
column_names = [f'feature_{i+1}' for i in range(data_dim)]
if dataset_path and dataset_path != 'your_dataset.csv': # Try to get original column names
    try:
        original_df_cols = pd.read_csv(dataset_path, nrows=0).columns[:data_dim]
        if len(original_df_cols) == data_dim:
            column_names = original_df_cols
    except Exception:
        logging.warning("Could not read original column names, using generic ones.")


synthetic_df = pd.DataFrame(synthetic_data_generated, columns=column_names)
csv_filename = 'synthetic_data.csv'
try:
    synthetic_df.to_csv(csv_filename, index=False)
    logging.info(f"Synthetic data successfully saved to {csv_filename}")
except Exception as e:
    logging.error(f"Error saving synthetic data to CSV: {e}")

logging.info("Script finished.")
