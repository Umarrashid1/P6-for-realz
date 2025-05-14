# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pipeline.config import numerical_columns_packets, categorical_columns_packets

# --- 0. Define your column lists ---
# In a real scenario, you might load these from a config file or have them imported.
# For this script, we define them directly.
# The user had: from pipeline.config import numerical_columns, categorical_columns
# We are defining them here to make the script self-contained.


# --- 1. Configure Logging ---
logging.basicConfig(filename='gan_script_pytorch.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')
logging.info("Script started (PyTorch version). Preprocessing will be applied and then reversed for final output.")

# --- 2. Define GAN Parameters & Device ---
latent_dim = 100
# data_dim will be determined after preprocessing
num_samples_to_generate = 100000
epochs = 5000  # Adjust as needed
batch_size = 64
g_lr = 0.0002
d_lr = 0.0002
beta1 = 0.5  # Adam optimizer beta1

# Determine device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

logging.info(f"Using device: {device}")


# --- 3. Prepare Real Data (Same as TensorFlow version, using scikit-learn) ---
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

        all_relevant_columns = numerical_cols + categorical_cols
        missing_cols = [col for col in all_relevant_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"The following specified columns are missing from the CSV: {missing_cols}")
            raise KeyError(f"Missing columns in CSV: {missing_cols}")

        df_relevant = df[all_relevant_columns].copy()
        logging.info(f"Selected relevant columns. Shape: {df_relevant.shape}")

        for col in numerical_cols:
            if df_relevant[col].isnull().any():
                median_val = df_relevant[col].median()
                df_relevant[col].fillna(median_val, inplace=True)
                logging.info(f"Filled NaNs in numerical column '{col}' with median {median_val}.")
        for col in categorical_cols:
            df_relevant[col] = df_relevant[col].astype(str)
            if df_relevant[col].isnull().any() or (df_relevant[col] == 'nan').any():
                df_relevant[col].fillna('MISSING', inplace=True)
                df_relevant[col].replace('nan', 'MISSING', inplace=True)
                logging.info(f"Filled NaNs/string 'nan' in categorical column '{col}' with placeholder 'MISSING'.")

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler(feature_range=(-1, 1)))])
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)],
            remainder='drop')

        logging.info("Fitting preprocessor and transforming data...")
        real_data_processed_np = preprocessor.fit_transform(df_relevant)
        logging.info(f"Data processed. Shape after preprocessing: {real_data_processed_np.shape}")
        processed_data_dim = real_data_processed_np.shape[1]
        logging.info(f"Effective data dimension for GAN: {processed_data_dim}")

        return real_data_processed_np.astype('float32'), preprocessor, processed_data_dim

    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        raise
    except KeyError as e:
        logging.error(f"KeyError: {e}. Column not found in CSV.")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise



file_path = "../../dataset/raw_dataset/DatasetAnomaly/Web-Based/Backdoor_Malware/Backdoor_Malware.csv"

try:
    real_training_data_np, data_preprocessor, data_dim = load_and_preprocess_real_data(
        file_path, numerical_columns_packets, categorical_columns_packets
    )
    # Convert to PyTorch tensor
    real_training_data = torch.tensor(real_training_data_np, dtype=torch.float32).to(device)
    logging.info(
        f"Using real training data (tensor on {device}) with shape: {real_training_data.shape}, data_dim for GAN: {data_dim}")
    if data_dim == 0:
        logging.error("Data dimension for GAN is 0. Check column lists and data. Exiting.")
        exit()
except Exception as e:
    logging.error(f"Failed to load or preprocess real data. Exiting. Error: {e}")
    exit()


# --- 4. Build the Generator (PyTorch) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 if output_dim < 256 else 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128 if output_dim < 256 else 256),

            nn.Linear(128 if output_dim < 256 else 256, 256 if output_dim < 512 else 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256 if output_dim < 512 else 512),

            nn.Linear(256 if output_dim < 512 else 512, 512 if output_dim < 1024 else 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512 if output_dim < 1024 else 1024),

            nn.Linear(512 if output_dim < 1024 else 1024, output_dim),
            nn.Tanh()  # Output matches scaled data range [-1, 1]
        )
        logging.info("Generator model (PyTorch) built.")

    def forward(self, z):
        return self.model(z)


# --- 5. Build the Discriminator (PyTorch) ---
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512 if input_dim < 1024 else 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512 if input_dim < 1024 else 1024, 256 if input_dim < 512 else 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256 if input_dim < 512 else 512, 1),
            nn.Sigmoid()  # Output probability (real/fake)
        )
        logging.info("Discriminator model (PyTorch) built.")

    def forward(self, data):
        return self.model(data)


# --- 6. Initialize Models, Optimizers, and Loss Function ---
generator = Generator(latent_dim, data_dim).to(device)
discriminator = Discriminator(data_dim).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, 0.999))

# Loss function
adversarial_loss = nn.BCELoss().to(device)  # Binary Cross Entropy Loss

logging.info("Models, optimizers, and loss function initialized.")

# --- 7. Training the GAN (PyTorch) ---
logging.info(f"Starting GAN training for {epochs} epochs on device: {device}")

for epoch in range(epochs):
    for i in range(real_training_data.size(0) // batch_size):  # Iterate over batches
        # --- Train Discriminator ---
        discriminator.train()
        generator.eval()  # Keep generator in eval mode while training discriminator
        optimizer_D.zero_grad()

        # Real samples
        idx = torch.randperm(real_training_data.size(0))[:batch_size]  # More robust batch selection
        real_samples = real_training_data[idx].to(device)
        real_labels = torch.full((batch_size, 1), 0.9, dtype=torch.float32, device=device)  # Label smoothing

        # Discriminator output for real samples
        d_output_real = discriminator(real_samples)
        d_loss_real = adversarial_loss(d_output_real, real_labels)

        # Fake samples
        noise = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():  # Detach generator output
            fake_samples = generator(noise)
        fake_labels = torch.full((batch_size, 1), 0.0, dtype=torch.float32, device=device)

        # Discriminator output for fake samples
        d_output_fake = discriminator(fake_samples.detach())  # Detach to avoid training generator here
        d_loss_fake = adversarial_loss(d_output_fake, fake_labels)

        # Total discriminator loss and backpropagation
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        generator.train()  # Switch generator to train mode
        discriminator.eval()  # Keep discriminator in eval mode
        optimizer_G.zero_grad()

        # Generate new fake samples
        noise = torch.randn(batch_size, latent_dim, device=device)
        gen_samples = generator(noise)

        # We want the discriminator to classify these as real (label 1)
        valid_labels_for_generator = torch.full((batch_size, 1), 1.0, dtype=torch.float32, device=device)

        # Discriminator output for the generator's fake samples
        d_output_for_g = discriminator(gen_samples)
        g_loss = adversarial_loss(d_output_for_g, valid_labels_for_generator)

        # Generator loss and backpropagation
        g_loss.backward()
        optimizer_G.step()

    # Log progress (e.g., at the end of each epoch)
    # For accuracy, we can check d_output_real and d_output_fake predictions
    # This is a simplified accuracy for the last batch
    with torch.no_grad():
        d_acc_real = ((d_output_real > 0.5).float().mean().item()) * 100
        d_acc_fake = ((d_output_fake < 0.5).float().mean().item()) * 100
        d_acc = (d_acc_real + d_acc_fake) / 2

    if (epoch + 1) % 100 == 0:  # Log every 100 epochs
        logging.info(
            f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | D Acc: {d_acc:.2f}% (R:{d_acc_real:.2f} F:{d_acc_fake:.2f})")

logging.info("GAN training finished.")

# --- 8. Generate Synthetic Data (PyTorch) ---
logging.info(f"Generating {num_samples_to_generate} synthetic samples (in processed format)...")
generator.eval()  # Set generator to evaluation mode
with torch.no_grad():  # No need to track gradients
    noise_for_generation = torch.randn(num_samples_to_generate, latent_dim, device=device)
    synthetic_data_processed_tensor = generator(noise_for_generation)

# Move to CPU and convert to NumPy array for scikit-learn's inverse_transform
synthetic_data_processed_np = synthetic_data_processed_tensor.cpu().numpy()
logging.info(f"Generated processed synthetic data (NumPy). Shape: {synthetic_data_processed_np.shape}")

# --- 9. Inverse Transform Synthetic Data and Save to CSV (Same as TensorFlow version) ---
try:
    logging.info("Attempting to inverse transform synthetic data to original format...")
    synthetic_data_reconstructed_np = data_preprocessor.inverse_transform(synthetic_data_processed_np)

    original_column_order = numerical_columns_packets + categorical_columns_packets
    synthetic_df_reconstructed = pd.DataFrame(synthetic_data_reconstructed_np, columns=original_column_order)
    logging.info(f"Synthetic data inverse transformed. Shape: {synthetic_df_reconstructed.shape}")

    for col in numerical_columns_packets:
        synthetic_df_reconstructed[col] = pd.to_numeric(synthetic_df_reconstructed[col], errors='coerce')

    csv_filename = 'synthetic_data_resembling_original_pytorch.csv'
    synthetic_df_reconstructed.to_csv(csv_filename, index=False)
    logging.info(f"Synthetic data resembling original format successfully saved to {csv_filename}")

except Exception as e:
    logging.error(f"Error during inverse transformation or saving CSV: {e}")
    logging.warning(
        "Saving processed data instead (before inverse transform) to 'synthetic_data_processed_fallback_pytorch.csv'")
    try:
        processed_feature_names = data_preprocessor.get_feature_names_out()
    except AttributeError:  # Older sklearn might not have get_feature_names_out directly
        transformers = data_preprocessor.transformers_
        processed_feature_names = []
        for name, trans, cols in transformers:
            if name == 'num':
                processed_feature_names.extend(cols)
            elif name == 'cat':
                # For OneHotEncoder, get_feature_names_out is on the OneHotEncoder instance
                ohe = trans.named_steps['onehot']
                processed_feature_names.extend(ohe.get_feature_names_out(cols))
            # Handle 'remainder' if not 'drop'
    synthetic_df_processed_fallback = pd.DataFrame(synthetic_data_processed_np, columns=processed_feature_names)
    synthetic_df_processed_fallback.to_csv('synthetic_data_processed_fallback_pytorch.csv', index=False)

logging.info("Script finished.")
