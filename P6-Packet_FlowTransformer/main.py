# main.py
from pipeline.iot_dataset import IoTDataset
from models.transformer import IoTTransformer
from train.train import train_model
from pipeline.config import numerical_columns, categorical_columns


dataset = IoTDataset("../../dataset/packet.pt")
model = IoTTransformer(
    num_numerical=len(numerical_columns),
    num_categorical=len(categorical_columns),
    num_classes=8
)

train_model(model, dataset)
