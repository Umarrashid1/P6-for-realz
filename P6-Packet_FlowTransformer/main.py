# main.py
from pipeline.iot_dataset import IoTDataset
from models.transformer import IoTTransformer
from train.train import train_model
from pipeline.config import numerical_columns, categorical_columns
from torch.utils.data import Subset



dataset = IoTDataset("../../dataset/packet.pt")
categorical_tensor = dataset["categorical"]

cat_cardinalities = [int(torch.max(categorical_tensor[:, i]) + 1) for i in range(categorical_tensor.shape[1])]


model = IoTTransformer(
    num_numerical=len(numerical_columns),
    cat_cardinalities=cat_cardinalities,
    num_classes=8
)


subset_indices = list(range(10000))
subset = Subset(dataset, subset_indices)

train_model(model, subset, epochs=3)
