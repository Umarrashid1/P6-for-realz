from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_model(model, dataset, epochs=10, batch_size=64, lr=1e-3, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in dataloader:
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(numerical, categorical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")
