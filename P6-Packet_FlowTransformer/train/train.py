from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=64, lr=1e-3, device='cuda'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # === Training ===
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
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

        train_acc = accuracy_score(all_labels, all_preds)

        # === Validation ===
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                numerical = batch['numerical'].to(device)
                categorical = batch['categorical'].to(device)
                labels = batch['label'].to(device)

                outputs = model(numerical, categorical)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")


def test_model(model, test_dataset, batch_size=64, device='cuda'):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)

            outputs = model(numerical, categorical)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
