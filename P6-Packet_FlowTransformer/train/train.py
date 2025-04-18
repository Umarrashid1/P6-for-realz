from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score  # 👈 added f1_score

def train_model(model, train_dataset, val_dataset, epochs=3, batch_size=64, lr=1e-5, device='cuda'):
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

        for batch_idx, batch in enumerate(train_loader):
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)

            # ✅ Only check the first batch for debugging
            if epoch == 0 and batch_idx == 0:
                print("\n📊 Debug: First training batch stats")
                print("NaNs in numerical:", torch.isnan(numerical).sum().item())
                print("Infs in numerical:", torch.isinf(numerical).sum().item())
                print("Numerical max:", numerical.max().item())
                print("Numerical min:", numerical.min().item())
                print("Unique labels:", torch.unique(labels))

            optimizer.zero_grad()
            outputs = model(numerical, categorical)
            loss = criterion(outputs, labels)
            loss.backward()

            # ✅ Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

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
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - "
              f"Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} - "
              f"Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

        torch.save(model.state_dict(), "iot_transformer_complete.pt")
        print("Saved pretrained model for finetuning.")


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
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 👈 F1-score added

    print(f"Test Accuracy: {acc:.4f} - Test F1: {f1:.4f}")
