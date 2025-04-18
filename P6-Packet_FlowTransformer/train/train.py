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
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            # 🔍 Inspect & fix batch only on the first iteration
            if epoch == 0 and batch_idx == 0:
                batch = inspect_and_fix_batch(batch, model, device=device, fix=True)

            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(numerical, categorical)

            # Check for NaNs/Infs in model output
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("❌ Detected NaNs or Infs in model output!")
                print("Logits min:", outputs.min().item())
                print("Logits max:", outputs.max().item())
                print("Sample logits:", outputs[0])
                return  # ⛔ Stop training if outputs are broken

            loss = criterion(outputs, labels)
            loss.backward()
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



def inspect_and_fix_batch(batch, model, device='cuda', fix=True):
    numerical = batch['numerical'].to(device)
    categorical = batch['categorical'].to(device)
    labels = batch['label'].to(device)

    print("\n🔎 Inspecting batch:")

    # === Check numerical values ===
    print("→ Numerical:")
    print("  NaNs:", torch.isnan(numerical).sum().item())
    print("  Infs:", torch.isinf(numerical).sum().item())
    print("  Min value:", numerical.min().item())
    print("  Max value:", numerical.max().item())
    print("  Row max values (first 5):", numerical.max(dim=1).values[:5])

    # === Check categorical indices ===
    print("→ Categorical:")
    num_cat = categorical.shape[1]
    for i in range(num_cat):
        max_idx = categorical[:, i].max().item()
        min_idx = categorical[:, i].min().item()
        emb_size = model.cat_embeddings[i].num_embeddings if hasattr(model, 'cat_embeddings') else '?'
        print(f"  Cat[{i}] min={min_idx} max={max_idx} | embedding size={emb_size}")

        if fix and max_idx >= emb_size:
            print(f"    ⚠️ Invalid index in Cat[{i}] — fixing by clamping.")
            categorical[:, i] = categorical[:, i].clamp(0, emb_size - 1)

        if fix and min_idx < 0:
            print(f"    ⚠️ Negative index in Cat[{i}] — fixing to 0.")
            categorical[:, i] = torch.where(categorical[:, i] < 0, torch.zeros_like(categorical[:, i]), categorical[:, i])

    return {
        'numerical': numerical,
        'categorical': categorical,
        'label': labels
    }

