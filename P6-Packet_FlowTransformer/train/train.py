import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from pipeline.config import categorical_columns


def train_model(model, train_dataset, val_dataset, epochs=3, batch_size=64, lr=1e-3, device='cuda'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            packet_seq = batch['packet_seq'].to(device)  # [B, T, F]
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # gather categorical features dict
            cat_feats = {col: batch[col].to(device) for col in categorical_columns}

            optimizer.zero_grad()
            outputs = model(packet_seq, cat_feats, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                packet_seq = batch['packet_seq'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                cat_feats = {col: batch[col].to(device) for col in categorical_columns}
                outputs = model(packet_seq, cat_feats, attention_mask=attention_mask)

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        torch.save(model.state_dict(), "iot_transformer_pretrained.pt")
        print("Saved pretrained model for finetuning.")



def test_model(model, test_dataset, batch_size=64, device='cuda'):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            packet_seq = batch['packet_seq'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            cat_feats = {col: batch[col].to(device) for col in categorical_columns}
            outputs = model(packet_seq, cat_feats, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Classification report (precision, recall, f1-score per class)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Optional: Class-wise accuracy
    cm_diag = np.diag(cm)
    class_counts = cm.sum(axis=1)
    classwise_acc = cm_diag / class_counts
    for i, acc in enumerate(classwise_acc):
        print(f"Class {i} Accuracy: {acc:.4f}")
