# transformer.py
import torch
import torch.nn as nn

class IoTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1, num_classes=8, max_seq_len=128):
        super().__init__()
        self.packet_proj = nn.Linear(input_dim, embed_dim)  # Project packet features to embedding

        # Positional encoding: learnable or sinusoidal
        self.position_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # For flow-level prediction (i.e., after sequence)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, packet_seq):
        # packet_seq: [B, T, F]
        x = self.packet_proj(packet_seq)  # → [B, T, E]
        x = x + self.position_encoding[:x.size(1)].unsqueeze(0)  # → [1, T, E]

        x = self.transformer(x)  # → [B, T, E]

        # Option 1: Mean pooling (for flow-level classification)
        x = x.mean(dim=1)  # [B, E]

        return self.classifier(x)  # → [B, num_classes]

