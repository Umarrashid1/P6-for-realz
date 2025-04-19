# models/transformer.py
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

    def forward(self, packet_seq, attention_mask=None):
        # packet_seq: [B, T, F]
        x = self.packet_proj(packet_seq)  # → [B, T, E]
        x = x + self.position_encoding[:x.size(1)].unsqueeze(0)  # [1, T, E] → broadcast

        if attention_mask is not None:
            # Transformer expects mask shape [B, T] → convert to [B, 1, 1, T] or [B, T] with batch_first=True
            # In PyTorch, True = keep, False = mask
            attn_mask = ~attention_mask.bool()  # invert: pad=1 → False, valid=1 → True
        else:
            attn_mask = None

        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, T, E]
        x = x.mean(dim=1)  # mean pooling over T

        return self.classifier(x)  # [B, num_classes]


