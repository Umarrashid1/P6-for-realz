# models/transformer.py
import torch
import torch.nn as nn


class IoTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1, num_classes=8, max_seq_len=128,
                 num_categories={}):
        super().__init__()

        self.packet_proj = nn.Linear(input_dim, embed_dim)  # Project packet features to embedding

        # Positional encoding: learnable or sinusoidal
        self.position_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            cat: nn.Embedding(num_categories[cat], embed_dim) for cat in num_categories
        })

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # For flow-level prediction
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 130),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(130, num_classes)
        )

    def forward(self, packet_seq, categorical_data, attention_mask=None):
        # packet_seq: [B, T, F] -> Numerical features sequence
        x = self.packet_proj(packet_seq)  # Project numerical features to embedding

        # Add embeddings for categorical features
        for i, cat in enumerate(categorical_data):  # categorical_data is a list of categorical features
            emb = self.embeddings[cat](categorical_data[cat])  # Get the embedding for each category
            x = x + emb  # Add categorical feature embeddings to the numerical features

        # Add positional encoding
        x = x + self.position_encoding[:x.size(1)].unsqueeze(0)  # [1, T, E] → broadcast

        if attention_mask is not None:
            attn_mask = ~attention_mask.bool()  # invert: pad=1 → False, valid=1 → True
        else:
            attn_mask = None

        # Transformer Encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, T, E]
        x = x.mean(dim=1)  # Mean pooling over sequence length T

        return self.classifier(x)  # Final classification


