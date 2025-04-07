import torch
import torch.nn as nn

class IoTTransformer(nn.Module):
    def __init__(self, num_numerical, num_categorical, num_classes, cat_cardinality=1000, embed_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super(IoTTransformer, self).__init__()

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cat_cardinality, embedding_dim=embed_dim)
            for _ in range(num_categorical)
        ])

        # Positional encoding for sequence of embeddings
        self.position_enc = nn.Parameter(torch.randn(num_categorical, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection (Transformer output + numerical features → classifier)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * num_categorical + num_numerical, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, numerical, categorical):
        # Embed categorical features and add positional encodings
        embedded = [embed(categorical[:, i]) + self.position_enc[i] for i, embed in enumerate(self.cat_embeddings)]
        cat_seq = torch.stack(embedded, dim=1)  # Shape: [B, T, E]

        # Transformer
        transformer_out = self.transformer(cat_seq)  # Shape: [B, T, E]
        flat_cat = transformer_out.flatten(start_dim=1)  # [B, T * E]

        # Concatenate with normalized numerical features
        x = torch.cat([flat_cat, numerical], dim=1)
        out = self.mlp(x)
        return out
