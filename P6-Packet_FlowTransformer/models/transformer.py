import torch
import torch.nn as nn

class IoTTransformer(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, num_classes, embed_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=embed_dim)
            for card in cat_cardinalities
        ])

        self.position_enc = nn.Parameter(torch.randn(len(cat_cardinalities), embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * len(cat_cardinalities) + num_numerical, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, numerical, categorical):
        embedded = [embed(categorical[:, i]) + self.position_enc[i] for i, embed in enumerate(self.cat_embeddings)]
        cat_seq = torch.stack(embedded, dim=1)
        transformer_out = self.transformer(cat_seq)
        flat_cat = transformer_out.flatten(start_dim=1)
        x = torch.cat([flat_cat, numerical], dim=1)
        return self.mlp(x)
