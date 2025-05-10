import torch
import torch.nn as nn
from typing import Dict

class IoTTransformer(nn.Module):
    """Transformer model that consumes:
        - `packet_seq`    : FloatTensor [B, T, F_num]  (numeric features)
        - `cat_feats`     : Dict[col, LongTensor [B, T]] (categorical IDs)
        - `attention_mask`: FloatTensor / LongTensor [B, T] with 1 = valid, 0 = pad

    Each categorical column is embedded separately and **added** to the
    projected numeric stream before feeding the sequence to the Transformer.
    """

    def __init__(
        self,
        input_dim: int,
        cat_sizes: Dict[str, int],
        cat_padding_idx: Dict[str, int],
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 8,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()

        # ── 1. Numeric projection ────────────────────────────────────────────
        self.packet_proj = nn.Linear(input_dim, embed_dim)

        # ── 2. Learnable positional encodings ───────────────────────────────
        # Shape: [1, T, E] so broadcasting over batch is trivial
        self.position_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # ── 3. Column‑wise embeddings with explicit padding_idx ─────────────
        self.cat_embeds = nn.ModuleDict(
            {
                col: nn.Embedding(
                    num_embeddings=cat_sizes[col],
                    embedding_dim=embed_dim,
                    padding_idx=cat_padding_idx[col],
                )
                for col in cat_sizes
            }
        )

        # ── 4. Transformer encoder ──────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 5. Classification head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 130),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(130, num_classes),
        )

    # ────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        packet_seq: torch.Tensor,               # [B, T, F_num]
        cat_feats: Dict[str, torch.Tensor],     # {col: [B, T] Long}
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits [B, num_classes]."""

        # 1. Project numeric features to embedding space
        x = self.packet_proj(packet_seq)  # [B, T, E]

        # 2. Add categorical embeddings (with safety clamp)
        for col, emb_layer in self.cat_embeds.items():
            cat_input = cat_feats[col].clamp(0, emb_layer.num_embeddings - 1)
            x = x + emb_layer(cat_input)  # broadcast add

        # 3. Add positional encoding
        x = x + self.position_encoding[:, : x.size(1)]

        # 4. Build key‑padding mask (True = pad)
        key_pad_mask = (attention_mask == 0) if attention_mask is not None else None

        # 5. Transformer encoder
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)  # [B, T, E]

        # 6. Global average pooling (mask-aware if attention_mask given)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, E]
        else:
            x = x.mean(dim=1)  # [B, E]

        # 7. Classification head
        return self.classifier(x)
