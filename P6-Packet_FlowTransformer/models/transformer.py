import torch
import torch.nn as nn
from typing import Dict

class IoTTransformer(nn.Module):
    """Transformer model that consumes:
        - `packet_seq`   : [B, T, F_num] numeric features
        - `cat_feats`    : Dict[col, LongTensor [B, T]] categorical IDs
        - `attention_mask`: [B, T] float / int with 1 = valid, 0 = pad
    The model embeds each categorical column separately and **adds** those
    embeddings onto the projected numeric stream.
    """

    def __init__(
        self,
        input_dim: int,
        cat_sizes: Dict[str, int],
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 8,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()

        # ❶ Numeric projection --------------------------------------------------
        self.packet_proj = nn.Linear(input_dim, embed_dim)

        # ❷ Positional encoding (learnable) ------------------------------------
        self.position_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        # ❸ Column‑wise embeddings --------------------------------------------
        self.cat_embeds = nn.ModuleDict({
            col: nn.Embedding(vocab, embed_dim) for col, vocab in cat_sizes.items()
        })

        # ❹ Transformer encoder -------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ❺ Classification head -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 130),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(130, num_classes),
        )

    # -------------------------------------------------------------------------
    def forward(
        self,
        packet_seq: torch.Tensor,   # [B, T, F_num]
        cat_feats: Dict[str, torch.Tensor],  # {col: [B, T] Long}
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Numeric projection ----------------------------------------------------
        x = self.packet_proj(packet_seq)  # [B, T, E]

        # Add column embeddings -------------------------------------------------
        for col, emb_layer in self.cat_embeds.items():
            x = x + emb_layer(cat_feats[col])  # broadcast add

        # Positional encoding ---------------------------------------------------
        x = x + self.position_encoding[: x.size(1)].unsqueeze(0)

        # Build key‑padding mask (True = pad) -----------------------------------
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # Transformer -----------------------------------------------------------
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, E]
        x = x.mean(dim=1)  # Simple global average pooling

        return self.classifier(x)
