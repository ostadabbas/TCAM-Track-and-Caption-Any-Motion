"""
resampler.py — Caption-Aware Resampler (Q-Former-lite) for TCAM v2.

Compresses the joint vision-motion latent H_v into a fixed set of N_q=64
learned query vectors Z that are then prepended to the FLAN-T5 encoder input.

Architecture:
    4 × (CrossAttention(Q=Q_lat, KV=H_v) → LayerNorm → FFN → LayerNorm)
    Linear projection d_model → d_t5 (maps to T5 encoder hidden size)

Parameters ~ 30M (64 queries × 4 layers × d=512).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        queries  : (B, N_q, d)
        context  : (B, N_ctx, d)
        → (B, N_q, d)
        """
        attn_out, _ = self.cross_attn(
            queries, context, context,
            key_padding_mask=context_mask,
        )
        queries = self.norm1(queries + self.drop(attn_out))
        queries = self.norm2(queries + self.drop(self.ff(queries)))
        return queries


class CaptionAwareResampler(nn.Module):
    """
    Q-Former-lite: 64 learnable queries cross-attend to H_v over 4 layers.

    Parameters
    ----------
    d_model   : dimension of H_v (= joint encoder output, default 512)
    d_t5      : FLAN-T5-base encoder hidden size (512 for base, 768 for large)
    num_queries: number of learnable query vectors (default 64)
    num_layers : number of cross-attention blocks (default 4)
    num_heads  : attention heads
    d_ff       : feed-forward hidden size
    dropout    : dropout rate
    use_gradient_checkpointing : checkpoint each layer during training
    """

    def __init__(
        self,
        d_model: int = 512,
        d_t5: int = 512,
        num_queries: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_gc = use_gradient_checkpointing

        # Learned query vectors: initialised from N(0, 0.02) à la BERT
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        self.layers = nn.ModuleList([
            _CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Project to T5 encoder dimension; small init avoids FP16 overflow at
        # the start of training when the resampler queries are random.
        self.proj = nn.Linear(d_model, d_t5)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(d_t5)

    def forward(
        self,
        H_v: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        H_v          : (B, N, d_model)  — joint vision-motion latent
        context_mask : (B, N) bool  — True = ignore position (padding)

        Returns
        -------
        Z : (B, num_queries, d_t5)  — compact summary for FLAN-T5 encoder
        """
        B = H_v.shape[0]
        Q = self.queries.expand(B, -1, -1)  # (B, N_q, d_model)

        for layer in self.layers:
            if self.use_gc and self.training:
                Q = ckpt.checkpoint(layer, Q, H_v, context_mask, use_reentrant=False)
            else:
                Q = layer(Q, H_v, context_mask)

        Z = self.norm(self.proj(Q))  # (B, N_q, d_t5)
        return Z
