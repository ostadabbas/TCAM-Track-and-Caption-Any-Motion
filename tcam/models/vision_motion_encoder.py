"""
vision_motion_encoder.py — Vision-Motion Joint Encoder for TCAM v2.

Two-stage transformer fusion:

  Stage A (3 × track-as-query cross-attention layers):
    Each track token gathers visual context from pooled ViT patch tokens.
    Q = T_tr  (B, K, d)
    K = V = P_pool  (B, N̂, d)   where N̂ = T · H_pool · W_pool

  Stage B (3 × joint self-attention layers):
    Track tokens and patch tokens attend to each other jointly.
    Input: concat([T_tr, P_pool], dim=1)  →  (B, K+N̂, d)

Output:
    H_v  ∈  R^{B × (K + N̂) × d}   — joint vision-motion latent

Gradient checkpointing is supported via nn.utils.checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


class _CrossAttentionLayer(nn.Module):
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
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query           : (B, Q, d)
        key_value       : (B, N, d)
        key_padding_mask: (B, N) bool — True for positions to ignore

        Returns
        -------
        (B, Q, d)
        """
        attn_out, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query + self.drop(attn_out))
        query = self.norm2(query + self.drop(self.ff(query)))
        return query


class _SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
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
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, N, d) → (B, N, d)"""
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class VisionMotionEncoder(nn.Module):
    """
    Fusion encoder combining track tokens and ViT patch tokens.

    Parameters
    ----------
    d_model              : shared embedding dimension (must match ViT proj)
    d_vit                : ViT-B/16 output dimension (768)
    num_cross_attn_layers: Stage A depth (default 3)
    num_joint_attn_layers: Stage B depth (default 3)
    num_heads            : attention heads
    d_ff                 : feed-forward hidden size
    dropout              : attention + FFN dropout
    patch_pool_stride    : average-pool ViT patch grid by this factor before
                           concatenation to keep token count manageable
    use_gradient_checkpointing : whether to checkpoint each layer
    """

    def __init__(
        self,
        d_model: int = 512,
        d_vit: int = 768,
        num_cross_attn_layers: int = 3,
        num_joint_attn_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        patch_pool_stride: int = 2,
        use_gradient_checkpointing: bool = False,
        num_frames: int = 16,
        use_temporal_pos: bool = False,
    ) -> None:
        super().__init__()
        self.use_gc = use_gradient_checkpointing
        self.pool_stride = patch_pool_stride

        # Project ViT dim → d_model
        self.patch_proj = nn.Linear(d_vit, d_model)

        # Per-frame temporal position embedding. The patch tokens are otherwise
        # flattened across all frames into a permutation-invariant bag, so the
        # model cannot tell WHICH frame a patch came from — making temporal
        # localization (the [f_s-f_e] windows) impossible. This learned per-frame
        # embedding gives each patch its frame identity. Zero-init so it starts as
        # a no-op (safe to resume a model trained without it) and learns from there.
        self.use_temporal_pos = use_temporal_pos
        if use_temporal_pos:
            self.temporal_pos = nn.Parameter(torch.zeros(num_frames, d_model))

        # Stage A: track tokens query patch tokens
        self.cross_layers = nn.ModuleList([
            _CrossAttentionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # Stage B: joint self-attention
        self.joint_layers = nn.ModuleList([
            _SelfAttentionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_joint_attn_layers)
        ])

    def forward(
        self,
        T_tr: torch.Tensor,
        P: torch.Tensor,
        track_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        T_tr        : (B, K, d_model)  — track tokens from TrajectoryTokenizer
        P           : (B, T, N_p, d_vit)  — ViT patch tokens
        track_valid : (B, K) bool  — True for real tracks; used as key mask

        Returns
        -------
        H_v : (B, K + N̂, d_model)
        """
        B, K, d_model = T_tr.shape
        _B, Tf, N_p, d_vit = P.shape

        # ── project & pool patch tokens ───────────────────────────────────────
        # P: (B, T, N_p, d_vit) → project → (B, T, N_p, d_model) → flatten T
        P_proj = self.patch_proj(P)                         # (B, T, N_p, d_model)

        # Add per-frame temporal position so patches carry frame identity (enables
        # temporal localization). Broadcast (T, d) over batch and patch dims.
        if self.use_temporal_pos:
            P_proj = P_proj + self.temporal_pos[:Tf].view(1, Tf, 1, d_model)

        # spatial pool if stride > 1 (approximate: pool across N_p tokens)
        if self.pool_stride > 1:
            # treat N_p dimension as 1D sequence and avg-pool
            # (B, T, N_p, d_model) → (B*T, d_model, N_p) → pool → (B*T, d_model, N̂)
            BT = B * Tf
            Pp = P_proj.reshape(BT, N_p, d_model).permute(0, 2, 1)  # (BT, d, N_p)
            Pp = torch.nn.functional.avg_pool1d(Pp, kernel_size=self.pool_stride,
                                                stride=self.pool_stride)
            N_hat = Pp.shape[-1]
            Pp = Pp.permute(0, 2, 1).reshape(B, Tf * N_hat, d_model)  # (B, T·N̂, d)
        else:
            Pp = P_proj.reshape(B, Tf * N_p, d_model)      # (B, T·N_p, d_model)

        # ── Stage A: cross-attention (track queries patch tokens) ─────────────
        H_tr = T_tr
        for layer in self.cross_layers:
            if self.use_gc and self.training:
                H_tr = ckpt.checkpoint(layer, H_tr, Pp, None, use_reentrant=False)
            else:
                H_tr = layer(H_tr, Pp)

        # ── Stage B: joint self-attention ─────────────────────────────────────
        # Build padding mask: False = attend, True = ignore
        # Track positions that are padding (track_valid=False) should be masked.
        if track_valid is not None:
            # (B, K) → (B, K + N̂): tracks may be padded, patches are always valid
            patch_valid = torch.ones(B, Pp.shape[1], dtype=torch.bool, device=T_tr.device)
            joint_valid = torch.cat([track_valid, patch_valid], dim=1)  # (B, K+N̂)
            key_padding_mask = ~joint_valid   # True = ignore (PyTorch convention)
        else:
            key_padding_mask = None

        H_joint = torch.cat([H_tr, Pp], dim=1)             # (B, K+N̂, d)
        for layer in self.joint_layers:
            if self.use_gc and self.training:
                H_joint = ckpt.checkpoint(
                    layer, H_joint, key_padding_mask, use_reentrant=False
                )
            else:
                H_joint = layer(H_joint, key_padding_mask)

        return H_joint  # (B, K+N̂, d_model)
