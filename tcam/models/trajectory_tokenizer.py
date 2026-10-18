"""
trajectory_tokenizer.py — φ_tr: map per-track trajectory data to token vectors.

For each track k:
    τ_k = MLP_pos(γ(X_k)) + MLP_vis(V_k) + MLP_app(pool(P[X_k]))

where:
    γ(·) is a 2D sinusoidal Fourier embedding (L frequencies per axis)
    X_k ∈ R^{T×2}  — normalised (x,y) trajectory across frames
    V_k ∈ R^T      — binary visibility per frame
    P[X_k]         — ViT patch tokens bilinearly sampled at X_k's coordinates,
                     mean-pooled over visible frames

Output: T_tr ∈ R^{B × K × d_model}
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalFourierEmbedding(nn.Module):
    """
    2D sinusoidal Fourier embedding for normalised (x, y) coordinates.

    For each of L frequencies f_i = 2^i:
        embed(p) = [sin(f_0 π p_x), cos(f_0 π p_x), ...,
                    sin(f_{L-1} π p_x), cos(f_{L-1} π p_x),
                    sin(f_0 π p_y), cos(f_0 π p_y), ...,
                    sin(f_{L-1} π p_y), cos(f_{L-1} π p_y)]

    Output dimension: 4L (2 axes × L freqs × 2 trig funcs)
    """

    def __init__(self, num_freqs: int = 8) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        # frequencies: 2^0, 2^1, ..., 2^{L-1}
        freqs = torch.pow(2.0, torch.arange(num_freqs).float())  # (L,)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 4 * self.num_freqs  # 2 axes × L × 2 functions

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (..., 2) float32 — normalised (x, y) in [0, 1]

        Returns
        -------
        embed : (..., 4L)
        """
        x = coords[..., 0:1]   # (..., 1)
        y = coords[..., 1:2]   # (..., 1)

        # (..., 1) × (L,) → (..., L) via broadcast
        fx = math.pi * x * self.freqs                    # (..., L)
        fy = math.pi * y * self.freqs                    # (..., L)

        embed = torch.cat(
            [fx.sin(), fx.cos(), fy.sin(), fy.cos()], dim=-1
        )  # (..., 4L)
        return embed


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class TrajectoryTokenizer(nn.Module):
    """
    Map per-track trajectory tensors to d_model-dimensional token vectors.

    Parameters
    ----------
    d_model      : output token dimension (default 512)
    d_vis        : ViT patch-token dimension (ViT-B/16 → 768)
    num_frames   : T
    num_freqs    : L for sinusoidal embedding (4L-dim output)
    mlp_hidden   : hidden dim inside the three MLPs
    """

    def __init__(
        self,
        d_model: int = 512,
        d_vis: int = 768,
        num_frames: int = 8,
        num_freqs: int = 8,
        mlp_hidden: int = 256,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.T = num_frames

        self.fourier = SinusoidalFourierEmbedding(num_freqs)
        pos_in_dim = self.fourier.out_dim * num_frames  # 4L·T (flattened trajectory)

        self.mlp_pos = _mlp(pos_in_dim, mlp_hidden, d_model)
        self.mlp_vis = _mlp(num_frames, mlp_hidden, d_model)
        self.mlp_app = _mlp(d_vis, mlp_hidden, d_model)

        # E6b: camera-motion residual term. Same featurisation as mlp_pos (the
        # residual is also (B,T,K,2)), but the output layer is zero-initialised so
        # a model resumed from run_028 starts identical to it.
        self.use_residual = use_residual
        self.res_scale = float(os.environ.get("TCAM_RES_SCALE", "50.0"))
        self.mlp_res = _mlp(pos_in_dim, mlp_hidden, d_model) if use_residual else None
        if self.mlp_res is not None:
            nn.init.zeros_(self.mlp_res[-1].weight)
            nn.init.zeros_(self.mlp_res[-1].bias)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        P: torch.Tensor,
        X_res: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (B, T, K, 2) float32 — normalised (x,y) trajectories
        X_res : (B, T, K, 2) or None — camera-compensated residual displacement
        V : (B, T, K)    float32 — visibility (0/1 per frame per track)
        P : (B, T, N_p, d_vis)  — ViT patch tokens

        Returns
        -------
        T_tr : (B, K, d_model)
        """
        B, T, K, _ = X.shape

        # ── positional embedding ──────────────────────────────────────────────
        # X_flat: (B, K, T, 2) → fourier → (B, K, T, 4L) → flatten T → (B, K, T·4L)
        X_perm = X.permute(0, 2, 1, 3)                          # (B, K, T, 2)
        pos_emb = self.fourier(X_perm)                           # (B, K, T, 4L)
        pos_flat = pos_emb.reshape(B, K, -1)                     # (B, K, T·4L)
        feat_pos = self.mlp_pos(pos_flat)                        # (B, K, d_model)

        # ── visibility embedding ──────────────────────────────────────────────
        V_perm = V.permute(0, 2, 1)                              # (B, K, T)
        feat_vis = self.mlp_vis(V_perm)                          # (B, K, d_model)

        # ── appearance embedding via bilinear sampling of patch tokens ────────
        # For each track k at each visible frame t, sample the patch token
        # nearest to X[b,t,k] and mean-pool over visible frames.
        feat_app = self._sample_appearance(X, V, P)              # (B, K, d_model)

        # ── camera-motion residual (E6b) ──────────────────────────────────────
        feat_res = 0.0
        if self.mlp_res is not None and X_res is not None:
            # Residuals have std ~0.014 while the Fourier frequencies are tuned for
            # coordinates spanning [0,1]; unscaled, 25% of the embedding channels
            # sit at ~0 and carry no signal (vs 3% for absolute coords). Scaling
            # into the same dynamic range restores them.
            r_perm = X_res.permute(0, 2, 1, 3) * self.res_scale   # (B, K, T, 2)
            r_flat = self.fourier(r_perm).reshape(B, K, -1)       # (B, K, T·4L)
            feat_res = self.mlp_res(r_flat)                       # (B, K, d_model)

        # ── combine ───────────────────────────────────────────────────────────
        T_tr = self.norm(feat_pos + feat_vis + feat_app + feat_res)
        return T_tr

    def _sample_appearance(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        P: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each track k, bilinearly sample P at the track's (x,y) coordinates
        across frames and mean-pool over visible frames.

        P is expected in shape (B, T, N_p, d_vis) where N_p = H_p * W_p patch tokens
        arranged in raster order.  We infer H_p = W_p = sqrt(N_p).

        Returns (B, K, d_model).
        """
        B, T, N_p, d_vis = P.shape
        H_p = W_p = int(math.isqrt(N_p))
        if H_p * W_p != N_p:
            # Non-square grid — use nearest-patch lookup instead
            H_p = N_p
            W_p = 1

        K = X.shape[2]

        # Reshape P to (B*T, d_vis, H_p, W_p) for grid_sample
        P_flat = P.reshape(B * T, H_p, W_p, d_vis).permute(0, 3, 1, 2)  # (BT, d, H, W)

        # Build sampling grid: X has (x,y) in [0,1]; grid_sample expects [-1,1]
        # X: (B, T, K, 2) → (B*T, K, 2) → (B*T, K, 1, 2)
        grid_xy = X.reshape(B * T, K, 2)             # (BT, K, 2)
        grid_xy = grid_xy * 2.0 - 1.0               # map [0,1] → [-1,1]
        grid = grid_xy.unsqueeze(2)                  # (BT, K, 1, 2)

        # grid_sample: input (BT, d, H, W), grid (BT, K, 1, 2) → (BT, d, K, 1)
        sampled = F.grid_sample(
            P_flat, grid,
            mode="bilinear", padding_mode="border", align_corners=True
        )  # (BT, d_vis, K, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (BT, K, d_vis)
        sampled = sampled.reshape(B, T, K, d_vis)       # (B, T, K, d_vis)

        # Mask invisible frames and mean-pool
        V_mask = V.unsqueeze(-1)                         # (B, T, K, 1)
        visible_sum = (sampled * V_mask).sum(dim=1)      # (B, K, d_vis)
        count = V_mask.sum(dim=1).clamp(min=1.0)         # (B, K, 1)
        app_vec = visible_sum / count                    # (B, K, d_vis)

        return self.mlp_app(app_vec)                     # (B, K, d_model)
