"""
pointer_head.py — Trajectory Pointer Head for TCAM v2.

At every decoder step where the generated token is <TRK>, the hidden state h_t
is projected and dot-producted against cached track tokens T_tr to produce a
soft distribution α over K tracks.

The same α is used both:
  - at training time, to compute L_grd = KL(target || α)
  - at inference time, to identify which tracks ground the current clause

Architecture:
    W_q : Linear(d_t5=768, d=512)   — project T5 hidden to pointer space
    W_k : Linear(d_model=512, d=512) — project track tokens to pointer space
    pointer logits  p_k = dot(W_q(h_t), W_k(T_tr_k)) / sqrt(d)
    α = softmax(p)                   — (B, K) soft distribution
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_motion_encoder import _CrossAttentionLayer


class TrackPointerDecoder(nn.Module):
    """
    Decoupled pointer over K track tokens (LISA "embedding-as-mask" pattern).

    The T5 decoder cross-attends ONLY to the 64 resampled Z tokens (so the
    language model is never polluted by 1296 distractor track tokens).  The
    hidden state at a <TRK> position is treated as a *query* carrying
    referential intent; this dedicated module resolves it against the full set
    of K enriched track features (H_v[:, :K]) via cross-attention, then scores
    each track with a bilinear readout and produces its OWN softmax over the K
    tracks.  The grounding KL backpropagates through this module into the <TRK>
    hidden state → the T5 decoder (LoRA) → the <TRK> embedding, and into the
    track keys → the vision-motion encoder.

    Why this works where the prior heads did not:
      - Bilinear-only head: a single dot-product could not resolve the query
        against tracks because the frozen query carried no track identity.  The
        cross-attention layers here let the query gather K-wide context before
        scoring (the resolution step the dot-product lacked).
      - Attention-as-pointer: reading the pointer off the LM cross-attention put
        the K tracks inside a 1360-way softmax shared with captioning, which
        starved the grounding gradient and wrecked captioning.  Here the softmax
        is dedicated (exactly K logits, its own loss term) so the full gradient
        mass lands on grounding, and the LM softmax is untouched.

    Parameters
    ----------
    d_t5       : query (T5 decoder hidden) dimension, 768 for flan-t5-base
    d_key      : track-key dimension (H_v[:, :K]), 512
    d_model    : working / scoring dimension, default 512
    num_layers : number of cross-attention layers, default 2
    num_heads  : attention heads, default 8
    d_ff       : FFN width inside each cross-attention layer
    dropout    : dropout (0.0 for deterministic fit tests)
    """

    def __init__(
        self,
        d_t5: int = 768,
        d_key: int = 512,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d_model
        self.q_in = nn.Linear(d_t5, d_model)
        self.kv_in = nn.Linear(d_key, d_model)
        self.layers = nn.ModuleList(
            _CrossAttentionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        # Bilinear readout: q · W · k.  Preserves the per-track identity that a
        # linear probe on the keys proved discriminative (AUC 0.87).
        self.W_bil = nn.Linear(d_model, d_model, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_in.weight)
        nn.init.zeros_(self.q_in.bias)
        nn.init.xavier_uniform_(self.kv_in.weight)
        nn.init.zeros_(self.kv_in.bias)
        nn.init.xavier_uniform_(self.W_bil.weight)

    def forward(
        self,
        h_trk: torch.Tensor,
        keys: torch.Tensor,
        track_valid: torch.Tensor | None = None,
        vis_bias: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute a soft distribution over K tracks for each <TRK> query.

        Parameters
        ----------
        h_trk       : (N, d_t5)        — decoder hidden states at <TRK> positions
        keys        : (N, K, d_key)    — enriched track features for each query
        track_valid : (N, K) bool      — True where the track slot is real
        vis_bias    : (N, K) float     — optional additive logit bias (reserved
                      for v1.1 temporal-window gating; unused in v1)

        Returns
        -------
        alpha : (N, K) float — soft distribution over tracks (sums to 1)
        """
        if h_trk.shape[0] == 0:
            K = keys.shape[1]
            return torch.zeros(0, K, device=h_trk.device, dtype=h_trk.dtype)

        # T5 decoder hidden is fp32 while keys may be fp16 under autocast — cast
        # both onto the module's parameter dtype so the matmuls agree.
        w_dtype = self.q_in.weight.dtype
        h_trk = h_trk.to(w_dtype)
        keys = keys.to(w_dtype)

        q = self.q_in(h_trk).unsqueeze(1)              # (N, 1, d)
        kv = self.kv_in(keys)                          # (N, K, d)

        kpm = None
        if track_valid is not None:
            # PyTorch MultiheadAttention convention: True = ignore.  Guard the
            # degenerate all-masked row (would yield NaN) by keeping it unmasked
            # here; the final softmax NaN-guard returns uniform for such rows.
            kpm = ~track_valid
            kpm = kpm.masked_fill(kpm.all(dim=-1, keepdim=True), False)

        for layer in self.layers:
            q = layer(q, kv, key_padding_mask=kpm)     # (N, 1, d)

        q_scored = self.W_bil(q.squeeze(1))            # (N, d)
        logits = torch.einsum("nd,nkd->nk", q_scored, kv) / math.sqrt(self.d)

        if vis_bias is not None:
            logits = logits + vis_bias.to(logits.dtype)

        if track_valid is not None:
            logits = logits.masked_fill(~track_valid, float("-inf"))

        if return_logits:
            # Raw masked logits — the loss applies per-event temporal gating and
            # hard cross-entropy.  No softmax / NaN-guard here.
            return logits

        alpha = F.softmax(logits, dim=-1)

        # All-masked rows → uniform (mirrors TrajectoryPointerHead guard).
        nan_rows = alpha.isnan().any(dim=-1)
        if nan_rows.any():
            K = keys.shape[1]
            alpha = torch.where(
                nan_rows.unsqueeze(-1),
                torch.full_like(alpha, 1.0 / K),
                alpha,
            )
        return alpha

    def pointer_at_trk_positions(
        self,
        decoder_hidden: torch.Tensor,
        trk_token_id: int,
        token_ids: torch.Tensor,
        keys: torch.Tensor,
        track_valid: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract decoder hidden states at every <TRK> position and compute a
        pointer distribution for each.

        Parameters
        ----------
        decoder_hidden : (B, L, d_t5) — T5 decoder last-layer hidden states
        trk_token_id   : int          — vocabulary ID of <TRK>
        token_ids      : (B, L) int64 — teacher-forced decoder token IDs
        keys           : (B, K, d_key)— enriched track features per sample
        track_valid    : (B, K) bool

        Returns
        -------
        alphas     : (N_trk, K) — pointer distributions, one per <TRK> hit
        batch_idxs : (N_trk,)   — which batch element each pointer belongs to
        """
        mask = token_ids == trk_token_id                # (B, L)
        b_idxs, t_idxs = mask.nonzero(as_tuple=True)

        if b_idxs.numel() == 0:
            K = keys.shape[1]
            return (
                torch.zeros(0, K, device=decoder_hidden.device),
                torch.zeros(0, dtype=torch.long, device=decoder_hidden.device),
            )

        h_trk = decoder_hidden[b_idxs, t_idxs]          # (N_trk, d_t5)
        keys_exp = keys[b_idxs]                          # (N_trk, K, d_key)
        tv_exp = track_valid[b_idxs] if track_valid is not None else None

        out = self.forward(h_trk, keys_exp, tv_exp, return_logits=return_logits)
        return out, b_idxs    # (N_trk, K) logits or alphas, batch idxs

    def pointer_referring(
        self,
        decoder_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        trk_token_id: int,
        cap_begin_id: int | None,
        keys: torch.Tensor,
        track_valid: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        REFormer-style query: instead of the lone <TRK> hidden (a structural,
        near-constant vector that makes the pointer collapse to a prior), the
        query is the MEAN of the decoder hidden states over the event's generated
        clause — the words "a person in white walks left".  The generated caption
        does the pointing; the tracks are the keys.

        For each <TRK> at position t, the clause span is (last <CAP_BEGIN> before
        t, t) — i.e. the timestamp + description tokens, excluding the structural
        <CAP_BEGIN>/<TRK> markers.
        """
        mask = token_ids == trk_token_id                 # (B, L)
        b_idxs, t_idxs = mask.nonzero(as_tuple=True)
        if b_idxs.numel() == 0:
            Kk = keys.shape[1]
            return (
                torch.zeros(0, Kk, device=decoder_hidden.device),
                torch.zeros(0, dtype=torch.long, device=decoder_hidden.device),
            )

        queries = []
        for b, t in zip(b_idxs.tolist(), t_idxs.tolist()):
            s = 0
            if cap_begin_id is not None:
                prev = (token_ids[b, :t] == cap_begin_id).nonzero(as_tuple=False)
                if prev.numel() > 0:
                    s = int(prev[-1].item()) + 1          # token after <CAP_BEGIN>
            span = decoder_hidden[b, s:t]                 # clause words (excl <TRK>)
            q = span.mean(dim=0) if span.shape[0] > 0 else decoder_hidden[b, t]
            queries.append(q)
        h = torch.stack(queries, dim=0)                   # (N_trk, d_t5)

        keys_exp = keys[b_idxs]
        tv_exp = track_valid[b_idxs] if track_valid is not None else None
        out = self.forward(h, keys_exp, tv_exp, return_logits=return_logits)
        return out, b_idxs


class TrajectoryPointerHead(nn.Module):
    """
    Soft pointer from a T5 decoder hidden state to a set of K track tokens.

    Parameters
    ----------
    d_t5    : T5 decoder hidden dimension (768 for flan-t5-base)
    d_model : track-token / pointer space dimension (default 512)
    """

    def __init__(self, d_t5: int = 768, d_model: int = 512) -> None:
        super().__init__()
        self.d = d_model
        self.W_q = nn.Linear(d_t5, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

    def forward(
        self,
        h_t: torch.Tensor,
        T_tr: torch.Tensor,
        track_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute a soft attention distribution over track tokens.

        Parameters
        ----------
        h_t         : (B, d_t5)    — T5 decoder hidden state at a <TRK> position
        T_tr        : (B, K, d_model) — track token embeddings (from tokenizer
                      or cached from vision-motion encoder)
        track_valid : (B, K) bool   — True where track slot is real (not padding)

        Returns
        -------
        alpha : (B, K) float32 — soft distribution over tracks (sums to 1)
        """
        q = self.W_q(h_t)                             # (B, d)
        k = self.W_k(T_tr)                            # (B, K, d)

        logits = torch.einsum("bd,bkd->bk", q, k) / math.sqrt(self.d)  # (B, K)

        # Mask padding track slots to -inf before softmax
        if track_valid is not None:
            logits = logits.masked_fill(~track_valid, float("-inf"))

        alpha = F.softmax(logits, dim=-1)

        # If all positions were masked (no valid tracks), return uniform
        nan_rows = alpha.isnan().any(dim=-1)
        if nan_rows.any():
            K = T_tr.shape[1]
            alpha = torch.where(
                nan_rows.unsqueeze(-1),
                torch.full_like(alpha, 1.0 / K),
                alpha,
            )

        return alpha

    def pointer_at_trk_positions(
        self,
        hidden_states: torch.Tensor,
        trk_token_id: int,
        token_ids: torch.Tensor,
        T_tr: torch.Tensor,
        track_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: given the full decoder hidden-state sequence and the
        generated token-id sequence, extract hidden states at <TRK> positions
        and compute pointer distributions only there.

        Parameters
        ----------
        hidden_states : (B, L, d_t5) — decoder output hidden states
        trk_token_id  : int — vocabulary ID of the <TRK> token
        token_ids     : (B, L) int64 — generated token IDs (from teacher-forcing
                        or autoregressive decoding)
        T_tr          : (B, K, d_model)
        track_valid   : (B, K) bool

        Returns
        -------
        alphas      : (N_trk, K)  — pointer distributions for each <TRK> hit
        batch_idxs  : (N_trk,)   — which batch element each pointer belongs to
        """
        # find (batch, time) indices of all <TRK> tokens
        mask = token_ids == trk_token_id             # (B, L)
        b_idxs, t_idxs = mask.nonzero(as_tuple=True)

        if b_idxs.numel() == 0:
            # No <TRK> tokens in batch — return empty tensors
            K = T_tr.shape[1]
            return (
                torch.zeros(0, K, device=hidden_states.device),
                torch.zeros(0, dtype=torch.long, device=hidden_states.device),
            )

        h_trk = hidden_states[b_idxs, t_idxs]       # (N_trk, d_t5)
        T_tr_exp = T_tr[b_idxs]                      # (N_trk, K, d_model)
        tv_exp = track_valid[b_idxs] if track_valid is not None else None

        alphas = self.forward(h_trk, T_tr_exp, tv_exp)  # (N_trk, K)
        return alphas, b_idxs
