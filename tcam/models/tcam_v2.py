"""
tcam_v2.py — Top-level TCAM v2 nn.Module.

Data flow
---------
  Video frames  → ViT-B/16 (frozen)           → P  (B, T, N_p, 768)
  Trajectories  → CoTracker3 (frozen/offline)  → X  (B, T, K, 2),  V (B, T, K)
  X, V, P       → TrajectoryTokenizer          → T_tr (B, K, 512)
  T_tr, P       → VisionMotionEncoder          → H_v  (B, K+N̂, 512)
  H_v           → CaptionAwareResampler        → Z    (B, 64, 512)
  Z prepended to T5 encoder context
  T5 decoder runs teacher-forcing / AR decoding
  At <TRK> positions → TrajectoryPointerHead   → α    (B, K)

Training losses (computed externally in losses.py, forwarded here for
architecture integration):
  L_cap  : label-smoothed CE on decoder token predictions
  L_grd  : KL(target_α || predicted_α) at <TRK> positions
  L_div  : pairwise cosine penalty across clause-level pointer distributions
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# HuggingFace imports are deferred to __init__ to allow module-level import
# without requiring transformers to be installed everywhere.
try:
    from transformers import T5ForConditionalGeneration, T5Config
    from transformers.modeling_outputs import BaseModelOutput
    from peft import get_peft_model, LoraConfig, TaskType
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    BaseModelOutput = None

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from .trajectory_tokenizer import TrajectoryTokenizer
from .vision_motion_encoder import VisionMotionEncoder
from .resampler import CaptionAwareResampler
from .pointer_head import TrackPointerDecoder


class TCAMv2(nn.Module):
    """
    TCAM v2: open-vocabulary, query-free generative captioning of trajectories.

    Parameters
    ----------
    d_model         : shared latent dim (default 512)
    K               : max track slots
    num_frames      : T
    img_size        : spatial resolution used for ViT input
    num_freqs       : sinusoidal Fourier frequencies in TrajectoryTokenizer
    num_cross_attn  : Stage A cross-attn layers in VisionMotionEncoder
    num_joint_attn  : Stage B self-attn layers in VisionMotionEncoder
    num_heads       : attention heads throughout
    d_ff            : FFN hidden dim
    num_queries     : Q-Former query count
    num_resampler_layers : resampler depth
    lora_r          : LoRA rank
    lora_alpha      : LoRA scaling
    lora_dropout    : LoRA dropout
    t5_model_name   : HuggingFace model id for FLAN-T5
    vit_model_name  : timm model id for ViT-B/16
    freeze_vit      : whether to freeze ViT entirely (unfreeze last N in stage 2)
    use_gradient_checkpointing : enable checkpointing throughout
    """

    # Dedicated time tokens <T0>..<T{N-1}> (one per sampled frame) — Vid2Seq-style.
    # Temporal boundaries are predicted as SINGLE tokens (clean N-way classification)
    # instead of brittle free-text "[f3-f8]" (6-7 subword tokens) which the model
    # couldn't learn and collapsed to [f0-f15]. The number of time tokens follows
    # num_frames (set per-instance in __init__); this class default is the 16-frame case.
    BASE_SPECIAL = ["<CAP_BEGIN>", "<CAP_END>", "<TRK>"]
    SPECIAL_TOKENS = BASE_SPECIAL + [f"<T{i}>" for i in range(16)]

    def __init__(
        self,
        d_model: int = 512,
        K: int = 256,
        num_frames: int = 8,
        img_size: int = 224,
        num_freqs: int = 8,
        num_cross_attn: int = 3,
        num_joint_attn: int = 3,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_queries: int = 64,
        num_resampler_layers: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        t5_model_name: str = "google/flan-t5-base",
        vit_model_name: str = "vit_base_patch16_clip_224.openai",
        freeze_vit: bool = True,
        use_gradient_checkpointing: bool = False,
        patch_pool_stride: int = 2,
        use_pointer_head: bool = True,
        unfreeze_lm_head: bool = False,
        z_gain_min: float = 0.0,
        use_temporal_pos: bool = False,
        use_frame_tokens: bool = False,
        ordinal_time_init: bool = False,
        t5_use_encoder: bool = True,
        use_p_sem: bool = True,
        use_residual: bool = False,
    ) -> None:
        super().__init__()

        # Time-token count follows num_frames (one <Ti> per sampled frame). Build the
        # per-instance special-token list before _build_t5 reads it.
        self.SPECIAL_TOKENS = self.BASE_SPECIAL + [f"<T{i}>" for i in range(num_frames)]
        self.use_pointer_head = use_pointer_head
        # Captioning-phase knobs (defaults = original frozen-head behaviour):
        #   unfreeze_lm_head — let the full T5 output layer specialise to the
        #                      caption domain (not just the 3 special-token rows)
        #   z_gain_min       — floor on the visual gain so the model can't throttle
        #                      the visual channel to ~0 and fall back to the LM prior
        self.unfreeze_lm_head = unfreeze_lm_head
        self.z_gain_min = float(z_gain_min)
        # Route the visual tokens Z THROUGH T5's pretrained encoder (inputs_embeds)
        # instead of bypassing it. Probes proved the encoder is essential for
        # video-specific captioning; bypass collapses to generic/degenerate output.
        # Constructor arg (default True) — a hardcoded False here once made probe
        # scripts silently run in bypass mode against encoder-trained checkpoints.
        self.t5_use_encoder = t5_use_encoder
        # P_sem (frozen-ViT frame-mean semantics concatenated into the resampler
        # input) was a run_040 band-aid for captioning. run_025 had NO P_sem and
        # captioned well; a probe confirmed H_v-only Z reads video. Default True
        # for back-compat with P_sem checkpoints; run_046+ sets False to match the
        # proven-good run_025 caption path exactly.
        self.use_p_sem = use_p_sem
        self.use_residual = use_residual
        self.K = K
        self.T = num_frames
        self.img_size = img_size
        self.use_gc = use_gradient_checkpointing

        # ── ViT-B/16 vision encoder ───────────────────────────────────────────
        self.vit = self._build_vit(vit_model_name, freeze_vit)
        d_vit = 768  # ViT-B/16 output dim

        # Number of patches per frame: (img_size / 16)²
        n_patches_per_side = img_size // 16
        self.N_p = n_patches_per_side ** 2

        # ── Trajectory Tokenizer ──────────────────────────────────────────────
        self.tokenizer_module = TrajectoryTokenizer(
            d_model=d_model,
            d_vis=d_vit,
            num_frames=num_frames,
            num_freqs=num_freqs,
            use_residual=use_residual,
        )

        # ── Vision-Motion Joint Encoder ───────────────────────────────────────
        self.encoder = VisionMotionEncoder(
            d_model=d_model,
            d_vit=d_vit,
            num_cross_attn_layers=num_cross_attn,
            num_joint_attn_layers=num_joint_attn,
            num_heads=num_heads,
            d_ff=d_ff,
            patch_pool_stride=patch_pool_stride,
            use_gradient_checkpointing=use_gradient_checkpointing,
            num_frames=num_frames,
            use_temporal_pos=use_temporal_pos,
        )

        # ── Caption-Aware Resampler ───────────────────────────────────────────
        # FLAN-T5-base d_model = 768 (T5-small = 512, T5-large = 1024)
        d_t5 = 768
        self.resampler = CaptionAwareResampler(
            d_model=d_model,
            d_t5=d_t5,
            num_queries=num_queries,
            num_layers=num_resampler_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Project + normalise resampler output before T5 encoder.
        # LayerNorm stabilises the magnitude; the learned scalar controls the
        # effective "gain" the visual signal applies to T5's cross-attention.
        # Initialised to a small positive value so training starts near the
        # language-prior regime and gradually opens up the visual channel.
        self.z_proj_norm = nn.LayerNorm(d_t5)
        # Start at 1.0 so T5 encoder sees a properly-scaled visual signal from
        # the first step.  The gain is learned and can grow/shrink freely.
        self.z_gain = nn.Parameter(torch.ones(1))

        # ── Per-frame temporal tokens (dedicated temporal-grounding path) ─────
        # The resampler pools H_v into 64 *unordered* tokens, so frame identity is
        # lost before T5 — making temporal localization ([f_s-f_e]) impossible. We
        # additionally feed T explicit per-frame tokens (spatially-pooled patches +
        # a per-frame position embedding) so the decoder can attend to "frame t" and
        # place events in time. Mirrors how the <TRK> pointer gave spatial grounding
        # a dedicated path. Captioning + grounding paths are untouched.
        self.use_frame_tokens = use_frame_tokens
        if use_frame_tokens:
            self.frame_proj = nn.Linear(d_vit, d_t5)
            self.frame_norm = nn.LayerNorm(d_t5)
            self.frame_temporal_pos = nn.Parameter(torch.zeros(num_frames, d_t5))

        # ── ViT-semantic context for the resampler ────────────────────────────
        # Problem: H_v (the resampler's key-value source) is shaped by grounding
        # gradients and encodes tracking features rather than semantic content.
        # The 64 resampler output tokens (Z) therefore carry non-discriminative
        # "mall scene" content regardless of what's actually in the video.
        #
        # Fix: inject frozen ViT frame-mean features directly into the resampler's
        # KV pool. P_sem = LayerNorm(Linear(P.mean(spatial))) gives T discriminative
        # per-frame tokens (dog vs giraffe vs counter) that are immune to grounding
        # gradient contamination (ViT is frozen). The resampler queries learn to
        # attend to P_sem, so Z inherits video-specific semantics.
        #
        # Standard Kaiming init — P_sem is competitive with H_v from step 1.
        # (A near-zero init starves the gradient path through p_cap_proj because
        # P_sem ≈ 0 means the resampler ignores it → ∂Z/∂P_sem ≈ 0 → p_cap_proj
        # never grows. Only use near-zero when fine-tuning from a strong prior.)
        self.p_cap_proj = nn.Linear(d_vit, d_model, bias=False)
        self.p_cap_norm = nn.LayerNorm(d_model)

        # ── FLAN-T5-base + LoRA ───────────────────────────────────────────────
        self.t5, self.text_tokenizer = self._build_t5(
            t5_model_name, lora_r, lora_alpha, lora_dropout,
            unfreeze_lm_head=self.unfreeze_lm_head,
            ordinal_time_init=ordinal_time_init,
        )
        self._trk_token_id: int | None = None  # set after tokenizer setup

        # ── Decoupled track pointer (LISA "embedding-as-mask") ────────────────
        # The T5 decoder cross-attends ONLY to the 64 Z tokens.  The <TRK>
        # hidden state is a query that this dedicated module resolves against the
        # K enriched track features (H_v[:, :K]) with its own softmax — keeping
        # captioning clean and giving grounding its full gradient mass.
        self.pointer_decoder = (
            TrackPointerDecoder(
                d_t5=d_t5,
                d_key=d_model,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=0.0,
            )
            if use_pointer_head
            else None
        )

    # ── builders ─────────────────────────────────────────────────────────────

    def _build_vit(self, model_name: str, freeze: bool) -> nn.Module:
        if not _TIMM_AVAILABLE:
            # Stub for testing without timm
            return _StubViT(d_out=768, patch_size=16)

        vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        if freeze:
            for p in vit.parameters():
                p.requires_grad_(False)
        return vit

    def _build_t5(
        self,
        model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        unfreeze_lm_head: bool = False,
        ordinal_time_init: bool = False,
    ):
        if not _HF_AVAILABLE:
            return _StubT5(), None

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": self.SPECIAL_TOKENS})

        t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        n_model_vocab = t5.config.vocab_size   # e.g. 32128 for FLAN-T5-base
        n_tok_vocab = len(tokenizer)            # e.g. 32103 (= 32100 + 3 new)

        # FLAN-T5 tokenizer has fewer IDs in use (32103) than the model vocab
        # (32128).  Our special token IDs (32100, 32101, 32102) already fall
        # within the existing model rows — calling resize_token_embeddings with
        # a *smaller* value would truncate the pretrained lm_head, destroying
        # the pretrained output distribution and causing NLL > 400.
        # Only resize if we genuinely need *more* rows than the model has.
        if n_tok_vocab > n_model_vocab:
            t5.resize_token_embeddings(n_tok_vocab)

        # Initialise the embedding rows for our special tokens to the mean of
        # existing embeddings (zero-mean approximation; avoids random large norms
        # that push the model away from its pretrained language prior).
        with torch.no_grad():
            emb_weight = t5.get_input_embeddings().weight
            existing_mean = emb_weight[:min(n_model_vocab, 32000)].mean(dim=0)
            for sp_tok in self.SPECIAL_TOKENS:
                sp_id = tokenizer.convert_tokens_to_ids(sp_tok)
                if sp_id is not None and sp_id < emb_weight.shape[0]:
                    emb_weight[sp_id] = existing_mean
            # If lm_head is not tied to input embeddings (happens after PEFT),
            # also initialise those rows so the output logits for our special
            # tokens start near-zero rather than random.
            lm_head = t5.get_output_embeddings()
            if lm_head is not None and lm_head.weight.data_ptr() != emb_weight.data_ptr():
                lm_head_w = lm_head.weight
                lm_mean = lm_head_w[:min(n_model_vocab, 32000)].mean(dim=0)
                for sp_tok in self.SPECIAL_TOKENS:
                    sp_id = tokenizer.convert_tokens_to_ids(sp_tok)
                    if sp_id is not None and sp_id < lm_head_w.shape[0]:
                        lm_head_w[sp_id] = lm_mean

            # Ordinal time-token init: give <T0>..<T{N-1}> an ORDERED starting point
            # (sinusoidal positional signal on top of the mean) so <T4> begins near
            # <T5> instead of cold/identical. Fixes the "deviates but imprecise"
            # symptom from cold mean-init. Applied to input emb + lm_head rows.
            if ordinal_time_init:
                time_toks = [t for t in self.SPECIAL_TOKENS if t.startswith("<T")]
                d = emb_weight.shape[1]
                k = torch.arange(0, d, 2, dtype=torch.float32)
                div = torch.exp(-math.log(10000.0) * k / d)        # (d/2,)
                for i, tok in enumerate(time_toks):
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if tid is None or tid >= emb_weight.shape[0]:
                        continue
                    pe = torch.zeros(d)
                    pe[0::2] = torch.sin(i * div)
                    pe[1::2] = torch.cos(i * div)[: pe[1::2].shape[0]]
                    pe = 0.1 * pe.to(emb_weight.dtype)              # gentle ordinal signal
                    emb_weight[tid] = existing_mean + pe
                    lm_head = t5.get_output_embeddings()
                    if lm_head is not None and lm_head.weight.data_ptr() != emb_weight.data_ptr() \
                            and tid < lm_head.weight.shape[0]:
                        lm_head.weight[tid] = lm_head.weight[tid] + pe

        # Apply LoRA to Q, K, V projections of decoder attention.
        # K is included because in the encoder-bypass architecture the decoder
        # cross-attention K projects Z (visual tokens) to key vectors; a
        # randomly-initialised (frozen) K would produce meaningless keys for Z.
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "k", "v"],
            lora_dropout=lora_dropout,
            task_type=TaskType.SEQ_2_SEQ_LM,
            bias="none",
        )
        t5 = get_peft_model(t5, lora_cfg)

        # Unfreeze the lm_head weight and register a gradient hook that zeros
        # all rows EXCEPT the 3 special token rows.  This confines updates to
        # <CAP_BEGIN>/<TRK>/<CAP_END> (2304 effective params) while leaving
        # the pretrained output distribution for all other tokens intact.
        #
        # NOTE: indexing into a leaf tensor with requires_grad=False yields a
        # new (non-leaf) tensor, and setting requires_grad on it does NOT make
        # the underlying weight row trainable nor add it to model.parameters().
        # The only correct approach is to unfreeze the whole weight and mask
        # gradients for non-special rows via a hook.
        lm_head = t5.get_output_embeddings()
        if lm_head is not None:
            special_ids_list: list[int] = []
            for sp_tok in self.SPECIAL_TOKENS:
                sp_id = tokenizer.convert_tokens_to_ids(sp_tok)
                if sp_id is not None and sp_id < lm_head.weight.shape[0]:
                    special_ids_list.append(sp_id)
            if special_ids_list:
                lm_head.weight.requires_grad_(True)
                if unfreeze_lm_head:
                    # Captioning phase: train the FULL output layer so it can
                    # specialise its word distribution to the caption domain
                    # (the frozen head is the main cause of generic captions).
                    # No gradient mask → every vocab row updates.
                    pass
                else:
                    _sp_ids = torch.tensor(special_ids_list, dtype=torch.long)

                    def _sparse_grad_hook(grad: torch.Tensor,
                                          sp_ids: torch.Tensor = _sp_ids) -> torch.Tensor:
                        mask = torch.zeros_like(grad)
                        mask[sp_ids] = 1.0
                        return grad * mask

                    lm_head.weight.register_hook(_sparse_grad_hook)

        if self.use_gc:
            # Non-reentrant checkpointing avoids the DDP+LoRA "marked ready twice"
            # error that occurs with reentrant (default) checkpointing on multi-GPU.
            gc_kwargs = {"use_reentrant": False}
            t5.base_model.model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            t5.base_model.model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

        return t5, tokenizer

    # ── helpers ───────────────────────────────────────────────────────────────

    def unfreeze_vit_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n transformer blocks of ViT (for stage 2 SFT)."""
        if not _TIMM_AVAILABLE or isinstance(self.vit, _StubViT):
            return
        blocks = list(self.vit.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)

    @property
    def trk_token_id(self) -> int | None:
        if self._trk_token_id is not None:
            return self._trk_token_id
        if self.text_tokenizer is not None:
            ids = self.text_tokenizer.convert_tokens_to_ids(["<TRK>"])
            if ids and ids[0] != self.text_tokenizer.unk_token_id:
                self._trk_token_id = ids[0]
        return self._trk_token_id

    @property
    def cap_begin_token_id(self) -> int | None:
        if getattr(self, "_cap_begin_token_id", None) is not None:
            return self._cap_begin_token_id
        self._cap_begin_token_id = None
        if self.text_tokenizer is not None:
            ids = self.text_tokenizer.convert_tokens_to_ids(["<CAP_BEGIN>"])
            if ids and ids[0] != self.text_tokenizer.unk_token_id:
                self._cap_begin_token_id = ids[0]
        return self._cap_begin_token_id

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        frames: torch.Tensor,
        query_pts: torch.Tensor,
        caption_ids: torch.Tensor,
        decoder_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        track_valid: torch.Tensor | None = None,
        X: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        X_res: Optional[torch.Tensor] = None,
        patch_tokens: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        no_tf: bool = False,
        **_,
    ) -> dict:
        """
        Parameters
        ----------
        frames        : (B, T, H, W, 3) uint8
        query_pts     : (B, K, 2) float32  — normalised (x,y) query seeds
        caption_ids   : (B, L) int64  — encoder input token IDs
        decoder_ids   : (B, L) int64  — teacher-forced decoder input
        attention_mask: (B, L) int64  — encoder attention mask
        track_valid   : (B, K) bool
        X             : (B, T, K, 2) float32  — trajectories (or None for dummy)
        V             : (B, T, K) float32     — visibility flags (or None for dummy)

        Returns
        -------
        dict with keys:
          logits          : (B, L, vocab_size)
          pointer_alphas  : (N_trk, K) or None
          pointer_batch   : (N_trk,)  batch index for each pointer prediction
          T_tr            : (B, K, d_model)  — track tokens (detached, for loss)
        """
        B, T_frames, H, W, C = frames.shape
        device = frames.device
        dtype = next(self.parameters()).dtype

        # ── ViT: extract patch tokens (or use pre-computed cache) ────────────
        # patch_mask (B,) bool: True = cached PLM features, False = compute from frames.
        # Mixed batches (PLM + MeViS/RVOS) must be handled per-sample: PLM rows
        # provide zero frames (dummy), so we must use cached tokens for those rows
        # and run ViT only on MeViS/RVOS rows that have real frames.
        if patch_tokens is not None and patch_mask is not None and patch_mask.all():
            # All samples have cached tokens (pure PLM batch)
            P = patch_tokens.to(device=device, dtype=dtype)
        elif patch_tokens is not None and patch_mask is not None and patch_mask.any():
            # Mixed batch: compute ViT for non-cached rows, use cache for PLM rows
            need_vit = ~patch_mask  # (B,) rows that need ViT
            frames_f = frames[need_vit].reshape(-1, H, W, C)
            frames_f = frames_f.float() / 255.0
            frames_f = frames_f.permute(0, 3, 1, 2).to(dtype)
            frames_f = _imagenet_normalise(frames_f)
            with torch.set_grad_enabled(
                any(p.requires_grad for p in self.vit.parameters())
            ):
                pt = self._extract_patch_tokens(frames_f)         # (n*T, N_p, 768)
            n_need = need_vit.sum().item()
            P = patch_tokens.to(device=device, dtype=dtype).clone()
            P[need_vit] = pt.reshape(n_need, T_frames, pt.shape[1], 768).to(dtype)
        else:
            # No cached tokens: decode all frames through ViT (MeViS/RVOS-only batch)
            frames_f = frames.reshape(B * T_frames, H, W, C)
            frames_f = frames_f.float() / 255.0
            frames_f = frames_f.permute(0, 3, 1, 2).to(dtype)
            frames_f = _imagenet_normalise(frames_f)
            with torch.set_grad_enabled(
                any(p.requires_grad for p in self.vit.parameters())
            ):
                pt = self._extract_patch_tokens(frames_f)         # (BT, N_p, 768)
            P = pt.reshape(B, T_frames, pt.shape[1], 768).to(dtype)

        N_p = P.shape[2]

        # ── trajectories X, V ─────────────────────────────────────────────────
        if X is None or V is None:
            # During stage-0 smoke test (no CoTracker3 run yet): use query_pts
            # repeated across T frames as dummy trajectories.
            X = query_pts.unsqueeze(1).expand(-1, T_frames, -1, -1)  # (B, T, K, 2)
            V = torch.ones(B, T_frames, self.K, device=device, dtype=dtype)
            # Mask padding tracks
            if track_valid is not None:
                V = V * track_valid.unsqueeze(1).to(dtype)

        X = X.to(dtype)
        V = V.to(dtype)

        # ── TrajectoryTokenizer ───────────────────────────────────────────────
        T_tr = self.tokenizer_module(X, V, P, X_res)                     # (B, K, 512)

        # ── VisionMotionEncoder ───────────────────────────────────────────────
        H_v = self.encoder(T_tr, P, track_valid)                  # (B, K+N̂, 512)

        # ── CaptionAwareResampler → Z ─────────────────────────────────────────
        Z, Z_resampler = self.encode_Z(P, H_v)

        # Truncate decoder sequence to the maximum real (non-padding) length in
        # the batch.  T5 was pretrained on sequences of 16–512 tokens; passing
        # 256 mostly-zero tokens degrades logit quality significantly.
        max_real = int(attention_mask.sum(dim=-1).max().item())
        max_real = max(max_real, 1)
        if no_tf:
            # No teacher forcing: give T5 only the start token so it must predict
            # the first caption token entirely from Z (direct gradient, no language-
            # prior shortcut).  Shapes stay consistent: logits=(B,1,V), labels=(B,1).
            decoder_ids_trunc = decoder_ids[:, 0:1]
            caption_ids_trunc = caption_ids[:, 0:1]
            attention_mask_trunc = torch.ones(B, 1, dtype=torch.long, device=device)
        else:
            decoder_ids_trunc = decoder_ids[:, :max_real]
            caption_ids_trunc = caption_ids[:, :max_real]
            attention_mask_trunc = attention_mask[:, :max_real]

        # Use H_v[:, :K] — the encoder-enriched track tokens — not the raw T_tr
        # from the trajectory tokenizer.  H_v[:, :K] has absorbed visual context
        # from the ViT patch tokens via Stage-A cross-attention, allowing the
        # pointer head to distinguish on-object vs off-object tracks by appearance.
        # Raw T_tr encodes only kinematics on a uniform 36×36 grid, so all K keys
        # look nearly identical and the pointer distribution stays near-uniform.
        K_actual = T_tr.shape[1]
        H_v_tracks = H_v[:, :K_actual]  # (B, K, d_model) — visually-enriched

        # Clean decoder: cross-attends only to the 64 Z tokens (captioning path
        # untouched — no track tokens in the LM cross-attention memory).
        logits, decoder_hidden = self._t5_forward(Z, decoder_ids_trunc)

        # ── Decoupled track pointer (Kosmos-2-on-<TRK>) ───────────────────────
        # The <TRK> hidden state queries the pointer module against the enriched
        # track keys (kept live so the encoder learns to be object-aware).  We
        # return RAW logits per <TRK>; the loss applies per-event temporal gating
        # and hard cross-entropy to the object's centroid grid cell.
        pointer_logits, pointer_batch = None, None
        if self.pointer_decoder is not None and self.trk_token_id is not None:
            pointer_logits, pointer_batch = (
                self.pointer_decoder.pointer_referring(
                    decoder_hidden=decoder_hidden,
                    token_ids=decoder_ids_trunc,
                    trk_token_id=self.trk_token_id,
                    cap_begin_id=self.cap_begin_token_id,
                    keys=H_v_tracks,
                    track_valid=track_valid,
                    return_logits=True,
                )
            )

        return {
            "logits": logits,
            "pointer_logits": pointer_logits,
            "pointer_batch": pointer_batch,
            "T_tr": T_tr,
            # Return truncated ids so callers can compute loss without 256-token padding
            "caption_ids_trunc": caption_ids_trunc,
            "attention_mask_trunc": attention_mask_trunc,
            # Resampler output before frame_tokens; used by z_diversity_loss to prevent
            # cross-video Z collapse without frame_tokens (which are always diverse) masking it.
            "Z_resampler": Z_resampler,
        }

    # ── T5 forward helpers ────────────────────────────────────────────────────

    def encode_Z(self, P: torch.Tensor, H_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single source of truth for building T5's visual input Z.

        MUST be used by every path that feeds T5 (training forward, generation,
        probe scripts). A hand-rolled eval copy that omitted P_sem fed T5 a Z
        distribution it was never trained on and produced language-prior-only
        captions ("man holding a phone" for a fish video).

        Steps:
          1. P_sem: frozen-ViT frame-mean semantics, immune to grounding gradients
          2. resampler over [H_v ; P_sem]  → (B, N_q, d_t5)
          3. LayerNorm + learnable gain (floored at z_gain_min)
          4. optional per-frame temporal tokens appended

        Returns (Z, Z_resampler) where Z_resampler is the pre-frame-token
        snapshot used by z_diversity_loss.
        """
        if self.use_p_sem:
            P_sem = self.p_cap_norm(self.p_cap_proj(P.mean(dim=2)))  # (B, T, d_model)
            Z = self.resampler(torch.cat([H_v, P_sem], dim=1))       # (B, N_q, d_t5)
        else:
            Z = self.resampler(H_v)                                  # run_025-native path
        Z = self.z_proj_norm(Z) * self.z_gain.abs().clamp_min(self.z_gain_min)
        Z_resampler = Z  # resampler-only; frame tokens are always diverse and would mask collapse
        if self.use_frame_tokens:
            Z = torch.cat([Z, self._frame_tokens(P)], dim=1)      # (B, N_q + T, d_t5)
        return Z, Z_resampler

    def _frame_tokens(self, P: torch.Tensor) -> torch.Tensor:
        """Per-frame temporal tokens for T5: spatially-pool patches per frame, project,
        and add a per-frame position embedding. (B, T, N_p, d_vit) -> (B, T, d_t5)."""
        ft = self.frame_norm(self.frame_proj(P.mean(dim=2)))      # (B, T, d_t5)
        return ft + self.frame_temporal_pos[:P.shape[1]].unsqueeze(0)

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run ViT and return patch tokens (excluding [CLS]). (BT, N_p, d_vit)"""
        if isinstance(self.vit, _StubViT):
            # Stub returns (BT, N_p, d) without CLS
            return self.vit.forward_features(x)[:, 1:, :]

        # timm ViT: forward_features returns (BT, N_p+1, 768) with CLS prepended
        feats = self.vit.forward_features(x)     # (BT, N_p+1, d)
        # Drop CLS token
        return feats[:, 1:, :]                   # (BT, N_p, d)

    def _t5_forward(
        self,
        Z: torch.Tensor,
        decoder_ids: torch.Tensor,
    ):
        """
        Run T5 decoder with Z passed directly as encoder_outputs.

        Architecture (encoder-bypass BLIP-2 variant):
          Instead of routing Z through T5's encoder self-attention layers
          (which were pretrained on text and cannot interpret visual features),
          we skip the encoder entirely and pass Z directly as the encoder
          hidden states.  The decoder's cross-attention then attends directly
          to the visual tokens via its LoRA-adapted Q and V projections.

          This is equivalent to:
            encoder_hidden_states = Z  (N_q × d_t5)
            decoder cross-attends to encoder_hidden_states

        Returns (logits (B, L, V), decoder_hidden (B, L, d_dec) or None)
        """
        if isinstance(self.t5, _StubT5):
            B, L = decoder_ids.shape
            V = 32128
            d = 768
            return (
                torch.randn(B, L, V, device=Z.device, dtype=Z.dtype),
                torch.randn(B, L, d, device=Z.device, dtype=Z.dtype),
            )

        B, N_q, _ = Z.shape
        device = Z.device

        # All N_q encoder tokens are valid (no padding)
        encoder_attn_mask = torch.ones(B, N_q, dtype=torch.long, device=device)

        # Decoder: embed decoder_ids (teacher-forced input)
        embed_fn = self.t5.base_model.model.shared
        dec_embeds = embed_fn(decoder_ids)                          # (B, L, d_t5)

        # Run T5 in FP32 to prevent FP16 overflow from large pretrained weights.
        # Pass Z directly as encoder_outputs to bypass the T5 encoder; the
        # decoder's cross-attention (with LoRA on Q and V) is trained to
        # extract visual information from Z.
        with torch.autocast(device_type="cuda", enabled=False):
            if self.t5_use_encoder:
                # Run Z THROUGH T5's pretrained encoder (standard seq2seq), then the
                # decoder cross-attends to the contextualised encoder output.
                outputs = self.t5(
                    inputs_embeds=Z.float(),
                    attention_mask=encoder_attn_mask,
                    decoder_inputs_embeds=dec_embeds.float(),
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                # Encoder-bypass: pass Z directly as encoder_outputs.
                outputs = self.t5(
                    encoder_outputs=BaseModelOutput(last_hidden_state=Z.float()),
                    attention_mask=encoder_attn_mask,
                    decoder_inputs_embeds=dec_embeds.float(),
                    output_hidden_states=True,
                    return_dict=True,
                )

        logits = outputs.logits                                     # (B, L, vocab)
        decoder_hidden = outputs.decoder_hidden_states[-1]          # (B, L, d_dec)
        return logits, decoder_hidden


# ── Stubs for testing without timm / transformers ────────────────────────────

class _StubViT(nn.Module):
    """Minimal ViT stub that returns random patch tokens."""
    def __init__(self, d_out: int = 768, patch_size: int = 16) -> None:
        super().__init__()
        self.d_out = d_out
        self.patch_size = patch_size
        self._dummy = nn.Linear(1, 1)  # so parameters() is non-empty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n = (H // self.patch_size) * (W // self.patch_size)
        # Return (B, N_p, d_out) — no CLS token prepended in stub
        return torch.randn(B, n, self.d_out, device=x.device, dtype=x.dtype)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n = (H // self.patch_size) * (W // self.patch_size)
        # CLS + patches
        return torch.randn(B, n + 1, self.d_out, device=x.device, dtype=x.dtype)

    def parameters(self, **kw):
        return super().parameters(**kw)


class _StubT5(nn.Module):
    """Minimal T5 stub that returns random logits."""
    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Linear(1, 1)


# ── image normalisation ───────────────────────────────────────────────────────

_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _imagenet_normalise(x: torch.Tensor) -> torch.Tensor:
    """Normalise (BT, 3, H, W) float tensor in [0,1] to ImageNet stats."""
    mean = _MEAN.to(x.device, dtype=x.dtype)[None, :, None, None]
    std = _STD.to(x.device, dtype=x.dtype)[None, :, None, None]
    return (x - mean) / std
