"""TCAM inference pipeline: raw video -> timestamped captions + grounded point tracks.

Preprocessing mirrors training exactly (scripts/precompute_tracks_plm.py and
scripts/precompute_vit_features.py in the research repo):
  * 16 frames sampled at int(i * total / 16)
  * CoTracker3 (offline) on those 16 frames at 640x480, 36x36 grid -> K=1296 tracks,
    coords normalised to [0, 1] by frame size
  * frozen CLIP ViT-B/16 on the same 16 frames at 224x224
Decoding mirrors the evaluation path used for every number in the paper
(scripts/eval_protocols.py:generate_events): beam search, num_beams=3.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .models.tcam_v2 import TCAMv2, _imagenet_normalise

DEFAULT_REPO = "bishoygaloaa/tcam"
NUM_FRAMES = 16
TRACK_HW = (480, 640)
VIT_SIZE = 224


# ── video I/O ─────────────────────────────────────────────────────────────────

def sample_indices(total: int, n: int = NUM_FRAMES) -> list[int]:
    if total <= n:
        return list(range(total)) + [total - 1] * (n - total)
    step = total / n
    return [min(int(i * step), total - 1) for i in range(n)]


def _read(path: str, idx: list[int], h: int, w: int) -> np.ndarray:
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(path), width=w, height=h)
    return vr.get_batch(idx).asnumpy()


def video_meta(path: str) -> tuple[int, float]:
    import decord
    vr = decord.VideoReader(str(path))
    return len(vr), float(vr.get_avg_fps() or 30.0)


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class TCAMEvent:
    caption: str
    start_token: int | None          # index into the 16 sampled frames (<T_i>)
    end_token: int | None
    start_sec: float | None
    end_sec: float | None
    confidence: float                # mean log-prob of the structural tokens
    track_weights: np.ndarray | None = field(repr=False, default=None)  # (K,) softmax

    def top_tracks(self, k: int = 40) -> np.ndarray:
        if self.track_weights is None:
            return np.zeros(0, dtype=np.int64)
        return np.argsort(-self.track_weights)[:k]


@dataclass
class TCAMResult:
    video: str
    fps: float
    num_video_frames: int
    frame_indices: list[int]
    tracks: np.ndarray = field(repr=False)      # (16, K, 2) normalised xy
    visibility: np.ndarray = field(repr=False)  # (16, K)
    events: list[TCAMEvent]
    raw_text: str
    offset_sec: float = 0.0

    def masks(self, event: int, top_k: int = 40, radius: int = 18,
              size: tuple[int, int] = (VIT_SIZE, VIT_SIZE), close_iter: int = 6) -> np.ndarray:
        """(16, H, W) uint8 masks for one event: discs of `radius` px around its
        top_k tracks, morphologically closed. Defaults = the paper's protocol."""
        from scipy.ndimage import binary_closing
        H, W = size
        w = self.events[event].track_weights
        sel = np.argsort(-w)[:top_k]
        out = np.zeros((self.tracks.shape[0], H, W), dtype=np.uint8)
        yy, xx = np.ogrid[:H, :W]
        for t in range(self.tracks.shape[0]):
            for k in sel:
                cx = int(np.clip(self.tracks[t, k, 0] * (W - 1), 0, W - 1))
                cy = int(np.clip(self.tracks[t, k, 1] * (H - 1), 0, H - 1))
                out[t][(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 1
            if close_iter > 0:
                out[t] = binary_closing(out[t], iterations=close_iter).astype(np.uint8)
        return out

    def to_dict(self, top_k: int = 40) -> dict:
        evs = []
        for e in self.events:
            sel = e.top_tracks(top_k)
            evs.append(dict(
                caption=e.caption, start_sec=e.start_sec, end_sec=e.end_sec,
                start_token=e.start_token, end_token=e.end_token,
                confidence=round(e.confidence, 4),
                top_tracks=sel.tolist(),
                top_track_weights=[round(float(e.track_weights[k]), 5) for k in sel]
                if e.track_weights is not None else [],
                # (16, top_k, 2) normalised xy of the grounded points per sampled frame
                top_track_xy=np.round(self.tracks[:, sel], 4).tolist(),
            ))
        return dict(video=self.video, fps=self.fps, num_video_frames=self.num_video_frames,
                    frame_indices=self.frame_indices, offset_sec=self.offset_sec,
                    raw_text=self.raw_text, events=evs)

    def save_json(self, path: str, top_k: int = 40) -> None:
        Path(path).write_text(json.dumps(self.to_dict(top_k), indent=2))


# ── pipeline ──────────────────────────────────────────────────────────────────

class TCAMPipeline:
    def __init__(self, model: TCAMv2, tracker, device: torch.device):
        self.model, self.tracker, self.device = model, tracker, device
        tok = model.text_tokenizer
        self._cb = tok.convert_tokens_to_ids("<CAP_BEGIN>")
        self._trk = model.trk_token_id
        self._time = {tok.convert_tokens_to_ids(f"<T{i}>"): i for i in range(NUM_FRAMES)}

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, repo_or_path: str = DEFAULT_REPO, device: str | None = None,
                        load_tracker: bool = True, cotracker_ckpt: str | None = None,
                        revision: str | None = None) -> "TCAMPipeline":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        p = Path(repo_or_path)
        if p.is_dir():
            cfg_path, w_path = p / "config.json", p / "model.safetensors"
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_or_path, "config.json", revision=revision)
            w_path = hf_hub_download(repo_or_path, "model.safetensors", revision=revision)
        cfg = json.loads(Path(cfg_path).read_text())
        model = build_model(cfg, w_path, dev)
        tracker = load_cotracker(dev, cotracker_ckpt) if load_tracker else None
        return cls(model, tracker, dev)

    # -------------------------------------------------------------- features
    @torch.no_grad()
    def track(self, frames_640x480: np.ndarray, grid_size: int = 36) -> tuple[np.ndarray, np.ndarray]:
        if self.tracker is None:
            raise RuntimeError("pipeline was loaded with load_tracker=False; pass tracks= explicitly")
        video = torch.from_numpy(frames_640x480).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        tr, vis = self.tracker(video, grid_size=grid_size)
        H, W = frames_640x480.shape[1:3]
        X = tr[0].float().cpu().numpy().copy()
        X[..., 0] /= max(W - 1, 1)
        X[..., 1] /= max(H - 1, 1)
        return (X.clip(0.0, 1.0).astype(np.float32),
                vis[0].float().cpu().numpy().clip(0.0, 1.0).astype(np.float32))

    @torch.no_grad()
    def patch_tokens(self, frames_224: np.ndarray) -> torch.Tensor:
        f = torch.from_numpy(frames_224).to(self.device).float() / 255.0
        f = _imagenet_normalise(f.permute(0, 3, 1, 2))
        pt = self.model._extract_patch_tokens(f)
        return pt.reshape(1, frames_224.shape[0], pt.shape[1], pt.shape[2])

    # ------------------------------------------------------------- inference
    def __call__(self, video: str, start_sec: float | None = None, end_sec: float | None = None,
                 chunk_sec: float | None = None, **gen) -> TCAMResult | list[TCAMResult]:
        """Run TCAM on a video (or a [start_sec, end_sec) segment of it).

        TCAM sees 16 frames per call, spread over the whole segment, and was trained
        on short clips (PLM-Video-Human / MeViS, typically < ~30 s). For long videos
        pass chunk_sec (e.g. 10) to get one result per consecutive chunk."""
        total, fps = video_meta(video)
        lo = 0 if start_sec is None else max(0, int(round(start_sec * fps)))
        hi = total if end_sec is None else min(total, int(round(end_sec * fps)))
        if hi - lo < 1:
            raise ValueError(f"empty segment [{start_sec}, {end_sec}) in {video}")
        if chunk_sec:
            step = max(1, int(round(chunk_sec * fps)))
            return [self._run(video, a, min(a + step, hi), total, fps, **gen)
                    for a in range(lo, hi, step) if min(a + step, hi) - a >= 2]
        return self._run(video, lo, hi, total, fps, **gen)

    @torch.no_grad()
    def _run(self, video, lo, hi, total, fps, num_beams: int = 3, max_new_tokens: int = 120,
             do_sample: bool = False, temperature: float = 0.9, top_p: float = 0.95,
             seed: int | None = None, tracks: tuple[np.ndarray, np.ndarray] | None = None) -> TCAMResult:
        idx = [lo + i for i in sample_indices(hi - lo)]
        if tracks is None:
            X, V = self.track(_read(video, idx, *TRACK_HW))
        else:
            X, V = tracks
        f224 = _read(video, idx, VIT_SIZE, VIT_SIZE)
        P = self.patch_tokens(f224)
        out = self.generate(P, X, V, num_beams=num_beams, max_new_tokens=max_new_tokens,
                            do_sample=do_sample, temperature=temperature, top_p=top_p, seed=seed)
        events = []
        for e in out["events"]:
            a, b = e.pop("start_token"), e.pop("end_token")
            s = idx[a] / fps if a is not None else None
            # an event ending on sampled frame b covers up to the next sampled frame
            en = (idx[b + 1] if b is not None and b + 1 < len(idx) else hi) / fps if b is not None else None
            events.append(TCAMEvent(start_token=a, end_token=b, start_sec=s, end_sec=en, **e))
        return TCAMResult(video=str(video), fps=fps, num_video_frames=total, frame_indices=idx,
                          tracks=X, visibility=V, events=events, raw_text=out["raw_text"],
                          offset_sec=lo / fps)

    @torch.no_grad()
    def generate(self, P: torch.Tensor, X: np.ndarray, V: np.ndarray, num_beams: int = 3,
                 max_new_tokens: int = 120, do_sample: bool = False, temperature: float = 0.9,
                 top_p: float = 0.95, seed: int | None = None) -> dict:
        """Core model call on precomputed features. P: (1,16,N,768) ViT patch tokens,
        X: (16,K,2) normalised tracks, V: (16,K) visibility."""
        from transformers.modeling_outputs import BaseModelOutput
        m, dev = self.model, self.device
        Xt = torch.from_numpy(np.asarray(X)).unsqueeze(0).to(dev).float()
        Vt = torch.from_numpy(np.asarray(V)).unsqueeze(0).to(dev).float()
        tv = torch.ones(1, Xt.shape[2], dtype=torch.bool, device=dev)

        T_tr = m.tokenizer_module(Xt, Vt, P, None)
        H_v = m.encoder(T_tr, P, tv)
        Z = m.resampler(H_v)
        Z = m.z_proj_norm(Z) * m.z_gain.abs().clamp_min(m.z_gain_min)
        if m.use_frame_tokens:
            Z = torch.cat([Z, m._frame_tokens(P)], dim=1)
        am = torch.ones(1, Z.shape[1], dtype=torch.long, device=dev)
        enc = m.t5.base_model.model.encoder(inputs_embeds=Z.float(), attention_mask=am, return_dict=True)
        kw = dict(encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
                  attention_mask=am, max_new_tokens=max_new_tokens, min_new_tokens=8,
                  repetition_penalty=1.3, forced_bos_token_id=self._cb,
                  bad_words_ids=[[self._cb, self._cb]])
        if do_sample:
            if seed is not None:
                torch.manual_seed(seed)
            gen = m.t5.generate(**kw, do_sample=True, num_beams=1, temperature=temperature, top_p=top_p)
        else:
            gen = m.t5.generate(**kw, num_beams=num_beams, early_stopping=True)

        logits, dec_hidden = m._t5_forward(Z, gen)
        pl, _ = m.pointer_decoder.pointer_referring(
            decoder_hidden=dec_hidden, token_ids=gen, trk_token_id=self._trk,
            cap_begin_id=self._cb, keys=H_v[:, :T_tr.shape[1]], track_valid=tv, return_logits=True)
        return dict(events=self._parse(gen[0].tolist(), logits[0], pl), gen_ids=gen[0].tolist(),
                    raw_text=m.text_tokenizer.decode(gen[0], skip_special_tokens=False))

    def _parse(self, ids: list[int], logits: torch.Tensor, pl: torch.Tensor | None) -> list[dict]:
        """Split generated ids into events at <TRK>; event i owns pointer row i."""
        tok = self.model.text_tokenizer
        logp = torch.log_softmax(logits.float(), dim=-1)
        struct = set(self._time) | {self._trk, self._cb}
        trk_pos = [i for i, t in enumerate(ids) if t == self._trk]
        n_ptr = 0 if pl is None else pl.shape[0]
        events, prev = [], 0
        for i, pos in enumerate(trk_pos[:n_ptr]):
            span = list(range(prev, pos + 1))
            prev = pos + 1
            s_lp = [logp[p - 1, ids[p]].item() for p in span if p >= 1 and ids[p] in struct]
            a_lp = [logp[p - 1, ids[p]].item() for p in span if p >= 1]
            times = [self._time[ids[p]] for p in span if ids[p] in self._time]
            events.append(dict(
                caption=_clean(tok.decode([ids[p] for p in span], skip_special_tokens=True)),
                start_token=times[0] if times else None,
                end_token=times[1] if len(times) > 1 else (times[0] if times else None),
                confidence=float(np.mean(s_lp)) if s_lp else (float(np.mean(a_lp)) if a_lp else float("-inf")),
                track_weights=torch.softmax(pl[i].float(), dim=-1).cpu().numpy()))
        return events


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<T\d+>|<CAP_BEGIN>|<CAP_END>|<TRK>", " ", t)).strip()


# ── builders ──────────────────────────────────────────────────────────────────

def build_model(cfg: dict, weights: str, dev: torch.device) -> TCAMv2:
    from safetensors.torch import load_file
    m = TCAMv2(d_model=cfg["d_model"], K=cfg["num_tracks"], num_frames=cfg["num_frames"],
               img_size=cfg["img_size"], use_pointer_head=True, unfreeze_lm_head=True,
               use_temporal_pos=cfg["use_temporal_pos"], use_frame_tokens=cfg["use_frame_tokens"],
               use_p_sem=cfg["use_p_sem"], t5_use_encoder=cfg["t5_use_encoder"],
               lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
               t5_model_name=cfg["t5_model_name"], vit_model_name=cfg["vit_model_name"])
    m.t5_use_encoder = cfg["t5_use_encoder"]
    miss, unexp = m.load_state_dict(load_file(str(weights)), strict=False)
    # Allowed to be missing: the frozen CLIP ViT (restored from timm), the deterministic
    # Fourier buffer, the unused P_sem projection (use_p_sem=False), and T5's encoder /
    # decoder embed_tokens, which are tied to t5...shared.weight (checked below).
    allowed = ("vit.", "tokenizer_module.fourier", "tokenizer_module.mlp_res", "p_cap_",
               "t5.base_model.model.encoder.embed_tokens.", "t5.base_model.model.decoder.embed_tokens.")
    bad = [k for k in miss if not k.startswith(allowed)]
    if bad or unexp:
        raise RuntimeError(f"checkpoint mismatch: missing={bad[:5]} unexpected={unexp[:5]}")
    t5 = m.t5.base_model.model
    assert (t5.encoder.embed_tokens.weight.data_ptr() == t5.shared.weight.data_ptr()
            == t5.decoder.embed_tokens.weight.data_ptr()), "T5 embeddings are not tied"
    return m.to(dev).float().eval()


def load_cotracker(dev: torch.device, ckpt: str | None = None):
    """CoTracker3 offline, weights fetched from Meta's official HF repo."""
    from cotracker.predictor import CoTrackerPredictor
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("facebook/cotracker3", "scaled_offline.pth")
    return CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).to(dev).eval()
