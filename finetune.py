#!/usr/bin/env python
"""Fine-tune TCAM on your own videos.

  python finetune.py --data train.json --out_dir my_tcam
  python infer.py --model my_tcam --video new_clip.mp4

train.json is a list of videos, each with its events (times in seconds; "box" is
optional and, if given, also trains grounding -- [x1, y1, x2, y2] normalised to [0, 1]):

  [{"video": "clips/drawer.mp4",
    "events": [{"start": 0.0, "end": 3.5, "caption": "a hand opens the drawer", "box": [0.2, 0.3, 0.8, 0.7]},
               {"start": 3.5, "end": 7.0, "caption": "the hand closes the drawer"}]}]

Uses the original training objective: label-smoothed caption cross-entropy (structural
tokens x1.5) + class-balanced BCE on the <TRK> pointer. The frozen ViT stays frozen;
everything else trains, as in the original run.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tcam import TCAMPipeline
from tcam.pipeline import DEFAULT_REPO, NUM_FRAMES, TRACK_HW, VIT_SIZE, _read, sample_indices, video_meta


# ── data ──────────────────────────────────────────────────────────────────────

def prepare(pipe, item, root):
    """Track + sample one video, and build its target text and grounding targets."""
    path = str(Path(root, item["video"]))
    total, fps = video_meta(path)
    idx = sample_indices(total)
    X, V = pipe.track(_read(path, idx, *TRACK_HW))
    times = np.array(idx) / fps

    def tok(t):  # seconds -> nearest of the 16 sampled frames
        return int(np.abs(times - t).argmin())

    parts, onobj = [], np.zeros((len(item["events"]), X.shape[1]), np.float32)
    for j, e in enumerate(item["events"]):
        a, b = tok(e["start"]), max(tok(e["start"]), tok(e["end"]))
        parts.append(f"<CAP_BEGIN> <T{a}><T{b}> {e['caption'].strip()} <TRK> <CAP_END>")  # run_028 format
        if "box" in e:
            x1, y1, x2, y2 = e["box"]
            win = X[a:b + 1]
            inside = ((win[..., 0] >= x1) & (win[..., 0] <= x2) & (win[..., 1] >= y1) & (win[..., 1] <= y2)).any(0)
            vis = V[a:b + 1].max(0) > 0.5
            on = inside & vis
            if not on.any() and vis.any():  # box smaller than the grid: nearest visible track
                d = ((win.mean(0) - [(x1 + x2) / 2, (y1 + y2) / 2]) ** 2).sum(-1)
                d[~vis] = np.inf
                on[int(d.argmin())] = True
            onobj[j] = on
    t = pipe.model.text_tokenizer(" ".join(parts), truncation=True, max_length=512)
    ids = np.array(t["input_ids"], np.int64)
    return dict(frames=_read(path, idx, VIT_SIZE, VIT_SIZE), X=X, V=V, ids=ids,
                dec=np.concatenate([[0], ids[:-1]]), onobj=onobj, video=item["video"])


def collate(samples, dev):
    B, L = len(samples), max(len(s["ids"]) for s in samples)
    E, K = max(len(s["onobj"]) for s in samples), samples[0]["X"].shape[1]
    ids = torch.zeros(B, L, dtype=torch.long); dec = torch.zeros_like(ids); am = torch.zeros_like(ids)
    onobj = torch.zeros(B, E, K)
    for b, s in enumerate(samples):
        n = len(s["ids"])
        ids[b, :n], dec[b, :n], am[b, :n] = torch.from_numpy(s["ids"]), torch.from_numpy(s["dec"]), 1
        onobj[b, :len(s["onobj"])] = torch.from_numpy(s["onobj"])
    st = lambda k: torch.from_numpy(np.stack([s[k] for s in samples])).to(dev)
    return dict(frames=st("frames"), X=st("X").float(), V=st("V").float(), caption_ids=ids.to(dev),
                decoder_ids=dec.to(dev), attention_mask=am.to(dev), onobj=onobj.to(dev),
                track_valid=torch.ones(B, K, dtype=torch.bool, device=dev),
                query_pts=torch.zeros(B, K, 2, device=dev))


# ── losses (verbatim objective from the original training) ────────────────────

def caption_loss(logits, targets, mask, structural_ids, weight=1.5, smoothing=0.1):
    B, L, Vs = logits.shape
    t = targets.clone(); t[mask == 0] = -100
    per = F.cross_entropy(logits.reshape(-1, Vs), t.reshape(-1), ignore_index=-100,
                          label_smoothing=smoothing, reduction="none")
    w = torch.ones_like(per)
    for sid in structural_ids:
        w[t.reshape(-1) == sid] = weight
    valid = (t.reshape(-1) != -100).float()
    return (per * w * valid).sum() / (w * valid).sum().clamp(min=1)


def pointer_loss(logits, batch_idx, onobj):
    if logits is None or logits.shape[0] == 0:
        return logits.new_zeros(()) if logits is not None else torch.zeros(())
    K, seen, losses = logits.shape[1], {}, []
    for i, b in enumerate(batch_idx.tolist()):
        j = seen.get(b, 0); seen[b] = j + 1          # i-th <TRK> of a video <-> i-th event
        if j >= onobj.shape[1] or onobj[b, j].sum() < 1:
            continue
        pos = onobj[b, j]
        n = pos.sum()
        losses.append(F.binary_cross_entropy_with_logits(
            torch.nan_to_num(logits[i], neginf=-30.0), pos, pos_weight=(K - n) / n))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="train.json (format above)")
    ap.add_argument("--video_root", default=None, help="prefix for relative video paths (default: next to --data)")
    ap.add_argument("--model", default=DEFAULT_REPO, help="starting checkpoint: HF repo id or local dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5, help="keep low; higher LRs can collapse the captioner")
    ap.add_argument("--grounding_weight", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    pipe = TCAMPipeline.from_pretrained(args.model, device=args.device)
    m, dev = pipe.model, pipe.device
    items = json.loads(Path(args.data).read_text())
    root = args.video_root or Path(args.data).parent
    print(f"preparing {len(items)} videos (CoTracker runs once per video)...")
    data = [prepare(pipe, it, root) for it in items]
    pipe.tracker = None; torch.cuda.empty_cache()

    tok = m.text_tokenizer
    structural = [tok.convert_tokens_to_ids(t) for t in ("<CAP_BEGIN>", "<CAP_END>", "<TRK>")]
    params = [p for n, p in m.named_parameters() if not n.startswith("vit.")]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.steps // 20)))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    m.train(); m.vit.eval()
    for step in range(1, args.steps + 1):
        b = collate(random.sample(data, min(args.batch_size, len(data))), dev)
        o = m(frames=b["frames"], query_pts=b["query_pts"], caption_ids=b["caption_ids"],
              decoder_ids=b["decoder_ids"], attention_mask=b["attention_mask"],
              track_valid=b["track_valid"], X=b["X"], V=b["V"])
        l_cap = caption_loss(o["logits"], o["caption_ids_trunc"], o["attention_mask_trunc"], structural)
        l_grd = pointer_loss(o["pointer_logits"], o["pointer_batch"], b["onobj"])
        loss = l_cap + args.grounding_weight * l_grd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  caption {l_cap.item():.4f}  grounding {l_grd.item():.4f}")
        if step % args.save_every == 0 or step == args.steps:
            save(m, args.model, out)
    print(f"done -> {out}   (python infer.py --model {out} --video <clip.mp4>)")


def save(m, src, out):
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().clone().contiguous() for k, v in m.state_dict().items() if not k.startswith("vit.")}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    p = Path(src)
    if p.is_dir():
        shutil.copy(p / "config.json", out / "config.json")
    else:
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(src, "config.json"), out / "config.json")
    print(f"  saved {out / 'model.safetensors'}")


if __name__ == "__main__":
    main()
