#!/usr/bin/env python
"""TCAM command-line inference.

  python infer.py --video clip.mp4                         # whole clip
  python infer.py --video long.mp4 --chunk_sec 10          # long video, 10 s chunks
  python infer.py --video a.mp4 b.mp4 --out_dir out --vis  # + overlay mp4 per video

Writes <out_dir>/<stem>.json (and <stem>_vis.mp4 with --vis); prints the events.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tcam import TCAMPipeline
from tcam.pipeline import DEFAULT_REPO, TRACK_HW, _read

COLORS = [(255, 140, 0), (0, 160, 255), (120, 255, 120), (255, 80, 200), (255, 230, 0), (0, 230, 230)]


def render(res, path, top_k=40, fps=4):
    """Overlay each event's grounded tracks on the 16 sampled frames."""
    import cv2, imageio
    H, W = TRACK_HW
    frames = _read(res.video, res.frame_indices, H, W)
    out = []
    for t, f in enumerate(frames):
        img = f.copy()
        live = [(i, e) for i, e in enumerate(res.events)
                if e.start_token is None or e.start_token <= t <= e.end_token]
        for i, e in live:
            c = COLORS[i % len(COLORS)]
            for k in e.top_tracks(top_k):
                x, y = res.tracks[t, k]
                cv2.circle(img, (int(x * (W - 1)), int(y * (H - 1))), 4, c, -1, cv2.LINE_AA)
        bar = np.full((22 * max(1, len(live)) + 30, W, 3), 22, np.uint8)
        cv2.putText(bar, f"t={res.frame_indices[t] / res.fps:.1f}s",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        for j, (i, e) in enumerate(live):
            cv2.putText(bar, e.caption[:90], (8, 44 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
        out.append(np.vstack([img, bar]))
    hmax = max(o.shape[0] for o in out)
    out = [np.vstack([o, np.zeros((hmax - o.shape[0], W, 3), np.uint8)]) for o in out]
    imageio.mimsave(path, out, fps=fps, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--model", default=DEFAULT_REPO, help="HF repo id or local directory")
    ap.add_argument("--out_dir", default="tcam_out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--start_sec", type=float, default=None)
    ap.add_argument("--end_sec", type=float, default=None)
    ap.add_argument("--chunk_sec", type=float, default=None,
                    help="split long videos into consecutive chunks of this length")
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--sample", action="store_true",
                    help="nucleus sampling instead of beam search (more varied time windows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=40, help="grounded tracks reported per event")
    ap.add_argument("--vis", action="store_true", help="write an overlay mp4 per video/chunk")
    args = ap.parse_args()

    pipe = TCAMPipeline.from_pretrained(args.model, device=args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for v in args.video:
        res = pipe(v, start_sec=args.start_sec, end_sec=args.end_sec, chunk_sec=args.chunk_sec,
                   num_beams=args.num_beams, do_sample=args.sample, seed=args.seed)
        chunks = res if isinstance(res, list) else [res]
        stem = Path(v).stem
        print(f"\n=== {v}")
        for ci, r in enumerate(chunks):
            for e in r.events:
                s = "?" if e.start_sec is None else f"{e.start_sec:6.2f}"
                en = "?" if e.end_sec is None else f"{e.end_sec:6.2f}"
                print(f"  [{s}s - {en}s] (conf {e.confidence:+.2f}) {e.caption}")
            if args.vis:
                suffix = f"_c{ci:03d}" if len(chunks) > 1 else ""
                render(r, str(out / f"{stem}{suffix}_vis.mp4"), args.top_k)
        payload = [r.to_dict(args.top_k) for r in chunks]
        (out / f"{stem}.json").write_text(json.dumps(payload if len(chunks) > 1 else payload[0], indent=2))
        print(f"  -> {out / (stem + '.json')}")


if __name__ == "__main__":
    main()
