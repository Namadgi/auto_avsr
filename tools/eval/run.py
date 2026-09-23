"""Run one VSR configuration over samples.csv and write results/<name>.json.

    python -m tools.eval.run --config tools/eval/configs/baseline.json --clips DIR --samples samples.csv [--out results] [--limit N]

Each clip is transcoded once (same ffmpeg step as the service) and inferred once; every
(clip, phrase) sample is then scored with the production postprocess. Per-clip timings are
CUDA-synchronized per stage, so 'encoder' and 'decode' reflect real GPU time.
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

import torch

from tools.eval.config import EvalConfig
from tools.eval.postprocess import postprocess
from tools.eval.preprocess import transcode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--limit', type=int, default=0, help='only first N distinct clips')
    ap.add_argument('--warmup', type=int, default=1, help='untimed warm-up inferences on the first clip')
    a = ap.parse_args()

    cfg = EvalConfig.load(a.config)
    samples = list(csv.DictReader(open(a.samples)))
    clips = list(dict.fromkeys(s['clip'] for s in samples))
    if a.limit:
        clips = clips[:a.limit]
    by_clip = defaultdict(list)
    for s in samples:
        if s['clip'] in clips:
            by_clip[s['clip']].append(s)

    print(f"config={cfg.name} device={cfg.device} clips={len(clips)} samples={sum(map(len, by_clip.values()))}", file=sys.stderr)
    model = cfg.build_model()
    cuda = cfg.device.startswith('cuda') and torch.cuda.is_available()

    def infer(path):
        with torch.no_grad():
            t0 = time.perf_counter()
            text = model(path)
            if cuda:
                torch.cuda.synchronize()
            return text, time.perf_counter() - t0, dict(model.timer.timings)

    first = transcode(os.path.join(a.clips, clips[0]))
    for _ in range(a.warmup):
        infer(first)
    os.remove(first)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    results = []
    for i, clip in enumerate(clips, 1):
        src = os.path.join(a.clips, clip)
        t0 = time.perf_counter()
        mp4 = transcode(src)
        t_transcode = time.perf_counter() - t0
        try:
            text, t_infer, stages = infer(mp4)
        except Exception as e:  # keep going; record the failure
            text, t_infer, stages = None, None, {'error': repr(e)}
        finally:
            os.remove(mp4)
        rec = {'clip': clip, 'transcript': text, 'transcode_s': round(t_transcode, 4),
               'infer_s': None if t_infer is None else round(t_infer, 4),
               'stages_s': {k: round(v, 4) for k, v in stages.items() if isinstance(v, float)},
               'error': stages.get('error'), 'samples': []}
        for s in by_clip[clip]:
            pp = postprocess(text or '', s['phrase']) if text is not None else None
            rec['samples'].append({'phrase': s['phrase'], 'kind': s['kind'], 'expected': s['expected'], **(pp or {})})
        results.append(rec)
        print(f"[{i:3d}/{len(clips)}] {clip:38s} infer={rec['infer_s']}s  {json.dumps(rec['stages_s'])}  -> {text!r}", file=sys.stderr)

    ok = [r for r in results if r['infer_s'] is not None]
    summary = {
        'config': cfg.to_dict(), 'n_clips': len(results), 'n_errors': len(results) - len(ok),
        'infer_s': {'mean': st.mean(r['infer_s'] for r in ok), 'p50': st.median(r['infer_s'] for r in ok),
                    'max': max(r['infer_s'] for r in ok)} if ok else None,
        'stages_mean_s': {k: round(st.mean(r['stages_s'].get(k, 0) for r in ok), 4)
                          for k in sorted({k for r in ok for k in r['stages_s']})} if ok else None,
        'peak_gpu_mem_mb': round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if cuda else None,
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'{cfg.name}.json')
    with open(path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=1)
    print(json.dumps(summary, indent=1), file=sys.stderr)
    print(f"wrote {path}", file=sys.stderr)


if __name__ == '__main__':
    main()
