"""torch.profiler over a few clips: CUDA kernel time by stage + Chrome trace.

    python -m tools.eval.profile --config tools/eval/configs/baseline.json --clips DIR --files clip_04_hJjTDB.mp4 [clip...] [--out results/profile]

Reads the trace back to attribute kernel time to the stage markers (record_function) that
StageTimer already delimits, and prints the top CUDA kernels. Use with `nvidia-smi dmon -s u`
in another shell to see SM% vs duty-cycle.
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tools.eval.config import EvalConfig
from tools.eval.preprocess import transcode

STAGES = ['read_video', 'landmarks', 'crop', 'transform', 'encoder', 'decode']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--out', default='results/profile')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = EvalConfig.load(a.config)
    model = cfg.build_model()
    cuda = cfg.device.startswith('cuda') and torch.cuda.is_available()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])

    # wrap each stage so record_function ranges appear in the trace
    orig_stage = model.timer.stage
    def stage(name):
        from contextlib import ExitStack
        es = ExitStack()
        es.enter_context(record_function(f'stage::{name}'))
        es.enter_context(orig_stage(name))
        return es
    model.timer.stage = stage

    paths = [transcode(os.path.join(a.clips, f)) for f in a.files]
    with torch.no_grad():
        model(paths[0])  # warm-up
        if cuda: torch.cuda.synchronize()
        with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
            for p in paths:
                with record_function('request'):
                    model(p)
                if cuda: torch.cuda.synchronize()
    for p in paths: os.remove(p)

    trace = os.path.join(a.out, f'{cfg.name}.trace.json')
    prof.export_chrome_trace(trace)

    # attribute GPU kernel time to enclosing stage via the exported trace
    ev = json.load(open(trace))['traceEvents']
    stages = [(e['ts'], e['ts'] + e['dur'], e['name'].split('::', 1)[1]) for e in ev
              if e.get('ph') == 'X' and e.get('name', '').startswith('stage::')]
    # kernels carry a correlation id to their launching CPU op; map by launch timestamp instead (simpler, adequate)
    launches = {e['args'].get('correlation'): e['ts'] for e in ev
                if e.get('ph') == 'X' and e.get('cat') in ('cuda_runtime', 'Runtime') and 'args' in e and 'correlation' in e['args']}
    gpu_by_stage, n_kernels, kernel_time = defaultdict(float), defaultdict(int), defaultdict(float)
    for e in ev:
        if e.get('ph') != 'X' or e.get('cat') not in ('kernel', 'Kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        ts = launches.get(e.get('args', {}).get('correlation'), e['ts'])
        for s0, s1, name in stages:
            if s0 <= ts <= s1:
                gpu_by_stage[name] += e['dur']; n_kernels[name] += 1; break
        kernel_time[e['name']] += e['dur']

    total_req = sum(e['dur'] for e in ev if e.get('ph') == 'X' and e.get('name') == 'request') / len(paths)
    print(f"\n{cfg.name}: {len(paths)} clips, mean request wall {total_req/1e3:.1f} ms\n")
    print(f"{'stage':12s} {'GPU busy ms':>12s} {'#kernels':>9s} {'mean us/kernel':>15s}")
    for s in STAGES:
        g, k = gpu_by_stage[s] / len(paths), n_kernels[s] / len(paths)
        print(f"{s:12s} {g/1e3:12.1f} {k:9.0f} {(g / k if k else 0):15.1f}")
    print(f"{'TOTAL':12s} {sum(gpu_by_stage.values())/len(paths)/1e3:12.1f} {sum(n_kernels.values())/len(paths):9.0f}")
    print(f"\nGPU duty within request: {100 * sum(gpu_by_stage.values()) / len(paths) / total_req:.0f}%  "
          f"(<100% => GPU idle gaps = launch/Python bound)")
    print("\ntop kernels (total ms across clips):")
    for name, d in sorted(kernel_time.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {d/1e3:8.1f}  {name[:100]}")
    print(f"\ntrace: {trace}  (open in chrome://tracing or https://ui.perfetto.dev)")


if __name__ == '__main__':
    main()
