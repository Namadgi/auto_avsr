"""Decision-parity report between two run results.

    python -m tools.eval.compare results/baseline.json results/ctc_greedy.json [--show-all]

Gate = pass/fail code parity on every (clip, phrase) sample, not identical text.
"""
import argparse
import json
import statistics as st

import torchaudio


def wer(ref, hyp):
    r, h = (ref or '').upper().split(), (hyp or '').upper().split()
    return torchaudio.functional.edit_distance(r, h) / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline'); ap.add_argument('candidate')
    ap.add_argument('--show-all', action='store_true')
    a = ap.parse_args()
    B, C = (json.load(open(p)) for p in (a.baseline, a.candidate))
    bres = {r['clip']: r for r in B['results']}
    cres = {r['clip']: r for r in C['results']}
    common = [c for c in bres if c in cres]

    flips, score_deltas, wers, n = [], [], [], 0
    by_kind = {}
    for clip in common:
        b, c = bres[clip], cres[clip]
        wers.append(wer(b['transcript'], c['transcript']))
        cs = {s['phrase']: s for s in c['samples']}
        for sb in b['samples']:
            sc = cs.get(sb['phrase'])
            if not sc or 'code' not in sb or 'code' not in sc:
                continue
            n += 1
            k = by_kind.setdefault(sb['kind'], {'n': 0, 'flips': 0})
            k['n'] += 1
            score_deltas.append(sc['score'] - sb['score'])
            if sb['code'] != sc['code']:
                k['flips'] += 1
                flips.append((clip, sb['kind'], sb['phrase'], sb['code'], sc['code'], sb['score'], sc['score'], b['transcript'], c['transcript']))
            elif a.show_all:
                print(f"  = {clip} [{sb['kind']}] code {sb['code']} score {sb['score']}->{sc['score']}")

    bs, cs_ = B['summary'], C['summary']
    print(f"baseline : {bs['config']['name']}  candidate: {cs_['config']['name']}")
    print(f"clips compared: {len(common)}   samples compared: {n}")
    print(f"\nDECISION PARITY: {n - len(flips)}/{n} identical  ({len(flips)} flips)")
    for k, v in by_kind.items():
        print(f"  {k:6s}: {v['n'] - v['flips']}/{v['n']}")
    for f in flips:
        print(f"  FLIP {f[0]} [{f[1]}] phrase={f[2]!r} code {f[3]}->{f[4]} score {f[5]}->{f[6]}\n"
              f"       base={f[7]!r}\n       cand={f[8]!r}")
    if score_deltas:
        print(f"\nscore delta: mean {st.mean(score_deltas):+.2f}  min {min(score_deltas):+.2f}  max {max(score_deltas):+.2f}")
    print(f"transcript WER vs baseline: mean {st.mean(wers):.3f}, clips with any change: {sum(w > 0 for w in wers)}/{len(wers)}")

    def spd(key):
        b_, c_ = bs['stages_mean_s'].get(key), cs_['stages_mean_s'].get(key)
        return f"{key:12s} {b_:8.3f}s -> {c_:8.3f}s  ({(c_ / b_ if b_ else float('nan')):5.2f}x)"
    print(f"\ninfer mean: {bs['infer_s']['mean']:.3f}s -> {cs_['infer_s']['mean']:.3f}s  ({cs_['infer_s']['mean'] / bs['infer_s']['mean']:.2f}x)")
    for k in bs['stages_mean_s']:
        if k in cs_['stages_mean_s']:
            print('  ' + spd(k))
    if bs.get('peak_gpu_mem_mb') and cs_.get('peak_gpu_mem_mb'):
        print(f"peak GPU mem: {bs['peak_gpu_mem_mb']} MB -> {cs_['peak_gpu_mem_mb']} MB")


if __name__ == '__main__':
    main()
