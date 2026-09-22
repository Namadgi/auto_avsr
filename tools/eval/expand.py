"""Expand labeled clips into decision samples: samples.csv with columns
clip, phrase, kind (match | wrong | near), expected (0/1/2 when known from prod, else '').

    python -m tools.eval.expand --manifest manifest.csv --out samples.csv [--seed 472]

- match: the labeled phrase (expected pass unless prod said otherwise / no face).
- wrong: a random different digit phrase of the same length (expected fail).
- near : the labeled phrase with one digit swapped (near-threshold; expectation unknown, parity only).
"""
import argparse
import csv
import random

WORDS = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']


def wrong_phrase(words, rng):
    while True:
        cand = [rng.choice(WORDS) for _ in words]
        if sum(a != b for a, b in zip(cand, words)) >= max(2, len(words) // 2):
            return ' '.join(cand)


def near_phrase(words, rng):
    i = rng.randrange(len(words))
    cand = list(words)
    cand[i] = rng.choice([w for w in WORDS if w != words[i]])
    return ' '.join(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=472)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    out = []
    for r in csv.DictReader(open(a.manifest)):
        phrase = r['phrase'].strip().upper()
        if not phrase or r['label_source'].startswith(('TBD', 'REVIEW')):
            continue
        out.append({'clip': r['clip'], 'phrase': phrase, 'kind': 'match', 'expected': r.get('prod_code', '')})
        words = phrase.split()
        if all(w in WORDS for w in words) and len(words) >= 3:
            out.append({'clip': r['clip'], 'phrase': wrong_phrase(words, rng), 'kind': 'wrong', 'expected': ''})
            out.append({'clip': r['clip'], 'phrase': near_phrase(words, rng), 'kind': 'near', 'expected': ''})
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['clip', 'phrase', 'kind', 'expected']); w.writeheader(); w.writerows(out)
    from collections import Counter
    print(f"{len(out)} samples from {len({s['clip'] for s in out})} clips:", dict(Counter(s['kind'] for s in out)))


if __name__ == '__main__':
    main()
