"""Fill missing phrases in manifest.csv from the clips' audio track with Whisper.

    python -m tools.eval.label --clips DIR --manifest manifest.csv [--model small] [--force]

Rows with label_source starting with 'TBD' get phrase=<ASR digits> and label_source='asr'.
Rows with a phrase already set are verified: mismatch -> label_source gets ';asr-mismatch:<asr>'.
Rows whose ASR output is not a clean digit-word sequence are marked 'REVIEW' for a human look.
"""
import argparse
import csv
import os
import re

from tools.eval.preprocess import extract_audio

DIGITS = {'zero': 'ZERO', 'oh': 'ZERO', 'o': 'ZERO', 'one': 'ONE', 'two': 'TWO', 'to': 'TWO', 'too': 'TWO',
          'three': 'THREE', 'four': 'FOUR', 'for': 'FOUR', 'five': 'FIVE', 'six': 'SIX', 'seven': 'SEVEN',
          'eight': 'EIGHT', 'ate': 'EIGHT', 'nine': 'NINE',
          '0': 'ZERO', '1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR', '5': 'FIVE', '6': 'SIX', '7': 'SEVEN', '8': 'EIGHT', '9': 'NINE'}


def normalize_digits(text: str):
    """Return (phrase, clean). clean=False if any token is not a digit word."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out, clean = [], True
    for t in toks:
        if t in DIGITS:
            out.append(DIGITS[t])
        elif t.isdigit():
            out.extend(DIGITS[c] for c in t)
        else:
            clean = False
    return ' '.join(out), clean and bool(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--model', default='small')
    ap.add_argument('--force', action='store_true', help='re-label rows that already have a phrase')
    a = ap.parse_args()

    import whisper
    asr = whisper.load_model(a.model)

    rows = list(csv.DictReader(open(a.manifest)))
    cache = {}
    for r in rows:
        clip = os.path.join(a.clips, r['clip'])
        if clip not in cache:
            wav = extract_audio(clip)
            try:
                raw = asr.transcribe(wav, language='en', fp16=False)['text']
            finally:
                os.remove(wav)
            cache[clip] = (raw.strip(), *normalize_digits(raw))
        raw, phrase, clean = cache[clip]
        r['asr_raw'] = raw
        if r['label_source'].startswith('TBD') or a.force:
            r['phrase'] = phrase if clean else ''
            r['label_source'] = 'asr' if clean else f'REVIEW:{raw[:60]}'
        elif r['phrase'] and phrase and r['phrase'].upper() != phrase:
            r['label_source'] += f';asr-mismatch:{phrase}'
        print(f"{r['clip']:40s} {r['label_source']:28s} {r['phrase']:50s} | {raw[:60]}")

    fields = list(rows[0].keys())
    with open(a.manifest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    n_review = sum(r['label_source'].startswith('REVIEW') for r in rows)
    print(f"\n{len(rows)} rows, {n_review} need review")


if __name__ == '__main__':
    main()
