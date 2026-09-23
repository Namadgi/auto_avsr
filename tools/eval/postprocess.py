"""Pure re-implementation of VSRService.postprocess / compare_texts (src/services/vsr_service.py)
without OpenTelemetry or model loading, so the eval harness can score transcripts offline.
Keep in sync with the service."""
import json
import re
from functools import lru_cache

import strsimpy

NLEV_THRESHOLD = 0.5
DESCRIPTIONS = ['Successful check', 'Phrase does not match', 'No face present']


@lru_cache(maxsize=1)
def _word_map(path='extra_files/word_map.json'):
    with open(path) as f:
        return json.load(f)


def apply_word_map(pred_text: str) -> str:
    pred_text = pred_text.strip().upper()
    for pat, repl in _word_map().items():
        if re.search(pat, pred_text):
            pred_text = re.sub(pat, repl, pred_text)
    return pred_text


def compare_texts(pred_text: str, target_text: str):
    pred_text, target_text = pred_text.strip().upper(), target_text.strip().upper()
    sim = strsimpy.NormalizedLevenshtein().similarity(pred_text, target_text)
    return int(sim < NLEV_THRESHOLD), sim


def postprocess(pred_text: str, target_text: str) -> dict:
    if pred_text == '':
        code, sim, result = 2, 0, ''
    else:
        result = apply_word_map(pred_text)
        code, sim = compare_texts(result, target_text)
    return {'code': code, 'description': DESCRIPTIONS[code], 'result': result, 'score': round(sim * 100, 2)}
