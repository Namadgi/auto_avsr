import json
from dataclasses import dataclass, asdict, fields

import torch

_DTYPES = {None: None, 'fp16': torch.float16, 'bf16': torch.bfloat16}


@dataclass
class EvalConfig:
    """One VSR configuration under test. Defaults == production."""
    name: str = 'baseline'
    device: str = 'cuda'
    beam_size: int = 40
    ctc_weight: float = 0.1
    decode_mode: str = 'beam'       # beam | ctc_greedy
    autocast: str = None            # None | fp16 | bf16
    skip_step: int = 5
    weights: str = 'src/model_weights/vsr.pth'

    @classmethod
    def load(cls, path: str) -> 'EvalConfig':
        with open(path) as f:
            d = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'{path}: unknown keys {sorted(unknown)}')
        return cls(**d)

    def build_model(self):
        from src.ml_models.vsr import VSR
        model = VSR(device=self.device, skip_step=self.skip_step, sync_timing=True,
                    beam_size=self.beam_size, ctc_weight=self.ctc_weight,
                    decode_mode=self.decode_mode, autocast_dtype=_DTYPES[self.autocast])
        state = torch.load(self.weights, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model

    def to_dict(self):
        return asdict(self)
