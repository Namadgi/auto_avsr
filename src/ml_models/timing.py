import time
from contextlib import contextmanager

import torch


class StageTimer:
    """Collects per-stage wall-clock timings. When `sync` is set, synchronizes CUDA
    before and after each stage so GPU work is attributed to the stage that launched it."""

    def __init__(self, device=None, sync=False):
        self.sync = bool(sync) and device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
        self.timings = {}

    def _sync(self):
        if self.sync:
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t0)

    def reset(self):
        self.timings = {}
