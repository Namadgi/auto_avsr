"""BM-522: vsr handler must not block the event loop.

Heavy deps (torch, the VSR model, GCP exporters) are stubbed so these run on a
plain CPU box: `pip install fastapi httpx pytest pytest-asyncio
opentelemetry-sdk strsimpy numpy python-multipart` then `pytest tests/`.
"""
import asyncio
import os
import sys
import threading
import time
import types

import pytest

os.environ["VSR_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _Cuda:
    @staticmethod
    def is_available():
        return False


_stub("torch", cuda=_Cuda, device=lambda t: types.SimpleNamespace(type=t), no_grad=lambda: _NullCtx(), load=lambda *a, **k: {})
_stub("src.ml_models.vsr", VSR=object)
_stub("opentelemetry.exporter.cloud_trace", CloudTraceSpanExporter=object)
_stub("opentelemetry.exporter.cloud_monitoring", CloudMonitoringMetricsExporter=object)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


from src.services import vsr_service as vsr_mod  # noqa: E402

# Skip real model loading; mark service ready.
def _fake_initialize(self):
    self.initialized = True
    self.device = types.SimpleNamespace(type="cpu")
    self.model = types.SimpleNamespace(eval=lambda: None)


vsr_mod.VSRService.initialize = _fake_initialize

import app as app_mod  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

SERVICE = app_mod.vsr_service
BODY = {"instances": [{"token": "t", "bucket_name": "b", "object_name": "o/v.mp4", "phrase": "HELLO"}]}
INFER_SECONDS = 1.5

_spans = InMemorySpanExporter()
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(_spans))


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch, tmp_path):
    state = {"active": 0, "max_active": 0, "workdirs": [], "max_live_workdirs": 0}
    lock = threading.Lock()

    def preprocess_into(data, workdir):
        state["workdirs"].append(workdir)
        live = sum(os.path.exists(d) for d in state["workdirs"])
        state["max_live_workdirs"] = max(state["max_live_workdirs"], live)
        out = os.path.join(workdir, "vsr_output.mp4")
        open(out, "wb").close()
        return out, data.instances[0].phrase, []

    def inference(file_path):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        with SERVICE.tracer.start_as_current_span("fake_model_inference"):
            time.sleep(INFER_SECONDS)  # blocking, like torch
        with lock:
            state["active"] -= 1
        return "HELLO"

    monkeypatch.setattr(SERVICE, "_preprocess_into", preprocess_into)
    monkeypatch.setattr(SERVICE, "inference", inference)
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # word_map.json
    _spans.clear()
    # Each test runs on a fresh event loop; drop the lazily created semaphore.
    monkeypatch.setattr(SERVICE, "_admission", None)
    return state


def _client():
    return AsyncClient(transport=ASGITransport(app=app_mod.app), base_url="http://t")


@pytest.mark.asyncio
async def test_health_not_blocked_by_inference():
    async with _client() as c:
        req = asyncio.create_task(c.post("/predictions/vsr", json=BODY))
        await asyncio.sleep(0.2)  # request is now mid-inference
        t0 = time.perf_counter()
        r = await c.get("/health")
        health_latency = time.perf_counter() - t0
        assert not req.done(), "inference finished too early to prove anything"
        resp = await req
    assert r.status_code == 200
    assert health_latency < 0.3, f"/health took {health_latency:.2f}s behind inference"
    assert resp.status_code == 200
    assert resp.json()["code"] == 0


@pytest.mark.asyncio
async def test_inference_serialized_on_gpu_pool(fake_pipeline):
    async with _client() as c:
        rs = await asyncio.gather(*(c.post("/predictions/vsr", json=BODY) for _ in range(3)))
    assert all(r.status_code == 200 for r in rs)
    assert fake_pipeline["max_active"] == 1
    for d in fake_pipeline["workdirs"]:
        assert not os.path.exists(d), "per-request workdir leaked"


@pytest.mark.asyncio
async def test_admission_bounds_workdirs_on_disk(fake_pipeline):
    limit = SERVICE._max_inflight
    n = limit + 3
    async with _client() as c:
        rs = await asyncio.gather(*(c.post("/predictions/vsr", json=BODY) for _ in range(n)))
    assert all(r.status_code == 200 for r in rs)
    assert len(fake_pipeline["workdirs"]) == n
    # Without the bound all n preprocess at once and n workdirs coexist.
    assert fake_pipeline["max_live_workdirs"] <= limit


@pytest.mark.asyncio
async def test_health_responsive_while_admission_full():
    async with _client() as c:
        reqs = [asyncio.create_task(c.post("/predictions/vsr", json=BODY))
                for _ in range(SERVICE._max_inflight + 2)]
        await asyncio.sleep(0.2)
        t0 = time.perf_counter()
        r = await c.get("/health")
        latency = time.perf_counter() - t0
        await asyncio.gather(*reqs)
    assert r.status_code == 200 and latency < 0.3


@pytest.mark.asyncio
async def test_http_exception_status_preserved(monkeypatch):
    def boom(data, workdir):
        raise HTTPException(status_code=404, detail="Failed to download file from GCS")

    monkeypatch.setattr(SERVICE, "_preprocess_into", boom)
    async with _client() as c:
        r = await c.post("/predictions/vsr", json=BODY)
    assert r.status_code == 404
    assert r.json()["detail"] == "Failed to download file from GCS"


@pytest.mark.asyncio
async def test_preprocess_failure_cleans_workdir(monkeypatch):
    seen = []

    def boom(data, workdir):
        seen.append(workdir)
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(SERVICE, "_preprocess_into", boom)
    async with _client() as c:
        r = await c.post("/predictions/vsr", json=BODY)
    assert r.status_code == 500
    assert seen and not os.path.exists(seen[0])


@pytest.mark.asyncio
async def test_spans_stay_nested_across_threads():
    async with _client() as c:
        r = await c.post("/predictions/vsr", json=BODY)
    assert r.status_code == 200
    by_name = {s.name: s for s in _spans.get_finished_spans()}
    root = by_name["vsr_process_video"]
    for child in ("preprocess", "inference", "postprocess"):
        assert by_name[child].parent.span_id == root.context.span_id
    # Span opened inside the GPU worker thread must parent to "inference".
    assert by_name["fake_model_inference"].parent.span_id == by_name["inference"].context.span_id
