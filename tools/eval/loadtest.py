"""Constant-arrival-rate load against a running ml-vsr service while sampling GPU utilization.

    python3 tools/eval/loadtest.py --url http://localhost:8080 --clips DIR --rate 0.56 --duration 300 --out results/load_baseline.json

Stdlib only (runs on the VM host, outside the container). Posts clips round-robin to /upload,
records latency/status, and samples `nvidia-smi` utilization.gpu every second — the same
signal Cloud Run's container/gpu/utilizations is derived from.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import threading
import time
import urllib.request
import uuid


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def post_upload(url, path):
    boundary = uuid.uuid4().hex
    with open(path, 'rb') as f:
        data = f.read()
    ctype = 'video/webm' if path.endswith('.webm') else 'video/mp4'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
            f'Content-Type: {ctype}\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(f'{url}/upload', data=body, method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()[:200]
    except Exception as e:
        status, text = 0, repr(e)
    return time.perf_counter() - t0, status, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8080')
    ap.add_argument('--clips', required=True)
    ap.add_argument('--rate', type=float, default=0.56)
    ap.add_argument('--duration', type=int, default=300)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    clips = sorted(os.path.join(a.clips, f) for f in os.listdir(a.clips) if f.lower().endswith(('.mp4', '.webm', '.mov')))
    # warm-up + health
    print('warm-up:', post_upload(a.url, clips[0])[:2])

    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used',
                            '--format=csv,noheader,nounits', '-l', '1'], stdout=subprocess.PIPE, text=True)
    results, threads = [], []
    lock = threading.Lock()

    def worker(path):
        lat, status, text = post_upload(a.url, path)
        with lock:
            results.append({'clip': os.path.basename(path), 'latency_s': round(lat, 3), 'status': status,
                            'text': text[:120] if status != 200 else json.loads(text).get('result', '')[:60]})

    t_start = time.time()
    interval = 1.0 / a.rate
    i = 0
    while time.time() - t_start < a.duration:
        th = threading.Thread(target=worker, args=(clips[i % len(clips)],), daemon=True)
        th.start(); threads.append(th); i += 1
        time.sleep(max(0.0, t_start + i * interval - time.time()))
    for th in threads:
        th.join(timeout=150)
    t_end = time.time()
    smi.terminate()
    util = []
    for line in smi.stdout.read().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4 and parts[1].isdigit():
            util.append({'gpu': int(parts[1]), 'mem': int(parts[2]), 'mem_used_mb': int(parts[3])})

    gpu = [u['gpu'] for u in util]
    lat = [r['latency_s'] for r in results if r['status'] == 200]
    summary = {
        'label': a.label, 'url': a.url, 'rate_target': a.rate, 'duration_s': round(t_end - t_start, 1),
        'requests': len(results), 'rate_actual': round(len(results) / (t_end - t_start), 3),
        'errors': sum(r['status'] != 200 for r in results),
        'gpu_util': {'mean': round(st.mean(gpu), 1), 'p50': pct(gpu, .5), 'p90': pct(gpu, .9), 'p95': pct(gpu, .95),
                     'p99': pct(gpu, .99), 'max': max(gpu), 'samples': len(gpu)} if gpu else None,
        'gpu_mem_util_mean': round(st.mean(u['mem'] for u in util), 1) if util else None,
        'gpu_mem_used_mb_max': max(u['mem_used_mb'] for u in util) if util else None,
        'latency_s': {'mean': round(st.mean(lat), 3), 'p50': pct(lat, .5), 'p95': pct(lat, .95), 'p99': pct(lat, .99), 'max': max(lat)} if lat else None,
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'summary': summary, 'gpu_samples': util, 'requests': results}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
