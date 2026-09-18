"""Shared measurement utilities. No changes to FlexGen's production path."""
import csv
import gc
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20, cwd=ROOT)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"error": repr(e)}


def pcie():
    root = Path('/sys/bus/pci/devices/0000:01:00.0')
    return {p.name: p.read_text().strip() for p in root.glob('*link*') if p.is_file()}


def inventory(out, cfg):
    import transformers
    import importlib.metadata
    value = {
        "kind": "measured_environment", "time_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "cuda_arch_list": torch.cuda.get_arch_list(), "pcie_idle": pcie(),
        "ram": psutil.virtual_memory()._asdict(), "disk": psutil.disk_usage(ROOT)._asdict(),
        "config": cfg,
        "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    }
    for name, args in {
        "commit": ['git', 'rev-parse', 'HEAD'], "status": ['git', 'status', '--short'],
        "diff": ['git', 'diff', '--stat'], "cpu": ['lscpu'],
        "gpu_status": ['nvidia-smi'], "gpu_topology": ['nvidia-smi', 'topo', '-m'],
        "pci": ['lspci', '-vv', '-s', '01:00.0'],
        "interfaces": ['ip', '-br', 'link'],
        "ethernet": ['ethtool', 'enp4s0'],
    }.items():
        value[name] = command(args)
    write_json(out / 'environment.json', value)
    return value


class Memory:
    """RSS is sampled, not an exact allocator peak. Pinned tensor bytes supplied by callers."""
    def __enter__(self):
        self.process = psutil.Process()
        self.rss = self.process.memory_info().rss
        self.stop = threading.Event()
        torch.cuda.reset_peak_memory_stats()
        def sample():
            while not self.stop.wait(.01):
                self.rss = max(self.rss, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.rss = max(self.rss, self.process.memory_info().rss)
        self.values = {
            'rss_sampled_peak_bytes': self.rss,
            'gpu_allocated_peak_bytes': torch.cuda.max_memory_allocated(),
            'gpu_reserved_peak_bytes': torch.cuda.max_memory_reserved(),
            'rss_sampling_ms': 10,
            'pinned_allocator_peak_bytes': None,
        }
        host_stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
        if host_stats:
            try:
                self.values['host_allocator_stats'] = host_stats()
            except Exception:
                pass


def event():
    return torch.cuda.Event(enable_timing=True)


def measure(fn, cfg):
    """Whole-sample synchronization only. fn must join any side stream to current stream."""
    for _ in range(cfg['warmup']):
        fn()
    torch.cuda.synchronize()
    rows = []
    start = time.perf_counter()
    while len(rows) < cfg['repeats'] or time.perf_counter() - start < cfg['min_seconds']:
        a, b = event(), event()
        t = time.perf_counter()
        a.record()
        fn()
        b.record()
        b.synchronize()
        rows.append({'sample': len(rows), 'cuda_ms': a.elapsed_time(b),
                     'wall_ms': (time.perf_counter() - t) * 1000})
    return rows


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99))}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def check_budget(cfg, gpu_bytes=0, ram_bytes=0):
    free, _ = torch.cuda.mem_get_info()
    if gpu_bytes > min(cfg['gpu_budget_gib'] * GIB, free * .75):
        raise MemoryError(f'GPU preflight rejected {gpu_bytes} bytes')
    if ram_bytes > min(cfg['rss_budget_gib'] * GIB, psutil.virtual_memory().available * .5):
        raise MemoryError(f'RAM preflight rejected {ram_bytes} bytes')


def weights_path(cfg):
    p = Path(cfg['weights_root']).expanduser() / (cfg['model'].split('/')[-1] + '-np')
    if not (p / 'decoder.embed_positions.weight').exists():
        raise FileNotFoundError(f'Cached converted weights missing: {p}. Downloads are disabled.')
    return p


def failure(out, experiment, case, exc):
    entry = {'experiment': experiment, 'case': case,
             'status': 'oom_or_budget' if isinstance(exc, (MemoryError, torch.OutOfMemoryError)) else 'error',
             'error': repr(exc)}
    with open(out / 'failures.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')
    return entry
