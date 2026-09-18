"""Generate a compact report without discarding poor configurations."""
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def generate(out):
    out=Path(out)
    lines=['# InferRelay precheck: single-machine results','',
           'Measured E0–E2; E3 is a parameterized prediction. OPT-125M forced offload is a mechanism test, not evidence of large-model necessity or two-machine speedup.', '']
    e0=load(out/'e0/summary.json')
    if e0:
        lines += ['## E0: original generation loops','',
                  '|case|wall P50/P95 ms|prefill P50 ms|decode P50 ms|GPU peak MiB|',
                  '|---|---:|---:|---:|---:|']
        for s in e0:
            if s['status']!='ok':
                lines.append(f"|{s.get('case')}|{s['status']}||||");continue
            lines.append(f"|{s['case']}|{s['wall_ms']['p50']:.3f}/{s['wall_ms']['p95']:.3f}|{s['prefill_ms']['p50']:.3f}|{s['decode_ms']['p50']:.3f}|{s['memory']['gpu_allocated_peak_bytes']/2**20:.1f}|")
        lines += ['', 'All successful modes require exact generated-ID equality with the all-GPU baseline. Phase times are CUDA-event spans, including host launch gaps; wall time includes generation setup/cleanup. Fixed-length throughput is not online goodput.', '',
                  'Trace CSV/JSON contains weight submission spans, layer compute and original CPU global-sync waits. Weight-ready gaps are a diagnostic proxy, not causal H2D wait attribution. Trace runs are separate from the baseline samples.', '']
        for s in e0:
            if 'trace' in s:
                t=s['trace']
                lines.append(f"- {s['case']}: trace wall/baseline P50 = {t['wall_over_baseline_p50']:.3f}; weight-ready gap proxy = {t['weight_ready_gap_proxy_ms']:.3f} ms. Single diagnostic sample; do not correct throughput by this ratio.")
        lines += ['', 'Decision: the baseline and timing path are executable. Inspect the full CPU serial/overlap table before claiming remaining prefetch opportunity; do not extrapolate the small-model ratio to a memory-constrained model.', '']
    e1=load(out/'e1/summary.json')
    if e1:
        lines += ['## E1: actual block-layout transfer and computation','',
                  '|blocks|FP16 weights MiB|pageable P50 ms|pinned P50 ms|pinned GB/s|',
                  '|---|---:|---:|---:|---:|']
        for s in e1:
            if s['status']!='ok':continue
            a=s['transfers']['actual_tensors_pageable'];b=s['transfers']['actual_tensors_pinned']
            lines.append(f"|{s['group']}|{s['weight_bytes']/2**20:.2f}|{a['cuda_ms']['p50']:.3f}|{b['cuda_ms']['p50']:.3f}|{b['effective_GBps']:.2f}|")
        ratios=[c['h2d_over_compute'] for s in e1 if s['status']=='ok' for c in s['compute']]
        lines += ['',f'Measured H2D/compute ratios span {min(ratios):.3f}–{max(ratios):.3f}. All query/batch/history combinations are retained in summary.json; raw.csv includes every sample.', '',
                  'Decision: retain near-matching shapes for E2 and the network sweep. q>1 is ordinary short-prompt prefill with past=0, not incremental/chunked prefill. Decode uses real attention KV prepared from synthetic hidden activations through actual pretrained blocks; it is a kernel workload, not full-model semantic validation.', '']
    e2=load(out/'e2/summary.json')
    if e2:
        valid=[s for s in e2 if s['status']=='ok']
        fast=sum(s['metrics']['concurrent_speedup_over_serial']>1.01 for s in valid)
        worse=sum(s['metrics']['concurrent_speedup_over_serial']<.99 for s in valid)
        lines += ['## E2: measured concurrency','',f'{fast}/{len(valid)} shapes have >1% speedup versus serial; {worse}/{len(valid)} regress by >1%. These thresholds summarize measurements, not statistical significance.', '',
                  '|blocks|query|past|batch|serial/concurrent P50 ms|speedup|copy slowdown|compute slowdown|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for s in valid:
            m=s['metrics']
            lines.append(f"|{s['group']}|{s['q']}|{s['past']}|{s['batch']}|{m['serial']['cuda_ms']['p50']:.3f}/{m['concurrent']['cuda_ms']['p50']:.3f}|{m['concurrent_speedup_over_serial']:.3f}|{m['copy_slowdown']:.3f}|{m['compute_slowdown']:.3f}|")
        lines += ['', 'Overlap efficiency = (isolated_copy + isolated_compute - concurrent_makespan) / min(isolated_copy, isolated_compute); it is intentionally not clamped. Independent CUDA streams do not guarantee a win. Isolated references were measured earlier in E1; DVFS and temperature drift can affect slowdown ratios.', '',
                  'Two-slot lifecycle traces check copy→read and prior-read→reuse dependencies and exact numerical output. The pipeline repeatedly computes the same independent group workload; it is not full model inference. Original FlexGen overlap is the E0 cpu_overlap arm, not a like-for-like component makespan.', '',
                  'Decision: use both the isolated and measured-contention scenarios in E3. Neither a positive component speedup nor an event-ordering check establishes relay end-to-end benefit.', '']
    e3=load(out/'e3/summary.json')
    if e3:
        lines += ['## E3: modeled conditions only','',f"{e3['scenarios']} scenarios; {e3['predicted_faster']} predict >1% improvement and {e3['predicted_slower']} predict >1% regression. Predicted speedup range: {e3['speedup_range'][0]:.3f}–{e3['speedup_range'][1]:.3f}.", '',
                  'These counts depend on the chosen grid and are not probabilities. predictions.csv retains losses and ties. example_timeline.json includes resource reservations, explicit network/staging and buffer dependencies.', '']
        lines += ['- '+s for s in e3['limitations']]
        lines += ['', 'Decision: a second machine and real network measurements are necessary. Do not claim relay or SLO benefit from this model. Validate a predicted favorable and unfavorable shape using identical baselines and weight-reuse policies.', '']
    lines += ['## Next minimum changes','',
              '1. Repeat representative configurations with longer sampling and randomized order; keep both positive and negative shapes.',
              '2. Obtain approval before downloading OPT-1.3B; this still fits 16GB and is only a larger mechanism test.',
              '3. Implement and numerically validate incremental prefill before describing a q sweep as chunked prefill.',
              '4. Measure a real second endpoint and calibrate staging, H2D contention, weight reuse and queue scheduling before making a system claim.', '',
              'Memory: GPU allocated/reserved peaks are allocator measurements. RSS is sampled every 10ms. Explicit pinned tensor bytes are tracked; absent host allocator statistics are null, not zero. PyTorch pinned caching can retain more physical RAM than the live tensors. No deliberate OOM stress was performed; rejected/failing cases are written to failures.jsonl when present.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import sys
    generate(Path(sys.argv[1]))
