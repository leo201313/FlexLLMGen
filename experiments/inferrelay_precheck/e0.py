"""Original OptLM generation loops, with separate event-instrumented diagnostic runs."""
import hashlib
import itertools
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from flexllmgen.flex_opt import OptLM, Policy, get_test_inputs
from flexllmgen.opt_config import get_opt_config
from flexllmgen.pytorch_backend import TorchDevice
from flexllmgen.utils import ExecutionEnv
from .common import (Memory, check_budget, cleanup, event, failure, stats,
                     weights_path, write_csv, write_json)


class DisabledDisk:
    """No disk policy is permitted. Avoid four unused stock 2-GiB pinned worker buffers."""
    def synchronize(self):
        pass

    def allocate(self, *args, **kwargs):
        raise RuntimeError('Disk offload is disabled in precheck')


class ManagedOptLM(OptLM):
    # Original destructor also calls delete_all_weights; make only cleanup idempotent.
    def delete_all_weights(self):
        for j, holder in enumerate(getattr(self, 'weight_home', [])):
            if holder.val is not None:
                self.delete_weight(j, 0)


def policy(batch, micro, mode):
    gpu_weights = mode == 'gpu'
    return Policy(batch, micro, 100 if gpu_weights else 0, 0 if gpu_weights else 100,
                  100, 0, 100, 0, mode != 'cpu_serial', True, True, False, 1.0,
                  False, None, False, None)


class Trace:
    def __init__(self, model):
        self.model = model
        self.records = []
        self.restores = []
        self.origin = event()
        self.origin.record()
        self.origin.synchronize()
        self.wall_origin = time.perf_counter()
        for j, layer in enumerate(model.layers):
            def wrap_load(original, j=j):
                def load(home, buf, k):
                    if k:
                        return original(home, buf, k)
                    a, b = event(), event()
                    a.record()
                    result = original(home, buf, k)
                    b.record()
                    self.records.append(('weight_load', j, -1, k, a, b))
                    return result
                return load
            self.patch(layer, 'load_weight', wrap_load)

        def wrap_compute(original):
            def compute(i, j, k):
                a, b = event(), event()
                a.record()
                result = original(i, j, k)
                b.record()
                self.records.append(('compute', j, i, k, a, b))
                return result
            return compute
        self.patch(model, 'compute_layer', wrap_compute)
        self.syncs = []
        def wrap_sync(original):
            def sync():
                a = time.perf_counter()
                original()
                b = time.perf_counter()
                self.syncs.append({'start_ms': (a-self.wall_origin)*1000,
                                   'duration_ms': (b-a)*1000})
            return sync
        self.patch(model, 'sync', wrap_sync)

    def patch(self, obj, name, factory):
        original = getattr(obj, name)
        self.restores.append((obj, name, original))
        setattr(obj, name, factory(original))

    def finish(self, path):
        torch.cuda.synchronize()
        rows = []
        loads = {}
        for kind, layer, token, micro, a, b in self.records:
            if kind == 'weight_load':
                token = loads.get(layer, 0)
                loads[layer] = token + 1
            start = self.origin.elapsed_time(a)
            end = self.origin.elapsed_time(b)
            rows.append({'kind': kind, 'layer': layer, 'token': token, 'micro': micro,
                         'start_ms': start, 'end_ms': end, 'duration_ms': end-start})
        for obj, name, original in self.restores:
            setattr(obj, name, original)
        self.restores.clear()
        self.model = None
        write_csv(path.with_suffix('.csv'), rows)
        # Perfetto/Chrome compatible; CPU clock and CUDA origin differ by origin synchronization.
        chrome = [{'name': f"{r['kind']} L{r['layer']} t{r['token']} b{r['micro']}",
                   'cat': r['kind'], 'ph': 'X', 'pid': 1,
                   'tid': 'weight submission stream' if r['kind'] == 'weight_load' else 'compute stream',
                   'ts': r['start_ms']*1000, 'dur': r['duration_ms']*1000} for r in rows]
        chrome += [{'name': 'original global sync (CPU wall)', 'ph': 'X', 'pid': 2,
                    'tid': 'host', 'ts': r['start_ms']*1000, 'dur': r['duration_ms']*1000}
                   for r in self.syncs]
        write_json(path.with_suffix('.json'), {'traceEvents': chrome,
                   'note': 'CUDA event spans include submission gaps; CPU origin approximate. No additional per-layer synchronization.'})
        # This diagnostic is deliberately NOT claimed to identify causal weight-only waiting.
        comp = sorted((r for r in rows if r['kind'] == 'compute'), key=lambda r:r['start_ms'])
        lookup = {(r['token'], r['layer']): r for r in rows if r['kind'] == 'weight_load'}
        gaps = []
        for prev, cur in zip(comp, comp[1:]):
            load = lookup.get((cur['token'], cur['layer']))
            if load and cur['micro'] == 0:
                gaps.append(max(0., min(load['end_ms'], cur['start_ms'])-prev['end_ms']))
        summary = {'weight_load_span_sum_ms': sum(r['duration_ms'] for r in rows if r['kind']=='weight_load'),
                   'compute_span_sum_ms': sum(r['duration_ms'] for r in comp),
                   'weight_ready_gap_proxy_ms': sum(gaps),
                   'original_sync_wall_sum_ms': sum(r['duration_ms'] for r in self.syncs),
                   'causal_weight_wait_ms': None,
                   'warning': 'Gap proxy can include host scheduling and allocator delay; global sync waits on all streams. GPU-resident weight_load spans are not H2D transfers.'}
        write_json(path.parent / (path.name + '_summary.json'), summary)
        return summary


def generate_sample(model, inputs, gen_len):
    markers = []
    original = model.update_attention_mask
    def update(i, k):
        if k == 0:
            e = event()
            e.record()
            markers.append(e)
        return original(i, k)
    model.update_attention_mask = update
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        output = model.generate(inputs, max_new_tokens=gen_len)
        end = event()
        end.record()
        end.synchronize()
    finally:
        model.update_attention_mask = original
    wall = (time.perf_counter()-t)*1000
    token_times = [a.elapsed_time(b) for a,b in zip(markers, markers[1:]+[end])]
    return output, {'wall_ms': wall, 'prefill_ms': token_times[0],
                    'decode_ms': sum(token_times[1:]), 'token_intervals_ms': token_times,
                    'output_tokens': len(inputs)*gen_len,
                    'output_tokens_per_s': len(inputs)*gen_len/(wall/1000)}


def run(cfg, out):
    weights_path(cfg)
    conf = get_opt_config(cfg['model'])
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b', padding_side='left', local_files_only=True)
    ec = cfg['e0']
    rows, summaries, reference = [], [], {}
    for prompt, batch, micro in itertools.product(ec['prompts'], ec['batches'], ec['microbatches']):
        for mode in ['gpu', 'cpu_serial', 'cpu_overlap']:
            case = f'{mode}_p{prompt}_b{batch}_m{micro}'
            print('E0', case, flush=True)
            model = None
            try:
                # Include embeddings, full-vocabulary prefill logits, caches, and an allocator margin.
                estimate = conf.model_bytes()*2 + conf.cache_bytes(batch*micro, prompt+ec['gen_len'])
                estimate += batch*prompt*conf.vocab_size*4 + 512*1024**2
                check_budget(cfg, estimate, conf.model_bytes()*3 + 1024**3)
                env = ExecutionEnv(TorchDevice('cuda:0'), TorchDevice('cpu'), DisabledDisk(), None)
                with Memory() as memory:
                    model = ManagedOptLM(conf, env, cfg['weights_root'], policy(batch,micro,mode))
                    inputs = get_test_inputs(prompt, batch*micro, tokenizer)
                    for _ in range(cfg['warmup']):
                        model.generate(inputs, max_new_tokens=ec['gen_len'])
                    torch.cuda.synchronize()
                    samples = []
                    start = time.perf_counter()
                    while len(samples)<cfg['repeats'] or time.perf_counter()-start<cfg['min_seconds']:
                        output, row = generate_sample(model, inputs, ec['gen_len'])
                        output = np.asarray(output).copy()
                        key = (prompt,batch,micro)
                        if mode == 'gpu':
                            reference.setdefault(key, output)
                        equal = bool(np.array_equal(output, reference[key]))
                        if not equal:
                            raise AssertionError('Generated IDs differ from GPU baseline')
                        row.update(case=case, mode=mode, prompt=prompt, batch=batch, microbatches=micro,
                                   sample=len(samples), kind='measured', output_matches_gpu=equal,
                                   output_sha256=hashlib.sha256(output.tobytes()).hexdigest())
                        samples.append(row)
                    pinned = sum(w.bytes for h in model.weight_home for w in h.val
                                 if w.data.device.type=='cpu' and w.data.is_pinned())
                for row in samples:
                    row.update(memory.values, pinned_weight_live_bytes=int(pinned))
                rows.extend(samples)
                summary = {'case':case, 'mode':mode, 'prompt':prompt, 'batch':batch, 'microbatches':micro,
                           'status':'ok', 'wall_ms':stats([r['wall_ms'] for r in samples]),
                           'prefill_ms':stats([r['prefill_ms'] for r in samples]),
                           'decode_ms':stats([r['decode_ms'] for r in samples]),
                           'memory':memory.values, 'pinned_weight_live_bytes':int(pinned)}
                if prompt == ec['trace_prompt'] and batch==1 and micro==1:
                    trace = Trace(model)
                    traced_output, traced = generate_sample(model, inputs, ec['gen_len'])
                    assert np.array_equal(traced_output, reference[(prompt,batch,micro)])
                    summary['trace'] = trace.finish(out / ('trace_'+case))
                    summary['trace']['wall_over_baseline_p50'] = traced['wall_ms']/summary['wall_ms']['p50']
                    summary['trace']['instrumented_samples'] = 1
                    write_json(out / (case+'_output.json'), {
                        'ids': traced_output.tolist(), 'texts': tokenizer.batch_decode(traced_output),
                        'note': 'Fixed-length greedy generation; no early EOS termination.'})
                summaries.append(summary)
            except Exception as exc:
                summaries.append(failure(out,'e0',case,exc))
                if not isinstance(exc,(MemoryError,torch.OutOfMemoryError)):
                    raise
            finally:
                if model is not None:
                    model.delete_all_weights()
                model = None
                cleanup()
            write_json(out/'summary.json', summaries)
            write_json(out/'raw.json', rows)
            write_csv(out/'raw.csv', [{**r,'token_intervals_ms':str(r['token_intervals_ms'])} for r in rows])
    return summaries
