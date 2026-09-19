"""E1/E2: actual OPT block tensors and the unmodified FlexGen mha/mha_gen/mlp kernels."""
import itertools
import time

import numpy as np
import torch

# flex_opt initializes the backend's recursive imports.
import flexllmgen.flex_opt
from flexllmgen.opt_config import get_opt_config
from flexllmgen.pytorch_backend import TorchDevice, TorchTensor
from .common import (Memory, check_budget, cleanup, event, failure, measure, pcie,
                     stats, weights_path, write_csv, write_json)

NAMES = [f'self_attn.{proj}.{kind}' for proj in ['q_proj','k_proj','v_proj','out_proj']
         for kind in ['weight','bias']] + ['self_attn_layer_norm.weight','self_attn_layer_norm.bias',
          'fc1.weight','fc1.bias','fc2.weight','fc2.bias','final_layer_norm.weight','final_layer_norm.bias']


def nbytes(tensors):
    return sum(t.numel()*t.element_size() for t in tensors)


def copy_into(dst, src):
    for d,s in zip(dst,src):
        d.copy_(s, non_blocking=True)


class Blocks:
    def __init__(self, cfg, count):
        self.config = get_opt_config(cfg['model'])
        assert count <= self.config.num_hidden_layers
        p = weights_path(cfg)
        self.pageable = [torch.from_numpy(np.load(p/f'decoder.layers.{i}.{name}')).to(torch.float16)
                         for i in range(count) for name in NAMES]
        start = time.perf_counter()
        self.pinned = [x.pin_memory() for x in self.pageable]
        self.first_pin_prepare_ms = (time.perf_counter()-start)*1000
        self.bytes = nbytes(self.pageable)
        self.gpu = TorchDevice('cuda:0')
        self.resident = [t.to('cuda') for t in self.pageable]
        self.wrapped = self.wrap(self.resident)
        self.count = count

    def wrap(self, ts):
        return [[TorchTensor.create_from_torch(t,self.gpu) for t in ts[i:i+16]]
                for i in range(0,len(ts),16)]

    def prefill_block(self, h, mask, ws):
        h,k,v = self.gpu.mha(h,mask,*ws[:10],self.config.n_head,[False]*12,False,None)
        h = self.gpu.mlp(h,*ws[10:],[False]*7)
        return h,k,v

    def state(self, q, past, batch):
        if q>1 and past:
            raise ValueError('Incremental prefill is not implemented; do not mislabel ordinary prefill')
        h = self.config.input_dim
        x = torch.randn(batch,q,h,device='cuda',dtype=torch.float16)*.1
        wrap = lambda t: TorchTensor.create_from_torch(t,self.gpu)
        mask = wrap(torch.ones(batch,q+past,device='cuda',dtype=torch.bool))
        caches = []
        if past:
            # Real KV from full prefill through the same block group, outside timed region.
            hp = wrap(torch.randn(batch,past,h,device='cuda',dtype=torch.float16)*.1)
            pm = wrap(torch.ones(batch,past,device='cuda',dtype=torch.bool))
            for ws in self.wrapped:
                hp,k,v = self.prefill_block(hp,pm,ws)
                pair = []
                for kv in [k,v]:
                    target = torch.empty((past+1,)+tuple(kv.data.shape[1:]),device='cuda',dtype=torch.float16)
                    target[:past].copy_(kv.data)
                    pair.append(wrap(target))
                caches.append(pair)
        return (wrap(x), mask, caches)

    def compute(self, state, weights=None):
        x,mask,caches = state
        kvs = []
        for i,ws in enumerate(self.wrapped if weights is None else weights):
            if caches:
                x,k,v = self.gpu.mha_gen(x,mask,*ws[:10],self.config.n_head,
                                         *caches[i],[False]*14,1.,False,None)
                x = self.gpu.mlp(x,*ws[10:],[False]*7)
            else:
                x,k,v = self.prefill_block(x,mask,ws)
            kvs.append((k,v))
        return x,kvs


def shapes(cfg):
    c = cfg['components']
    return [(q,0,b,'short_prefill_proxy') for q,b in itertools.product(c['prefill_queries'],c['batches'])] + [
        (1,p,b,'decode_real_history') for p,b in itertools.product(c['decode_past'],c['batches'])]


def pin_preparation(blocks,cfg):
    values = []
    for _ in range(cfg['warmup']+cfg['repeats']):
        start=time.perf_counter()
        ts=[x.pin_memory() for x in blocks.pageable]
        values.append((time.perf_counter()-start)*1000)
        del ts
    return values[cfg['warmup']:]


def run_e1(cfg,out):
    all_rows,summary = [],[]
    for g in cfg['components']['groups']:
        print('E1 group',g,flush=True)
        blocks=None
        try:
            conf=get_opt_config(cfg['model'])
            est=g*(12*conf.input_dim**2+13*conf.input_dim)*2
            check_budget(cfg,est*4+512*1024**2,est*6+1024**3)
            with Memory() as mem:
                blocks=Blocks(cfg,g)
                prep=pin_preparation(blocks,cfg)
                write_json(out/f'pin_prepare_g{g}.json',{
                    'first_pin_prepare_ms':blocks.first_pin_prepare_ms,
                    'warmed_allocator_allocate_and_copy_ms':prep,
                    'note':'Not OS-cold allocation. Includes pageable-to-pinned CPU copy; excluded from steady-state H2D.'})
                dst=[torch.empty_like(x,device='cuda') for x in blocks.pageable]
                transfer={}
                for layout in ['actual_tensors','contiguous_slab']:
                    for host in ['pageable','pinned']:
                        source=blocks.pageable if host=='pageable' else blocks.pinned
                        if layout=='contiguous_slab':
                            source=[torch.cat([t.reshape(-1) for t in blocks.pageable])]
                            if host=='pinned':
                                source=[source[0].pin_memory()]
                            targets=[torch.empty_like(source[0],device='cuda')]
                        else:
                            targets=dst
                        rows=measure(lambda:copy_into(targets,source),cfg)
                        for row in rows:
                            row.update(kind='measured',phase='h2d',group=g,layout=layout,host=host,
                                       weight_bytes=blocks.bytes,
                                       pinned_source_live_bytes=blocks.bytes*(1+int(layout=='contiguous_slab' and host=='pinned')))
                        all_rows.extend(rows)
                        key=layout+'_'+host
                        transfer[key]={m:stats([r[m] for r in rows]) for m in ['cuda_ms','wall_ms']}
                        transfer[key]['effective_GBps']=blocks.bytes/transfer[key]['cuda_ms']['p50']/1e6
                        transfer[key]['pcie_after_transfer']=pcie()
                        del source,targets
                del dst
                stage_allocated_peak=torch.cuda.max_memory_allocated()
                stage_reserved_peak=torch.cuda.max_memory_reserved()
                comp=[]
                for q,p,b,semantic in shapes(cfg):
                    with Memory() as shape_mem:
                        state=blocks.state(q,p,b)
                        rows=measure(lambda:blocks.compute(state),cfg)
                    stage_allocated_peak=max(stage_allocated_peak,shape_mem.values['gpu_allocated_peak_bytes'])
                    stage_reserved_peak=max(stage_reserved_peak,shape_mem.values['gpu_reserved_peak_bytes'])
                    for row in rows:
                        row.update(kind='measured',phase='compute',group=g,q=q,past=p,batch=b,
                                   semantic=semantic,weight_bytes=blocks.bytes,
                                   gpu_allocated_peak_bytes=shape_mem.values['gpu_allocated_peak_bytes'],
                                   gpu_reserved_peak_bytes=shape_mem.values['gpu_reserved_peak_bytes'],
                                   rss_sampled_peak_bytes=shape_mem.values['rss_sampled_peak_bytes'])
                    all_rows.extend(rows)
                    cs={m:stats([r[m] for r in rows]) for m in ['cuda_ms','wall_ms']}
                    cs.update(q=q,past=p,batch=b,semantic=semantic,
                              memory=shape_mem.values,
                              h2d_over_compute=transfer['actual_tensors_pinned']['cuda_ms']['p50']/cs['cuda_ms']['p50'])
                    comp.append(cs)
                    del state
            mem.values['gpu_allocated_peak_bytes']=stage_allocated_peak
            mem.values['gpu_reserved_peak_bytes']=stage_reserved_peak
            summary.append({'group':g,'weight_bytes':blocks.bytes,'status':'ok',
                            'transfers':transfer,'compute':comp,'memory':mem.values,
                            'pinned_weight_live_bytes':blocks.bytes,
                            'max_explicit_pinned_live_bytes':blocks.bytes*2})
        except Exception as exc:
            summary.append(failure(out,'e1',g,exc))
            if not isinstance(exc,(MemoryError,torch.OutOfMemoryError)):
                raise
        finally:
            blocks=None
            cleanup()
        write_csv(out/'raw.csv',all_rows)
        write_json(out/'summary.json',summary)
    return summary


def concurrent_sample(blocks,state,dst,copy_stream,compute_stream,mode):
    """All weights for computation are separate from the concurrently copied destination."""
    start,ce,cb,ke,kb,end=[event() for _ in range(6)]
    wall=time.perf_counter()
    start.record()
    copy_stream.wait_event(start)
    compute_stream.wait_event(start)
    with torch.cuda.stream(copy_stream):
        cb.record()
        copy_into(dst,blocks.pinned)
        ce.record()
    if mode=='serial':
        compute_stream.wait_event(ce)
    with torch.cuda.stream(compute_stream):
        kb.record()
        result=blocks.compute(state)
        ke.record()
    torch.cuda.current_stream().wait_event(ce)
    torch.cuda.current_stream().wait_event(ke)
    end.record()
    end.synchronize()
    return {'copy_ms':cb.elapsed_time(ce),'compute_ms':kb.elapsed_time(ke),
            'cuda_ms':start.elapsed_time(end),'wall_ms':(time.perf_counter()-wall)*1000,
            'copy_start_ms':start.elapsed_time(cb),'compute_start_ms':start.elapsed_time(kb)}, result[0].data


def buffer_pipeline(blocks,state,cfg):
    """Two reusable weight slots. Each copy waits for prior reader; every reader waits for ready."""
    slots=[[torch.empty_like(t,device='cuda') for t in blocks.pinned] for _ in range(2)]
    wrapped=[blocks.wrap(s) for s in slots]
    copy_stream,compute_stream=torch.cuda.Stream(),torch.cuda.Stream()
    def iteration(trace=False):
        start,end=event(),event()
        ready=[None,None]
        used=[None,None]
        records=[]
        start.record()
        copy_stream.wait_event(start)
        compute_stream.wait_event(start)
        for step in range(cfg['components']['buffer_steps']):
            slot=step%2
            c0,c1,k0,k1,bw0,bw1,rw0,rw1=[event() for _ in range(8)]
            with torch.cuda.stream(copy_stream):
                bw0.record()
                if used[slot] is not None:
                    copy_stream.wait_event(used[slot])
                bw1.record()
                c0.record()
                copy_into(slots[slot],blocks.pinned)
                c1.record()
                ready[slot]=c1
            with torch.cuda.stream(compute_stream):
                rw0.record()
                compute_stream.wait_event(ready[slot])
                rw1.record()
                k0.record()
                result=blocks.compute(state,wrapped[slot])
                k1.record()
                used[slot]=k1
            records.append((step,slot,c0,c1,k0,k1,bw0,bw1,rw0,rw1))
        torch.cuda.current_stream().wait_event(k1)
        end.record()
        if trace:
            end.synchronize()
            rows=[]
            for step,slot,c0,c1,k0,k1,bw0,bw1,rw0,rw1 in records:
                rows.append({'step':step,'slot':slot,'copy_start_ms':start.elapsed_time(c0),
                             'copy_end_ms':start.elapsed_time(c1),'compute_start_ms':start.elapsed_time(k0),
                             'compute_end_ms':start.elapsed_time(k1),'buffer_reuse_wait_ms':bw0.elapsed_time(bw1),
                             'data_ready_wait_ms':rw0.elapsed_time(rw1)})
            # Check event ordering, not just apparently sensible output.
            for i,row in enumerate(rows):
                assert row['compute_start_ms']+.02>=row['copy_end_ms']
                if i>=2:
                    assert row['copy_start_ms']+.02>=rows[i-2]['compute_end_ms']
            return rows,result[0].data.clone()
    rows=measure(iteration,cfg)
    trace,output=iteration(True)
    reference=blocks.compute(state)[0].data
    torch.testing.assert_close(output,reference,atol=0,rtol=0)
    # These slots are microbenchmark-only; the resident group is a reference overhead.
    return rows,trace


def run_e2(cfg,out,e1):
    all_rows,summary=[],[]
    for g in cfg['components']['groups']:
        print('E2 group',g,flush=True)
        blocks=None
        try:
            conf=get_opt_config(cfg['model'])
            est=g*(12*conf.input_dim**2+13*conf.input_dim)*2
            check_budget(cfg,est*5+512*1024**2,est*4+1024**3)
            blocks=Blocks(cfg,g)
            dst=[torch.empty_like(t,device='cuda') for t in blocks.pinned]
            cs,ks=torch.cuda.Stream(),torch.cuda.Stream()
            baseline=next(s for s in e1 if s.get('group')==g and s.get('status')=='ok')
            for q,p,b,semantic in shapes(cfg):
                with Memory() as mem:
                    state=blocks.state(q,p,b)
                    alone_copy=baseline['transfers']['actual_tensors_pinned']['cuda_ms']['p50']
                    alone_compute=next(s['cuda_ms']['p50'] for s in baseline['compute'] if (s['q'],s['past'],s['batch'])==(q,p,b))
                    metrics={}
                    for mode in ['serial','concurrent']:
                        for _ in range(cfg['warmup']):
                            concurrent_sample(blocks,state,dst,cs,ks,mode)
                        samples=[]
                        start=time.perf_counter()
                        while len(samples)<cfg['repeats'] or time.perf_counter()-start<cfg['min_seconds']:
                            row,output=concurrent_sample(blocks,state,dst,cs,ks,mode)
                            row.update(kind='measured',group=g,q=q,past=p,batch=b,semantic=semantic,mode=mode,sample=len(samples))
                            samples.append(row)
                        metrics[mode]={m:stats([r[m] for r in samples]) for m in ['cuda_ms','wall_ms','copy_ms','compute_ms']}
                        all_rows.extend(samples)
                    for d,s in zip(dst,blocks.pinned):
                        torch.testing.assert_close(d.cpu(),s,atol=0,rtol=0)
                    concurrent=metrics['concurrent']
                    ms=concurrent['cuda_ms']['p50']
                    serial_ms=metrics['serial']['cuda_ms']['p50']
                    metrics.update(alone_copy_ms=alone_copy,alone_compute_ms=alone_compute,
                                   overlap_efficiency=(alone_copy+alone_compute-ms)/min(alone_copy,alone_compute),
                                   concurrent_speedup_over_serial=serial_ms/ms,
                                   copy_slowdown=concurrent['copy_ms']['p50']/alone_copy,
                                   compute_slowdown=concurrent['compute_ms']['p50']/alone_compute)
                    # Representative lifecycle test for both phases and every group size.
                    if b==1 and ((q==128 and p==0) or (q==1 and p==256)):
                        pipe_rows,trace=buffer_pipeline(blocks,state,cfg)
                        for r in pipe_rows:
                            r.update(kind='measured',mode='two_slot_pipeline',group=g,q=q,past=p,batch=b)
                        all_rows.extend(pipe_rows)
                        write_csv(out/f'buffer_g{g}_q{q}_p{p}.csv',trace)
                        metrics['buffer_pipeline_cuda_ms']=stats([r['cuda_ms'] for r in pipe_rows])
                        metrics['buffer_checks']='event ordering and exact output PASS'
                        metrics['buffer_trace']=f'buffer_g{g}_q{q}_p{p}.csv'
                summary.append({'status':'ok','group':g,'q':q,'past':p,'batch':b,'semantic':semantic,
                                'metrics':metrics,'memory':mem.values,'pinned_weight_live_bytes':blocks.bytes,
                                'next_group_buffer_bytes':blocks.bytes,
                                'two_slots_bytes':blocks.bytes*2,
                                'note':'Memory peak includes resident reference weights and diagnostic destinations; not a deployed two-buffer footprint.'})
                del state,output
                write_json(out/'summary.json',summary)
                write_csv(out/'raw.csv',all_rows)
            del dst,cs,ks
        except Exception as exc:
            summary.append(failure(out,'e2',g,exc))
            write_json(out/'summary.json',summary)
            if not isinstance(exc,(MemoryError,torch.OutOfMemoryError)):
                raise
        finally:
            blocks=None
            cleanup()
    return summary
