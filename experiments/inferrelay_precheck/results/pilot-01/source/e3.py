"""Deterministic resource-constrained list scheduling; all outputs are predictions."""
import itertools
import math

from flexllmgen.opt_config import get_opt_config
from .common import GIB, stats, write_csv, write_json


def schedule(groups, owners, requests=1, arrival_gap_ms=0, cycles=1, buffers=2,
             bandwidth_gbps=10, latency_us=100, staging_gbps=8, staging_fixed_us=10,
             activation_bytes=0, return_bytes=4, remote_compute_factor=1,
             copy_penalty=1, compute_penalty=1):
    """FCFS per request with copy lookahead bounded by slots, independent of input readiness.

    Resources: one compute/H2D/D2H engine per node, one shared half-duplex wire.
    Activations explicitly compete with weights for destination H2D. Future jobs cannot
    backfill already reserved intervals: this is a stated scheduler, not an optimal oracle.
    """
    free={f'{kind}{node}':0. for kind in ['compute','h2d','d2h'] for node in [0,1]}
    free['network']=0.
    slots=[[0.]*buffers for _ in range(2)]
    turns=[0,0]
    events=[]
    ends=[]
    first=[]
    copy_wait=0.
    communication=0.
    def reserve(resource,ready,duration,kind,req,group=-1,slot=-1):
        start=max(ready,free[resource]); end=start+duration
        free[resource]=end
        events.append(dict(resource=resource,kind=kind,request=req,group=group,slot=slot,start_ms=start,end_ms=end))
        return end
    def send(src,dst,ready,size,req):
        nonlocal communication
        # CPU staging endpoint model, not GPU Direct RDMA.
        dt=staging_fixed_us/1000+size/(staging_gbps*1e6)
        a=reserve(f'd2h{src}',ready,dt,'activation_d2h',req)
        b=reserve('network',a,latency_us/1000+size*8/(bandwidth_gbps*1e6),'wire',req)
        c=reserve(f'h2d{dst}',b,dt,'activation_h2d',req)
        communication+=c-ready
        return c
    for req in range(requests):
        arrival=req*arrival_gap_ms
        ready=arrival
        prev_owner=0
        start_compute=None
        for cycle in range(cycles):
            for j,((copy_ms,compute_ms,_),owner) in enumerate(zip(groups,owners)):
                slot=turns[owner]%buffers
                turns[owner]+=1
                # The CPU retains all weights, but GPU slots evict after each scheduled use.
                loaded=reserve(f'h2d{owner}',max(arrival,slots[owner][slot]),copy_ms*copy_penalty,
                               'weight_h2d',req,j,slot)
                if prev_owner!=owner:
                    ready=send(prev_owner,owner,ready,activation_bytes,req)
                available=max(ready,free[f'compute{owner}'])
                copy_wait+=max(0.,loaded-available)
                begin=max(available,loaded)
                if start_compute is None:
                    start_compute=begin
                duration=compute_ms*compute_penalty*(remote_compute_factor if owner else 1)
                ready=reserve(f'compute{owner}',begin,duration,'block_group',req,j,slot)
                slots[owner][slot]=ready
                prev_owner=owner
            if prev_owner!=0:
                ready=send(prev_owner,0,ready,return_bytes,req)
                prev_owner=0
        first.append(start_compute-arrival)
        ends.append(ready)
    latency=[end-i*arrival_gap_ms for i,end in enumerate(ends)]
    return {'makespan_ms':max(ends),'request_latency_ms':latency,
            'queue_before_first_compute_ms':first,'exposed_weight_wait_sum_ms':copy_wait,
            'communication_path_sum_ms':communication,'events':events,
            'wire_messages':sum(e['resource']=='network' for e in events)}


def run(cfg,out,e1,e2):
    mc=get_opt_config(cfg['model']); ec=cfg['e3']
    lookup={s['group']:s for s in e1 if s.get('status')=='ok'}
    contention={(s['group'],s['q'],s['past'],s['batch']):s for s in e2 if s.get('status')=='ok'}
    rows=[]; example=None
    for g,s in lookup.items():
        sizes=[min(g,mc.num_hidden_layers-i) for i in range(0,mc.num_hidden_layers,g)]
        if len(sizes)<2 or any(n not in lookup for n in sizes):
            continue  # Never extrapolate an unmeasured remainder group silently.
        for comp in s['compute']:
            q,p,b=comp['q'],comp['past'],comp['batch']
            groups=[]
            for n in sizes:
                data=lookup[n]
                c=next(c for c in data['compute'] if (c['q'],c['past'],c['batch'])==(q,p,b))
                groups.append((data['transfers']['actual_tensors_pinned']['cuda_ms']['p50'],
                               c['cuda_ms']['p50'],data['weight_bytes']))
            cm=contention[(g,q,p,b)]['metrics']
            for bw,lat,remote,buffers,nreq,gap,penalty in itertools.product(
                ec['bandwidth_gbps'],ec['one_way_latency_us'],ec['remote_compute_factors'],
                ec['buffers'],ec['requests'],ec['arrival_gap_ms'],['isolated','sustained_contention']):
                cp=max(1.,cm['copy_slowdown']) if penalty=='sustained_contention' else 1.
                kp=max(1.,cm['compute_slowdown']) if penalty=='sustained_contention' else 1.
                # Exact KV budget for layer count; workspace is an explicit coarse safety reserve.
                max_weight=max(t[2] for t in groups)
                kv_bytes=4*nreq*b*(p+q)*mc.input_dim*mc.num_hidden_layers
                gpu_est=buffers*max_weight+kv_bytes+512*1024**2
                if gpu_est>cfg['gpu_budget_gib']*GIB:
                    rows.append(dict(kind='prediction',status='budget_excluded',group=g,q=q,past=p,batch=b,
                                     buffers=buffers,requests=nreq,estimated_gpu_bytes=gpu_est))
                    continue
                kwargs=dict(requests=nreq,arrival_gap_ms=gap,cycles=ec['cycles'] if q==1 else 1,
                            buffers=buffers,bandwidth_gbps=bw,latency_us=lat,
                            staging_gbps=ec['endpoint_staging_gbps'][0],staging_fixed_us=ec['endpoint_fixed_us'],
                            activation_bytes=b*q*mc.input_dim*2,return_bytes=b*4,
                            remote_compute_factor=remote,copy_penalty=cp,compute_penalty=kp)
                # Both use exactly these devices, precisions, budgets and workload. Give the
                # contiguous baseline all group-boundary cuts to account for heterogeneity.
                candidates=[]
                for cut in range(1,len(groups)):
                    plan=[0]*cut+[1]*(len(groups)-cut)
                    result=schedule(groups,plan,**kwargs)
                    candidates.append((result['makespan_ms'],cut,result))
                _,cut,base=min(candidates,key=lambda c:c[0])
                relay=schedule(groups,[i%2 for i in range(len(groups))],**kwargs)
                row=dict(kind='prediction',status='ok',group=g,q=q,past=p,batch=b,
                         semantic=comp['semantic'],bandwidth_gbps=bw,one_way_latency_us=lat,
                         remote_compute_factor=remote,buffers=buffers,requests=nreq,arrival_gap_ms=gap,
                         penalty_scenario=penalty,contiguous_best_cut=cut,
                         contiguous_ms=base['makespan_ms'],relay_ms=relay['makespan_ms'],
                         predicted_speedup=base['makespan_ms']/relay['makespan_ms'],
                         contiguous_weight_wait_ms=base['exposed_weight_wait_sum_ms'],
                         relay_weight_wait_ms=relay['exposed_weight_wait_sum_ms'],
                         contiguous_wire_messages=base['wire_messages'],relay_wire_messages=relay['wire_messages'],
                         relay_request_p95_ms=stats(relay['request_latency_ms'])['p95'],
                         relay_queue_p95_ms=stats(relay['queue_before_first_compute_ms'])['p95'],
                         estimated_gpu_bytes=gpu_est,activation_bytes=kwargs['activation_bytes'],
                         omitted_embedding_and_lm_head=True)
                rows.append(row)
                if example is None and g==2 and q==128 and b==1 and nreq==1 and gap==0 and buffers==2 and bw==2.5 and lat==100 and remote==1 and penalty=='isolated':
                    example={'parameters':kwargs,'measurement_shape':{'group':g,'q':q,'past':p,'batch':b},
                             'contiguous':base,'relay':relay}
    write_csv(out/'predictions.csv',rows)
    write_json(out/'example_timeline.json',example)
    valid=[r for r in rows if r['status']=='ok']
    summary={'kind':'prediction_not_two_machine_measurement','scenarios':len(valid),
             'predicted_faster':sum(r['predicted_speedup']>1.01 for r in valid),
             'predicted_slower':sum(r['predicted_speedup']<.99 for r in valid),
             'speedup_range':[min(r['predicted_speedup'] for r in valid),max(r['predicted_speedup'] for r in valid)],
             'best':max(valid,key=lambda r:r['predicted_speedup']),
             'worst':min(valid,key=lambda r:r['predicted_speedup']),
             'limitations':[
                 'Block-stack model, excludes input/output embedding and LM head compute/transfer; not total serving latency.',
                 'Remote H2D equal to local; remote compute scaled synthetically. Wire/staging parameters are assumptions.',
                 'Every group uses measurements of the first n actual blocks; layer-to-layer variation is not calibrated.',
                 'Prefill q>1 has past=0; these are short-prompt proxies, not chunked prefill.',
                 'Two decode cycles hold KV length fixed; no real growing-cache workload.',
                 'FCFS list schedule, no backfilling; queue results are synthetic and not online SLO goodput.',
                 'No multi-request weight reuse. This can materially overstate relay advantage versus a batched baseline.',
                 'One half-duplex wire. GPU staging competes with weight H2D; host DRAM/NIC/compute contention only bounded by constant penalty scenarios.',
                 'Memory estimate reserves full-stack KV on each node conservatively; 512MiB workspace reserve is not a proof of fitting a larger model.',
                 'No retained GPU weight optimization even when model fits; these are forced-offload conditions.'
             ]}
    write_json(out/'summary.json',summary)
    return summary
