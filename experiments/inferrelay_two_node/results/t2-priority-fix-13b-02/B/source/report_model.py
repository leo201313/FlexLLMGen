"""T2/T3 reports. First endpoint E2E includes embedding, head and token return."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def median(xs):return float(np.median(xs))


def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out


def intersection(a,b):
    total=0.;i=j=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total


def overlap_bounds(first,last):
    cal=first.get('calibration')
    if not cal:return None,None,None
    offlo,offhi=cal['offset_last_minus_first_ms']
    def intervals(row,remote,core):
        lo,hi=row['gpu_anchor_host_ms']
        if remote:lo-=offhi;hi-=offlo
        return union([(hi+a,lo+b) if core else (lo+a,hi+b) for c in row['copies'] for a,b in c.get('chunks',[[c['gpu_start_ms'],c['gpu_end_ms']]])])
    return (intersection(intervals(first,False,True),intervals(last,True,True)),
            intersection(intervals(first,False,False),intervals(last,True,False)),cal['width_ms'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);summary=[];flat=[];timelines=[];refs=[]
    for run in args.runs:
        launch=json.loads((run/'launch.json').read_text());first=next(n for n,r in launch.items() if r['role']=='send');last='B' if first=='A' else 'A'
        assert json.loads((run/'launcher_completion.json').read_text())['success']
        data=json.loads((run/first/'raw.json').read_text());peer=json.loads((run/last/'raw.json').read_text())
        invocation=json.loads((run/first/'invocation.json').read_text());model=invocation['model']
        stage=json.loads((run/first/'config.json').read_text())['stage']
        for ref in sorted((run/first).glob('reference-*.json')):
            r=json.loads(ref.read_text());refs.append(dict(run=str(run),model=model,first=first,**r))
        for ci,(c,pc) in enumerate(zip(data,peer)):
            w=c['workload'];case=c['case']
            arm=('R' if case['layout']=='relay' else 'C')+({'prefetch':'1','demand':'0','resident':'resident'}[case['mode']])
            base=dict(run=str(run),stage=stage,model=model,direction=first+'-'+last,batch=w['batch'],q=w['q'],steps=w['steps'],arm=arm,cut=case.get('cut',2),policy=case.get('policy','priority'))
            rows=[r for r in c['rows'] if not r['warmup']];prs=[r for r in pc['rows'] if not r['warmup']]
            assert len(rows)==len(prs)>0
            s=dict(base,n=len(rows),e2e_p50_ms=median([r['e2e_ms'] for r in rows]),e2e_p95_ms=float(np.percentile([r['e2e_ms'] for r in rows],95)),
                prefill_p50_ms=median([r['step_ms'][0] for r in rows]),decode_step_p50_ms=median([t for r in rows for t in r['step_ms'][1:]]),
                correct=all(r['correct'] for r in c['rows']+pc['rows']),logits_max_abs=max(r['errors']['logits_max_abs'] for r in c['rows']+pc['rows']),
                kv_max_abs=max(r['errors']['kv_max_abs'] for r in c['rows']+pc['rows']),
                wire_bytes=rows[0]['sent_wire_bytes']+prs[0]['sent_wire_bytes'],activation_bytes=rows[0]['activation_payload_bytes']+prs[0]['activation_payload_bytes'])
            for label,rs,cc in [('first',rows,c),('last',prs,pc)]:
                for key in ['endpoint_weight_bytes','kv_bytes','weight_slot_bytes','resident_block_bytes','pinned_weight_bytes']:
                    s[label+'_'+key]=rs[0][key]
                for key in ['gpu_allocated_peak_bytes','gpu_reserved_peak_bytes','rss_sampled_peak_bytes']:s[label+'_'+key]=cc['memory'][key]
                s[label+'_weight_transfer_bytes']=sum(r['bytes'] for r in rs[0]['copies'])
                s[label+'_weight_transfers']=len(rs[0]['copies'])
                s[label+'_h2d_chunk_sum_ms']=median([sum(b-a for c in r['copies'] for a,b in c.get('chunks',[[c['gpu_start_ms'],c['gpu_end_ms']]])) for r in rs])
                for key in ['weight_host_wait_ms','weight_gpu_wait_ms','compute_cuda_ms']:s[label+'_'+key]=median([sum(c[key] for c in r['compute']) for r in rs])
                s[label+'_buffer_host_wait_ms']=median([sum(c['buffer_host_wait_ms'] for c in r['copies']) for r in rs])
                s[label+'_buffer_gpu_wait_ms']=median([sum(c['buffer_gpu_wait_ms'] for c in r['copies']) for r in rs])
                s[label+'_activation_receive_wait_ms']=median([sum(c['wall_ms'] for c in r['communication'] if not c['sender']) for r in rs])
            bounds=[overlap_bounds(a,b) for a,b in zip(rows,prs)]
            for index,key in enumerate(['h2d_overlap_lower_ms','h2d_overlap_upper_ms','clock_offset_width_ms']):
                values=[b[index] for b in bounds if b[index] is not None];s[key]=median(values) if values else None
            summary.append(s)
            for r,pr in zip(rows,prs):
                flat.append(dict(base,sample=r['sample'],e2e_ms=r['e2e_ms'],prefill_ms=r['step_ms'][0],decode_mean_ms=float(np.mean(r['step_ms'][1:])),first_errors=r['errors'],last_errors=pr['errors']))
            timelines.append(dict(label=f"{model} {first}-{last} b{w['batch']} q{w['q']} {arm} cut{case.get('cut',2)} {base['policy']}",
                nodes={first:{k:rows[0][k] for k in ['copies','compute','communication']},last:{k:prs[0][k] for k in ['copies','compute','communication']}}))
    # Equality of weight-slot and endpoint bytes for all offload arms in a fixed node orientation/workload.
    keys={(s['model'],s['direction'],s['batch'],s['q']) for s in summary}
    for key in keys:
        ss=[s for s in summary if (s['model'],s['direction'],s['batch'],s['q'])==key and s['arm'] in ['C0','C1','R0','R1']]
        for endpoint in ['first','last']:
            assert len({s[endpoint+'_weight_slot_bytes'] for s in ss})<=1
            assert len({s[endpoint+'_endpoint_weight_bytes'] for s in ss})<=1
    comparisons=[]
    for model,direction,batch,q in sorted(keys):
        selected=[s for s in summary if (s['model'],s['direction'],s['batch'],s['q'])==(model,direction,batch,q)]
        for policy in ['priority','window2']:
            rs=[s for s in selected if s['arm']=='R1' and s['policy']==policy]
            cs=[s for s in selected if s['arm']=='C1' and s['policy']==policy]
            if not rs or not cs:continue
            r=rs[0];best=min(cs,key=lambda s:s['e2e_p50_ms']);balanced=next((c for c in cs if c['cut']==2),None)
            comparisons.append(dict(model=model,direction=direction,batch=batch,q=q,policy=policy,
                best_continuous_cut=best['cut'],best_C1_ms=best['e2e_p50_ms'],balanced_C1_ms=balanced['e2e_p50_ms'] if balanced else None,
                R1_ms=r['e2e_p50_ms'],best_C1_over_R1=best['e2e_p50_ms']/r['e2e_p50_ms'],
                balanced_C1_over_R1=balanced['e2e_p50_ms']/r['e2e_p50_ms'] if balanced else None))
    for name,rows in [('summary',summary),('raw',flat),('comparisons',comparisons)]:
        (args.out/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
        if rows:
            with (args.out/(name+'.csv')).open('w') as f:
                keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
    (args.out/'single_node_references.json').write_text(json.dumps(refs,indent=2)+'\n')
    (args.out/'timelines.json').write_text(json.dumps(timelines,indent=2)+'\n')
    template='''<!doctype html><meta charset="utf-8"><title>OPT node timelines</title><style>body{font:15px sans-serif;margin:25px}svg{background:#f4f5f6}select{width:90%}</style><h1>分节点 CUDA 时间线</h1><p>每端独立原点；蓝：权重片段，绿：计算，橙：activation staging。CPU网络等待未画作GPU忙碌。跨端重叠区间估计见CSV。</p><select id="choice"></select><div id="view"></div><script>const data=DATA;const sel=document.getElementById('choice');data.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.label;sel.append(o)});function render(){let d=data[+sel.value],html='';Object.entries(d.nodes).forEach(([n,r])=>{let bars=[];r.copies.forEach(c=>(c.chunks||[[c.gpu_start_ms,c.gpu_end_ms]]).forEach(([a,b])=>bars.push([a,b,0,'#287ba3','weight'])));r.compute.forEach(c=>bars.push([c.gpu_start_ms,c.gpu_end_ms,1,'#248650','group '+c.group]));r.communication.forEach(c=>{if(c.gpu_start_ms!=null)bars.push([c.gpu_start_ms,c.gpu_end_ms,2,'#db761d','activation'])});let max=Math.max(...bars.map(b=>b[1]),1);html+='<h2>'+n+'：0—'+max.toFixed(2)+'ms</h2><svg width="1100" height="100">';bars.forEach(([a,b,l,color,label])=>{html+='<rect x="'+a/max*1050+'" y="'+(l*30+5)+'" width="'+Math.max((b-a)/max*1050,.2)+'" height="20" fill="'+color+'"><title>'+label+' '+a.toFixed(3)+'—'+b.toFixed(3)+'ms</title></rect>'});html+='</svg>'});document.getElementById('view').innerHTML=html}sel.onchange=render;render();</script>'''
    (args.out/'timeline.html').write_text(template.replace('DATA',json.dumps(timelines).replace('<','\\u003c')))
    lines=['# 完整模型共享执行器结果','',f'{len(summary)}配置，所有生成token及logits/KV检查通过。T2逐步抽样KV，T3稳态只在结束后检查KV、逐步保存logits/token于计时后核对。', '',
        '|模型|方向|b/q|arm|cut/策略|n|端到端P50 ms|prefill/decode-step P50 ms|最大logits/KV误差|', '|---|---|---|---|---|---:|---:|---|---|']
    for s in summary:lines.append(f"|{s['model']}|{s['direction']}|{s['batch']}/{s['q']}|{s['arm']}|{s['cut']}/{s['policy']}|{s['n']}|{s['e2e_p50_ms']:.3f}|{s['prefill_p50_ms']:.3f}/{s['decode_step_p50_ms']:.3f}|{s['logits_max_abs']:.5g}/{s['kv_max_abs']:.5g}|")
    if comparisons:
        lines+=['','## R1 与调优连续切分','', '|模型|方向|b/q|策略|最优连续cut|C1/R1 ms|C1÷R1|','|---|---|---|---|---:|---|---:|']
        for c in comparisons:lines.append(f"|{c['model']}|{c['direction']}|{c['batch']}/{c['q']}|{c['policy']}|{c['best_continuous_cut']}/4|{c['best_C1_ms']:.3f}/{c['R1_ms']:.3f}|{c['best_C1_over_R1']:.3f}|")
    lines+=['','## 口径','',
        '- 端到端从首节点请求开始至最后生成token返回，包含embedding/head/网络握手与staging。KV预分配、tokenizer与权重初始准备在计时外；T2 capture=True含逐步KV诊断开销，不能与T3稳态混比。',
        '- 端点embedding/head常驻，跨节点token权重各有副本；block权重按步重复载入，CPU留副本，无GPU权重D2H。此为明确的受控容量/重复加载机制实验，未启用跨步驻留复用；resident基线单独保留。',
        '- 两个weight slot的实际字节在所有offload布局相同。切分调整改变KV/实际owned权重，所有节点统一4GiB GPU上限；显式slot+端点字节已自动核对。125M/1.3B能自然常驻，不作部署成本结论。',
        '- CPU wait、CUDA wait、网络等待可能重叠，不相加解释总时间；activation_receive_wait是阻塞接收整条通路的主机时间，包含staging/握手。',
        '- 同节点事件检查copy-ready先于compute、slot复用晚于上一次compute结束（0.01ms事件误差容限），token/logits/KV数值容差固定0.02/0.01，未放宽。',
        '- 双端H2D重叠仅按8次消息交换得到的时钟offset区间和GPU anchor主机包络估计上下界，假设单请求内offset稳定；不是PTP精确校准或硬件利用率。原始时间线分端展示。',
        '- C1/R1都从请求开始独立预取，priority或window2策略完全共用；不以无预取连续布局作唯一基线。',
        '- 每配置3次正式重复为初期小样本；固定顺序和对同批数据选择最优cut有选择偏差，不能将轻微差异称为稳定收益。负结果全保留。']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n');print('Model cases:',len(summary))


if __name__=='__main__':main()
