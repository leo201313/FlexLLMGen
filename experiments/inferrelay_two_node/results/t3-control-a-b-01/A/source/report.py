"""Summarize ACK-completion transport measurements; never align host clocks."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np


def quantile(xs, q=50):
    return float(np.percentile(xs,q)) if xs else None


def overlap(interval,jobs):
    a,b=interval
    if a is None: return 0.
    return sum(max(0.,min(b,j['gpu_end_ms'])-max(a,j['gpu_start_ms'])) for j in jobs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    summaries=[];flat=[];timelines=[]
    for run in args.runs:
        launch=json.loads((run/'launch.json').read_text())
        assert json.loads((run/'launcher_completion.json').read_text())['success']
        src=next(n for n in launch if launch[n]['role']=='send');dst='B' if src=='A' else 'A'
        send=json.loads((run/src/'raw.json').read_text());receive=json.loads((run/dst/'raw.json').read_text())
        for case,rcase in zip(send,receive):
            assert case['case']==rcase['case'] and case['shape']==rcase['shape']
            peer={r['sample']:r for r in rcase['rows']}
            steady=[r for r in case['rows'] if r['phase']=='steady']
            assert all(r['correct'] for r in case['rows']+rcase['rows'])
            base=dict(run=str(run),direction=src+'-'+dst,case=case['case'],model=case['model'],batch=case['shape'][0],q=case['shape'][1],hidden=case['shape'][2],mode=case['mode'],bytes=case['bytes'])
            summary=dict(base,n=len(steady),correct=True,sender_setup_ms=case['setup_ms'],receiver_setup_ms=rcase['setup_ms'],
                first_transfer_e2e_ms=case['rows'][0]['e2e_ack_ms'],
                sender_background_alone_cuda_p50_ms=quantile(case['background_alone_cuda_ms']),
                receiver_background_alone_cuda_p50_ms=quantile(rcase['background_alone_cuda_ms']))
            for key in ['e2e_ack_ms','d2h_cuda_ms','d2h_host_ms','h2d_cuda_ms','h2d_host_ms','network_host_ack_residual_ms']:
                for pct in [50,95]: summary[key+'_p'+str(pct)]=quantile([r[key] for r in steady],pct)
            sender_jobs=[j['cuda_ms'] for r in steady for j in r['background_sender_jobs']]
            receiver_jobs=[j['cuda_ms'] for r in steady for j in peer[r['sample']]['background_receiver_jobs']]
            summary['sender_background_jobs']=len(sender_jobs);summary['receiver_background_jobs']=len(receiver_jobs)
            for end,jobs in [('sender',sender_jobs),('receiver',receiver_jobs)]:
                alone=summary[end+'_background_alone_cuda_p50_ms']
                summary[end+'_background_slowdown']=quantile(jobs)/alone if jobs and alone else None
            summary['samples_with_d2h_gpu_overlap']=sum(overlap(r['d2h_gpu_interval'],r['background_sender_jobs'])>0 for r in steady)
            summary['samples_with_h2d_gpu_overlap']=sum(overlap(r['h2d_gpu_interval'],peer[r['sample']]['background_receiver_jobs'])>0 for r in steady)
            summaries.append(summary)
            for r in case['rows']:
                flat.append({**base,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))}})
            if case['mode'] in ['gpu_h2d','gpu_compute'] and (case['shape'][0],case['shape'][1]) in [(1,1),(4,256)]:
                r=steady[0];pr=peer[r['sample']]
                timelines.append(dict(base,sample=r['sample'],nodes={src:{'stage':'D2H','interval':r['d2h_gpu_interval'],'jobs':r['background_sender_jobs']},dst:{'stage':'H2D','interval':pr['h2d_gpu_interval'],'jobs':pr['background_receiver_jobs']}}))
    for s in summaries:
        baseline=next(r for r in summaries if all(r[k]==s[k] for k in ['run','batch','q']) and r['mode']=='gpu')
        s['e2e_slowdown_vs_gpu']=s['e2e_ack_ms_p50']/baseline['e2e_ack_ms_p50'] if s['mode'].startswith('gpu') else None
        for stage in ['d2h','h2d']:
            key=stage+'_cuda_ms_p50';s[stage+'_slowdown_vs_gpu']=s[key]/baseline[key] if baseline[key] and s['mode'].startswith('gpu') else None
    for name,data in [('summary',summaries),('raw',flat)]:
        (args.out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        keys=list(dict.fromkeys(k for row in data for k in row))
        with (args.out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    (args.out/'timelines.json').write_text(json.dumps(timelines,indent=2)+'\n')
    content=['<!doctype html><meta charset="utf-8"><title>T1 local GPU timelines</title><style>body{font:15px sans-serif;max-width:1100px;margin:30px auto}svg{background:#f3f5f7}rect.stage{fill:#db761d}rect.job{fill:#287ba3}</style><h1>T1 分节点 GPU 时间线</h1><p>每个节点独立 CUDA event 原点，不能跨节点对齐。橙色为 activation 拷贝，蓝色为背景工作；背景区间包含该工作内核/拷贝间的设备间隙。</p>']
    for t in timelines:
        content.append('<h2>'+html.escape(f"{t['model']} {t['direction']} b{t['batch']} q{t['q']} {t['mode']}")+'</h2>')
        for node,d in t['nodes'].items():
            maximum=max([d['interval'][1]]+[j['gpu_end_ms'] for j in d['jobs']]+[.001]);scale=900/maximum
            content.append(f'<p>{node}: 0—{maximum:.3f} ms（本地GPU）</p><svg width="1000" height="70">')
            for interval,cls,y in [(d['interval'],'stage',5)]+[([j['gpu_start_ms'],j['gpu_end_ms']],'job',35) for j in d['jobs']]:
                a,b=interval;content.append(f'<rect class="{cls}" x="{a*scale:.2f}" y="{y}" width="{max((b-a)*scale,.2):.2f}" height="20"><title>{a:.6f}—{b:.6f} ms</title></rect>')
            content.append('</svg>')
    (args.out/'timeline.html').write_text('\n'.join(content))
    lines=['# T1：真实 activation 通路','',f'{len(summaries)} 个配置，{sum(s["n"] for s in summaries)} 次稳态传输，全部逐字节/逐元素校验通过。首次传输和 warmup 原始数据保留。','',
        '|模型|方向|batch|q|模式|字节|ACK完成 P50/P95 ms|D2H/H2D CUDA P50 ms|相对 GPU 独占|', '|---|---|---:|---:|---|---:|---|---|---|']
    for s in summaries:
        ratio=s['e2e_slowdown_vs_gpu'];ratio='—' if ratio is None else f'{ratio:.2f}×'
        lines.append(f"|{s['model']}|{s['direction']}|{s['batch']}|{s['q']}|{s['mode']}|{s['bytes']}|{s['e2e_ack_ms_p50']:.3f}/{s['e2e_ack_ms_p95']:.3f}|{s['d2h_cuda_ms_p50']:.3f}/{s['h2d_cuda_ms_p50']:.3f}|{ratio}|")
    lines+=['','## 测量口径与限制','',
        '- 发送端从背景激活/D2H前至对端完成后的ACK计时；包含ACK和主机调度，不是纯单向网络延迟。元数据准备/ready握手在此时间之外，未来端到端原型必须计入。',
        '- GPU buffer和pinned buffer预分配，背景线程每配置预启动；JSON、CUDA events和FlexGen计算临时张量仍可能产生分配。setup包含分配及独占背景基准，不能解释为纯分配耗时。first_transfer是本配置首次传输，不是重启/冷缓存。',
        '- 校验及GPU readback在ACK后、下一轮前。20次稳态重复的P95仅为初期描述，不证明尾延迟稳定；模式固定顺序，受温度/DVFS/CPU调度影响。',
        '- gpu_h2d/gpu_compute同时在两个端点运行对应模型的1个真实block负载。线程与主线程可能有Python/GIL争用；不能把所有slowdown归给GPU。background_slowdown以本配置同节点独占工作为分母。',
        '- 实際CUDA区间相交数见summary.csv；零相交样本不证明GPU并发。工作区间包含内核间隙，区间相交也不等于每条指令同时执行。',
        '- residual=ACK完成墙钟−发送端D2H主机时间−接收端H2D主机时间，包含网络、ACK、其余处理/等待；它不是精确可加的wire时间或跨节点时钟差。',
        '- 内存峰值在各节点completion.json，包含1个常驻背景参考block及诊断对象；不代表最终权重槽位部署预算。',
        '- 分节点时间线见timeline.html，原始事件见timelines.json/各端raw.json，不存在精确全局时钟对齐。',
        '- T1仅验证通信和争用，尚未执行C1/R1模型接力，不能报告接力加速或在线SLO收益。下一步T2必须先完整logits/token/KV校验。']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print('Summarized cases:',len(summaries))


if __name__=='__main__':main()
