import argparse
import csv
import json
from pathlib import Path
import numpy as np


def med(xs):return float(np.median(xs))


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);summary=[];raw=[]
    for run in args.runs:
        launch=json.loads((run/'launch.json').read_text());src=next(n for n,r in launch.items() if r['role']=='send');dst='B' if src=='A' else 'A'
        data=json.loads((run/src/'raw.json').read_text());peerdata=json.loads((run/dst/'raw.json').read_text())
        for c,pc in zip(data,peerdata):
            base=dict(direction=src+'-'+dst,q=c['q'],side=c['side'],policy=c['policy'],phase=c['phase'])
            rows=c['rows'][1:];pr=pc['rows'][1:]
            assert all(r['correct'] for r in rows+pr)
            result={**base,'n':len(rows),'activation_p50_ms':med([r['activation_ack_ms'] for r in rows]),'activation_p95_ms':float(np.percentile([r['activation_ack_ms'] for r in rows],95))}
            for end,rs in [('sender',rows),('receiver',pr)]:
                for key in ['finite_work_complete_ms','post_activation_weight_wait_ms','weight_work_ms','subsequent_compute_cuda_ms','local_activation_arrival_ms']:
                    result[end+'_'+key]=med([r[key] for r in rs])
                for key in ['event_prepare_ms','copy_call_ms','post_call_submit_ms','completion_wait_ms','cuda_ms']:
                    result[end+'_activation_'+key]=med([r['activation_copy'][key] for r in rs])
                result[end+'_weight_bytes']=rs[0]['weight_bytes']
                result[end+'_weight_GBps']=med([r['weight_bytes']/r['weight_work_ms']/1e6 for r in rs]) if rs[0]['weight_bytes'] else None
                for r in rs:
                    raw.append({**base,'node':src if end=='sender' else dst,'end':end,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},**{'activation_'+k:v for k,v in r['activation_copy'].items()}})
            summary.append(result)
    for s in summary:
        b=next(x for x in summary if all(x[k]==s[k] for k in ['direction','q','side','phase']) and x['policy']=='bulk')
        s['activation_ratio_vs_bulk']=s['activation_p50_ms']/b['activation_p50_ms']
        for end in ['sender','receiver']:
            s[end+'_finite_work_ratio_vs_bulk']=s[end+'_finite_work_complete_ms']/b[end+'_finite_work_complete_ms']
            bw=s[end+'_weight_GBps'];base=b[end+'_weight_GBps'];s[end+'_weight_bandwidth_ratio_vs_bulk']=bw/base if bw and base else None
    for name,rows in [('summary',summary),('raw',raw)]:
        (args.out/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
        with (args.out/(name+'.csv')).open('w') as f:
            keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# T1.5 干扰诊断','',f'{len(summary)}配置，每配置10个稳态样本。所有payload和最终权重校验通过。','',
        '|方向|q|背景|策略|到达相位|activation P50/P95 ms|发送/接收端有限工作完成 ms|发送/接收端权重GB/s|', '|---|---:|---|---|---|---|---|---|']
    for s in summary:
        bw=lambda n:'—' if s[n+'_weight_GBps'] is None else f"{s[n+'_weight_GBps']:.2f}"
        lines.append(f"|{s['direction']}|{s['q']}|{s['side']}|{s['policy']}|{s['phase']}|{s['activation_p50_ms']:.3f}/{s['activation_p95_ms']:.3f}|{s['sender_finite_work_complete_ms']:.3f}/{s['receiver_finite_work_complete_ms']:.3f}|{bw('sender')}/{bw('receiver')}|")
    lines+=['','## 解释与边界','',
        '- 每活跃端固定拷贝同一个真实完整block两遍，之后用该GPU权重执行一次真实block；没有减少字节数或取消全部预取。finite_work_complete是各端本地时间，不能直接取max伪装成精确跨机全局makespan。',
        '- bulk可一次排入较大工作量；chunk按最多8MiB片段限制一个在途，window2最多两个，priority在activation准备提交时暂停新的片段提交，已经排入的DMA不会被软件撤回。activation流在所有策略中同样设priority=-1。',
        '- copy_call_ms是Python copy_调用耗时；completion_wait_ms是提交事件后等待完成的主机时间。它包括设备排队和可能的运行时/线程调度，不将其全部定性为GPU实际复制服务时间。',
        '- 到达相位0/2ms/固定种子均匀0–8ms；实际到达和每片段CUDA区间在各端raw.json，不能假设线程启动就是DMA启动。不同策略随机样本不是严格逐样本配对，样本量10，只做机制诊断。',
        '- 此处是固定两遍权重的有限工作，与旧T1持续背景负载不同，不要求复现旧4.405/8.107ms的精确数值。',
        '- 接收端背景能复现主要退化，发送端单独背景明显较小。细粒度限制降低activation等待，但可能降低权重GB/s并把等待转移到后续计算前；完整净收益必须结合T3。',
        '- 负结果与所有相位都保留。无CUPTI/驱动级跟踪，尚不能精确拆解驱动调度与物理copy engine内部排队。']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print('T1.5 configurations:',len(summary))


if __name__=='__main__':main()
