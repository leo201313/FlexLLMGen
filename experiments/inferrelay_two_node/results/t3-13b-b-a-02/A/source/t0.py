"""Normalize measured A/B E1/E2 without extrapolating missing shapes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1] / 'inferrelay_precheck/results'
    rows, sources = [], {}
    for node, run in [('A', 'smoke-02'), ('B', 'node-b-smoke-01')]:
        folder = root / run
        names = ['config.json', 'environment.json', 'source_sha256.json', 'e1/summary.json', 'e2/summary.json']
        sources[node] = {name: {'path': str(folder / name), 'sha256': hashlib.sha256((folder / name).read_bytes()).hexdigest()} for name in names}
        cfg = json.loads((folder / 'config.json').read_text())
        e1 = json.loads((folder / 'e1/summary.json').read_text())
        e2 = json.loads((folder / 'e2/summary.json').read_text())
        for group in e1:
            assert group['status'] == 'ok'
            h = group['transfers']['actual_tensors_pinned']
            for c in group['compute']:
                match = next(v for v in e2 if (v['group'],v['q'],v['past'],v['batch']) == (group['group'],c['q'],c['past'],c['batch']))
                assert match['status'] == 'ok'
                m = match['metrics']
                rows.append(dict(node=node, model=cfg['model'], dtype='float16', blocks=group['group'], first_layer=0,
                    batch=c['batch'], query_len=c['q'], past_len=c['past'], semantic=c['semantic'],
                    weight_bytes=group['weight_bytes'], pinned_weight_live_bytes=group['pinned_weight_live_bytes'],
                    two_weight_slots_bytes=match['two_slots_bytes'],
                    h2d_cuda_p50_ms=h['cuda_ms']['p50'], h2d_wall_p50_ms=h['wall_ms']['p50'],
                    compute_cuda_p50_ms=c['cuda_ms']['p50'], compute_wall_p50_ms=c['wall_ms']['p50'],
                    h2d_GBps=h['effective_GBps'], copy_slowdown=m['copy_slowdown'], compute_slowdown=m['compute_slowdown'],
                    overlap_speedup=m['concurrent_speedup_over_serial'],
                    gpu_allocated_peak_bytes=match['memory']['gpu_allocated_peak_bytes'],
                    gpu_reserved_peak_bytes=match['memory']['gpu_reserved_peak_bytes'],
                    rss_sampled_peak_bytes=match['memory']['rss_sampled_peak_bytes']))
    keys = lambda r: (r['model'],r['dtype'],r['blocks'],r['batch'],r['query_len'],r['past_len'],r['weight_bytes'])
    assert {keys(r) for r in rows if r['node']=='A'} == {keys(r) for r in rows if r['node']=='B'}
    with (args.out/'components.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'components.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    (args.out/'README.md').write_text('''# T0：已有组件测量对齐

两端各56个可比形状，OPT-125M、FP16；group 是从 layer0 开始的1/2/4/8个完整 Transformer block。batch1为主要分析，batch4保留。所有 q/past/权重字节数逐项匹配，不用节点倍率。原始来源与SHA256见 sources.json。

CUDA p50 是同机events；wall p50 包含主机投递与完成等待。H2D 为实际16个张量/block的pinned拷贝，非slab。compute调用原版FlexGen kernels。q>1,past0为普通prefill代理；decode过去KV来自组内真实prefill，不是全模型KV正确性证明。

E2 slowdown 分母来自较早的E1独占测量，存在DVFS/温度漂移；内存包含常驻参考权重和诊断目标，不能当最终两槽位部署占用。pinned记录显式活跃权重字节，不引用已发现不可靠的allocator allocated counters。RSS为10ms采样峰值。

缺失：125M group3、1.3B group6，以及不同起始layer分组。现有数据足以做T1独立通信测量，进入四组原型前需针对真实分组补测；不做线性外推。本文件不提供两机接力性能结论。
''')
    print('T0 matched rows:',len(rows))


if __name__ == '__main__':
    main()
