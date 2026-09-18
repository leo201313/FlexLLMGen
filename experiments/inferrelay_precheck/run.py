"""python -m experiments.inferrelay_precheck.run --config ..."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Set before importing transformers/huggingface_hub. No implicit downloads.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

import torch
from .common import GIB, ROOT, inventory, weights_path, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/smoke.json')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--suite',nargs='+',choices=['e0','e1','e2','e3'],default=['e0','e1','e2','e3'])
    parser.add_argument('--repeats',type=int)
    parser.add_argument('--min-seconds',type=float)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if args.repeats is not None: cfg['repeats']=args.repeats
    if args.min_seconds is not None: cfg['min_seconds']=args.min_seconds
    assert cfg['warmup']>=1 and cfg['repeats']>=3 and cfg['min_seconds']>=0
    assert cfg['e0']['gen_len']>=2
    assert 0<cfg['gpu_budget_gib'] and 0<cfg['rss_budget_gib']
    if 'e2' in args.suite and 'e1' not in args.suite:
        parser.error('E2 requires E1 in the same run')
    if 'e3' in args.suite and not {'e1','e2'}.issubset(args.suite):
        parser.error('E3 requires E1/E2 in the same run')
    out=args.out or Path(__file__).parent/'results'/time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    weights_path(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA GPU is required; run outside the sandbox when needed.')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    if free<cfg['gpu_budget_gib']*GIB:
        raise RuntimeError('Available GPU memory below the configured budget; do not disturb other tasks.')
    torch.cuda.set_per_process_memory_fraction(min(cfg['gpu_budget_gib']*GIB/total,.9))
    inventory(out,cfg)
    write_json(out/'config.json',cfg)
    write_json(out/'invocation.json',{'argv':sys.argv,'cwd':str(Path.cwd()),'python':sys.executable})
    # Preserve runnable source used for this result, even before a Git commit exists.
    snapshot=out/'source'
    snapshot.mkdir()
    hashes={}
    for p in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copyfile(p,snapshot/p.name)
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(out/'source_sha256.json',hashes)
    from . import e0, components, e3, report, visualize
    start=time.perf_counter()
    with torch.inference_mode():
        for stage in args.suite:
            dest=out/stage; dest.mkdir()
            (dest/'README.md').write_text({
                'e0': '# E0\n\nMeasured original generation loops. raw.csv/raw.json contain every sample; summary.json gives quantiles. Trace files are separate diagnostic runs. See [experiment README](../../../README.md) for phase and memory definitions.\n',
                'e1': '# E1\n\nMeasured actual-weight/slab H2D and original FlexGen block compute. raw.csv retains every sample. pin_prepare files measure CPU allocation+pin copy separately. q>1,past=0 is a short-prefill proxy, not incremental prefill.\n',
                'e2': '# E2\n\nMeasured serial/independent-stream concurrency and two-buffer lifecycle. raw.csv retains every sample; buffer CSVs report device waits and ordering. Original FlexGen overlap comparison is in ../e0.\n',
                'e3': '# E3\n\nPredictions only. predictions.csv retains wins/losses/ties; example_timeline.json contains resource reservations. summary.json lists unmodeled effects. Not measured two-machine speedup or online SLO goodput.\n'
            }[stage])
            if stage=='e0': e0.run(cfg,dest)
            elif stage=='e1': measurements=components.run_e1(cfg,dest)
            elif stage=='e2': overlaps=components.run_e2(cfg,dest,measurements)
            else: e3.run(cfg,dest,measurements,overlaps)
            print(stage,'complete',flush=True)
    write_json(out/'completion.json',{'elapsed_seconds':time.perf_counter()-start,'stages':args.suite})
    report.generate(out)
    visualize.render(out)
    print('Results:',out.resolve(),flush=True)


if __name__=='__main__':
    main()
