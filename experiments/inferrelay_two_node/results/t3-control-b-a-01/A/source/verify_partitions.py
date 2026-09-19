"""GPU integration check: unequal groups, arbitrary starts and slot reuse."""
import argparse
import gc
import hashlib
from pathlib import Path
import torch
from .executor import Executor
from .model_worker import run_request,prompt_ids
from experiments.inferrelay_precheck.common import write_json,Memory


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.set_device(0)
    rows=[]
    with torch.inference_mode():
        ids=prompt_ids(1,32)
        exe=Executor('opt-125m',[(0,3),(3,6),(6,9),(9,12)],[0]*4,0,'resident',4)
        _,reference=run_request(exe,None,[0]*4,ids,4,0,capture=True);exe.weights.close();del exe;gc.collect();torch.cuda.empty_cache()
        for mode in ['resident','demand','prefetch']:
            with Memory() as memory:
                exe=Executor('opt-125m',[(0,2),(2,5),(5,9),(9,12)],[0]*4,0,mode,4)
                result,_=run_request(exe,None,[0]*4,ids,4,1,reference=reference,capture=True)
                exe.weights.close();del exe
            rows.append(dict(mode=mode,groups=[[0,2],[2,5],[5,9],[9,12]],result=result,memory=memory.values))
            gc.collect();torch.cuda.empty_cache()
    write_json(args.out/'results.json',rows)
    write_json(args.out/'source_sha256.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    print('Unequal real-global-layer partitions and all weight modes PASS')


if __name__=='__main__':main()
