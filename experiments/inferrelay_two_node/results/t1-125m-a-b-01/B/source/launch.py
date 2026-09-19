"""Run bounded A/B T1 workers. Ctrl-C stops only processes owned by this run."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/leocao/miniconda3/envs/llmexp/bin/python'


def remote_stop(run_id):
    script="""import os,signal,json
from pathlib import Path
p=Path('/tmp')/('inferrelay-'+%r+'.pid')
if p.exists():
 pid=int(p.read_text())
 c=Path('/proc')/str(pid)/'cmdline'
 if c.exists() and %r.encode() in c.read_bytes():
  os.killpg(pid,signal.SIGTERM)
""" % (run_id,run_id)
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','inferrelay-b','python3 -c '+shlex.quote(script)],timeout=15,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['opt-125m','opt-1.3b'],default='opt-125m')
    p.add_argument('--direction',choices=['A-B','B-A'],default='A-B')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    run_id=out.name
    if not all(c.isalnum() or c in '-_' for c in run_id): p.error('Output basename must be alphanumeric, dash or underscore')
    receiver='B' if args.direction=='A-B' else 'A'
    host='192.168.20.105' if receiver=='B' else '192.168.20.103'
    records={}
    processes=[]
    def start(node):
        role='receive' if node==receiver else 'send'
        argv=[PYTHON,'-u','-m','experiments.inferrelay_two_node.activation','--node',node,'--role',role,
            '--model',args.model,'--host',host,'--out',str(out/node),'--run-id',run_id,'--repeats',str(args.repeats)]
        if args.quick: argv.append('--quick')
        bounded=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(args.timeout)]+argv
        if node=='B':
            bootstrap="import os; from pathlib import Path; os.setsid() if os.getsid(0)!=os.getpid() else None; Path(%r).write_text(str(os.getpid())); os.chdir(%r); os.execv(%r,%r)" % ('/tmp/inferrelay-'+run_id+'.pid',str(ROOT),bounded[0],bounded)
            cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','inferrelay-b','python3 -c '+shlex.quote(bootstrap)]
        else: cmd=bounded
        log=(out/(node+'.log')).open('w')
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        processes.append((node,proc,log))
        records[node]={'argv':argv,'launcher_pid':proc.pid,'role':role}
        (out/'launch.json').write_text(json.dumps(records,indent=2)+'\n')
        return proc
    def interrupted(signum,frame): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    success=False
    try:
        server=start(receiver)
        deadline=time.monotonic()+60
        while 'LISTENING' not in (out/(receiver+'.log')).read_text():
            if server.poll() is not None: raise RuntimeError('Receiver exited before ready')
            if time.monotonic()>deadline: raise TimeoutError('Receiver startup timed out')
            time.sleep(.1)
        client=start('A' if receiver=='B' else 'B')
        deadline=time.monotonic()+args.timeout+10
        while any(proc.poll() is None for _,proc,_ in processes):
            for node,proc,_ in processes:
                if proc.poll() not in (None,0): raise RuntimeError(f'{node} failed: {proc.returncode}')
            if time.monotonic()>deadline: raise TimeoutError('Run deadline')
            time.sleep(.2)
        success=True
    finally:
        if not success:
            for node,proc,_ in processes:
                if node=='A' and proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM)
            remote_stop(run_id)
        for _,proc,log in processes:
            try: proc.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        # Fetch partial results on failure as well. rsync never deletes remote data.
        fetched=subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',f'inferrelay-b:{out}/B',str(out)+'/'],timeout=60)
        (out/'launcher_completion.json').write_text(json.dumps({'success':success,'fetch_exit_code':fetched.returncode,'exit_codes':{n:p.returncode for n,p,_ in processes}},indent=2)+'\n')
    if not success: raise RuntimeError('Run failed; inspect per-node logs')
    if fetched.returncode: raise RuntimeError('Result fetch failed')
    print('PASS',out,flush=True)


if __name__=='__main__': main()
