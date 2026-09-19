"""T1.5 finite equal-work interference experiment."""
import argparse
import collections
import gc
import hashlib
import json
import random
import socket
import threading
import time
from pathlib import Path
import torch
from .activation import event,ms
from .wire import send_json,recv_json,receive_into,validate
from experiments.inferrelay_precheck.components import Blocks
from experiments.inferrelay_precheck.common import inventory,Memory,write_json


def timed_copy(dst,src,stream,anchor):
    t=ms();a,b=event(),event()
    with torch.cuda.stream(stream):
        stream.wait_event(anchor);a.record()
        submitted=ms();dst.copy_(src,non_blocking=True);called=ms();b.record()
    enqueued=ms();b.synchronize();end=ms()
    return dict(event_prepare_ms=submitted-t,copy_call_ms=called-submitted,post_call_submit_ms=enqueued-called,
        completion_wait_ms=end-enqueued,host_ms=end-t,cuda_ms=a.elapsed_time(b),
        gpu_start_ms=anchor.elapsed_time(a),gpu_end_ms=anchor.elapsed_time(b))


class Prefetch:
    def __init__(self,blocks,policy):
        self.blocks=blocks;self.policy=policy
        self.dst=[torch.empty_like(t,device='cuda') for t in blocks.pinned]
        self.stream=torch.cuda.Stream();self.pause=threading.Event()
        self.go=threading.Event();self.done=threading.Event();self.closed=False;self.error=None
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()

    def start(self,anchor):
        self.anchor=anchor;self.records=[];self.pause.clear();self.done.clear();self.go.set()

    def loop(self):
        torch.cuda.set_device(0)
        while True:
            self.go.wait();self.go.clear()
            if self.closed:return
            try:self.work()
            except BaseException as exc:self.error=exc
            finally:self.done.set()

    def work(self):
        pieces=[]
        for d,s in zip(self.dst,self.blocks.pinned):
            d=d.view(-1);s=s.view(-1)
            size=s.numel() if self.policy=='bulk' else 4*1024**2
            pieces.extend((d[i:i+size],s[i:i+size]) for i in range(0,s.numel(),size))
        queue=collections.deque();cap=2 if self.policy=='window2' else 1
        if self.policy=='bulk':cap=len(pieces)
        def finish():
            a,b,row=queue.popleft();t=ms();b.synchronize();row['completion_wait_host_ms']=ms()-t
            row['gpu_start_ms']=self.anchor.elapsed_time(a);row['gpu_end_ms']=self.anchor.elapsed_time(b)
            row['cuda_ms']=a.elapsed_time(b);self.records.append(row)
        start=ms()
        for cycle in range(2):
            for d,s in pieces:
                while len(queue)>=cap:finish()
                while self.policy=='priority' and self.pause.is_set():time.sleep(.00005)
                a,b=event(),event();t=ms()
                with torch.cuda.stream(self.stream),torch.inference_mode():
                    self.stream.wait_event(self.anchor);a.record();pre=ms();d.copy_(s,non_blocking=True);post=ms();b.record()
                queue.append((a,b,dict(cycle=cycle,bytes=s.numel()*2,submit_host_ms=ms()-t,copy_call_ms=post-pre)))
        while queue:finish()
        self.wall_ms=ms()-start

    def wait(self):
        if not self.done.wait(30):raise TimeoutError('Prefetch deadline')
        if self.error:raise self.error

    def close(self):
        self.closed=True;self.go.set();self.thread.join(5)


def main():
    p=argparse.ArgumentParser()
    for k in ['role','node','model','host','run-id']:p.add_argument('--'+k,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--port',type=int,default=5211);p.add_argument('--quick',action='store_true');p.add_argument('--config')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.set_device(0)
    cfg={'model':'facebook/opt-1.3b','weights_root':'~/opt_weights','node_id':args.node}
    inventory(args.out,cfg);write_json(args.out/'invocation.json',vars(args))
    sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    write_json(args.out/'source_sha256.json',sources)
    sender=args.role=='send';sock=None;listener=None;pf=None;results=[]
    try:
      with torch.inference_mode(),Memory() as memory:
        blocks=Blocks(cfg,1)
        stream=torch.cuda.Stream(priority=-1)
        if not sender:
            listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind((args.host,args.port));listener.listen(1);listener.settimeout(60)
            print('LISTENING',flush=True);sock,_=listener.accept()
        else:sock=socket.create_connection((args.host,args.port),30)
        sock.settimeout(30);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        matrix=[]
        for q in ([1] if args.quick else [1,128]):
            matrix.append((q,'none','bulk','zero'))
            for side in ['sender','receiver','both']:
                for policy in ['bulk','chunk','window2','priority']:
                    for phase in (['zero'] if args.quick else ['zero','delay2','jitter']):matrix.append((q,side,policy,phase))
        for idx,(q,side,policy,phase) in enumerate(matrix):
            active=side=='both' or side==('sender' if sender else 'receiver')
            pf=Prefetch(blocks,policy) if active else None
            host=torch.empty((1,q,2048),dtype=torch.float16,pin_memory=True)
            expected=(torch.arange(host.numel())%503).reshape(host.shape).half();host.copy_(expected)
            device=host.cuda();view=memoryview(host.numpy()).cast('B');digest=hashlib.sha256(view).hexdigest()
            state=blocks.state(q,128 if q==1 else 0,1)
            baseline=[]
            if pf:
                for _ in range(3):
                    anchor=event();anchor.record();pf.start(anchor);pf.wait();baseline.append(pf.wall_ms)
            torch.cuda.synchronize()
            send_json(sock,{'ready':idx});validate(recv_json(sock),{'ready':idx})
            rows=[]
            for sample in range(args.repeats+1):
                desc=dict(run_id=args.run_id,request_id=idx,step_id=sample,group_id=0,shape=[1,q,2048],dtype='float16',bytes=q*4096)
                if sender:send_json(sock,desc)
                else:validate(recv_json(sock),desc)
                anchor=event();anchor.record();work_start=ms()
                if pf:pf.start(anchor)
                if sender:validate(recv_json(sock),{'go':sample})
                else:send_json(sock,{'go':sample})
                delay=0 if phase=='zero' else (2 if phase=='delay2' else random.Random(17000+idx*100+sample).uniform(0,8))
                if sender and delay:time.sleep(delay/1000)
                if sender:
                    origin=ms()
                    if pf and policy=='priority':pf.pause.set()
                    timing=timed_copy(host,device,stream,anchor)
                    if pf:pf.pause.clear()
                    sock.sendall(view);validate(recv_json(sock),{'ack':sample});activation_ms=ms()-origin
                else:
                    before_recv=ms();receive_into(sock,view);origin=ms()
                    if pf and policy=='priority':pf.pause.set()
                    timing=timed_copy(device,host,stream,anchor)
                    if pf:pf.pause.clear()
                    send_json(sock,{'ack':sample});activation_ms=None
                wait=ms()
                if pf:pf.wait()
                weight_wait=ms()-wait
                a,b=event(),event();a.record()
                result=blocks.compute(state,blocks.wrap(pf.dst) if pf else None);b.record();b.synchronize()
                complete=ms()-work_start
                if not sender:
                    assert hashlib.sha256(view).hexdigest()==digest and torch.equal(device.cpu(),expected)
                if pf:
                    for d,s in zip(pf.dst,blocks.pinned):assert torch.equal(d.cpu(),s)
                row=dict(sample=sample,warmup=sample==0,correct=True,requested_delay_ms=delay,
                    local_activation_arrival_ms=origin-work_start,activation_ack_ms=activation_ms,
                    activation_copy=timing,post_activation_weight_wait_ms=weight_wait,
                    subsequent_compute_cuda_ms=a.elapsed_time(b),finite_work_complete_ms=complete,
                    weight_bytes=2*blocks.bytes if active else 0,weight_work_ms=pf.wall_ms if pf else 0,
                    weight_records=pf.records if pf else [])
                # Receiver full records stay local; ACK avoids transferring trace in timed path.
                if sender:
                    peer=recv_json(sock);validate(peer,{'correct':True,'sample':sample});row['peer']=peer
                else:
                    send_json(sock,{k:v for k,v in row.items() if k!='weight_records'})
                rows.append(row)
            results.append(dict(case=idx,q=q,side=side,policy=policy,phase=phase,active=active,
                weight_alone_ms=baseline,rows=rows))
            write_json(args.out/'raw.json',results)
            if pf:pf.close();pf=None
            print('case',idx,q,side,policy,phase,'PASS',flush=True);gc.collect()
      write_json(args.out/'completion.json',{'pass':True,'cases':len(results),'memory':memory.values})
    except BaseException as exc:
        write_json(args.out/'failure.json',{'error':repr(exc)});raise
    finally:
        if pf:pf.pause.clear();pf.close()
        if sock:sock.close()
        if listener:listener.close()


if __name__=='__main__':main()
