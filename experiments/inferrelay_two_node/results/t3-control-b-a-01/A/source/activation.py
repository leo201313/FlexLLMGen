"""T1 FP16 activation transport. CPU buffers are preallocated pinned tensors."""
import argparse
import gc
import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from experiments.inferrelay_precheck.components import Blocks, copy_into
from experiments.inferrelay_precheck.common import Memory, inventory, write_json
from .wire import send_json, recv_json, receive_into, validate


def ms():
    return time.perf_counter() * 1000


def event():
    return torch.cuda.Event(enable_timing=True)


class Background:
    """A bounded one-job-at-a-time stream. Host scheduling cost is part of contention."""
    def __init__(self, blocks, state, mode):
        self.blocks, self.state, self.mode = blocks, state, mode
        self.stream = torch.cuda.Stream()
        self.dst = [torch.empty_like(t, device='cuda') for t in blocks.pinned] if mode == 'h2d' else []
        self.result = None
        self.jobs = []
        self.error = None
        self.anchor = None

    def job(self):
        a,b = event(),event()
        t=ms()
        with torch.cuda.stream(self.stream), torch.inference_mode():
            if self.anchor is not None: self.stream.wait_event(self.anchor)
            a.record()
            if self.mode == 'h2d':
                copy_into(self.dst,self.blocks.pinned)
            else:
                self.result=self.blocks.compute(self.state)
            b.record()
        b.synchronize()
        return {'cuda_ms':a.elapsed_time(b),'host_start_ms':t,'host_end_ms':ms(),
            'gpu_start_ms':self.anchor.elapsed_time(a) if self.anchor is not None else None,
            'gpu_end_ms':self.anchor.elapsed_time(b) if self.anchor is not None else None}

    def baseline(self):
        for _ in range(3): self.job()
        return [self.job()['cuda_ms'] for _ in range(20)]

    def prepare_worker(self):
        self.go=threading.Event()
        self.finished=threading.Event()
        self.stop_event=threading.Event()
        self.shutdown=threading.Event()
        def loop():
            torch.cuda.set_device(0)
            while True:
                self.go.wait();self.go.clear()
                if self.shutdown.is_set(): return
                try:
                    while not self.stop_event.is_set() and len(self.jobs)<10000:
                        row=self.job()
                        row['host_start_ms']-=self.origin;row['host_end_ms']-=self.origin
                        self.jobs.append(row)
                except BaseException as exc:
                    self.error=exc
                finally:
                    self.finished.set()
        self.thread=threading.Thread(target=loop,daemon=True)
        self.thread.start()

    def start(self, origin, anchor):
        self.anchor=anchor;self.origin=origin;self.jobs=[]
        self.finished.clear();self.stop_event.clear();self.go.set()

    def stop(self):
        self.stop_event.set()
        if not self.finished.wait(10): raise TimeoutError('Background did not finish')
        if self.error: raise self.error
        if self.mode=='h2d':
            for a,b in zip(self.dst,self.blocks.pinned):
                if not torch.equal(a.cpu(),b): raise AssertionError('Background weight copy corrupted')
        return self.jobs

    def close(self):
        self.shutdown.set();self.go.set();self.thread.join(10)
        if self.thread.is_alive(): raise TimeoutError('Background worker did not exit')


def cases(model, quick):
    h={'opt-125m':768,'opt-1.3b':2048}[model]
    out=[]
    for batch in ([1] if quick else [1,4]):
        for q in ([1,32] if quick else [1,32,128,256]):
            for mode in ['cpu','gpu','gpu_h2d','gpu_compute']:
                out.append({'model':model,'shape':[batch,q,h],'dtype':'float16','bytes':batch*q*h*2,'mode':mode})
    return out


def phase_copy(dst,src,stream,anchor):
    a,b=event(),event()
    t=ms()
    with torch.cuda.stream(stream):
        stream.wait_event(anchor)
        a.record();dst.copy_(src,non_blocking=True);b.record()
    b.synchronize()
    return {'cuda_ms':a.elapsed_time(b),'host_ms':ms()-t,
        'gpu_start_ms':anchor.elapsed_time(a),'gpu_end_ms':anchor.elapsed_time(b)}


def run_case(sock, cfg, idx, args, blocks):
    sender=args.role=='send'
    shape=cfg['shape'];mode=cfg['mode'];gpu=mode!='cpu'
    setup=ms()
    host=torch.empty(shape,dtype=torch.float16,pin_memory=True)
    # Reproducible exact FP16 payload, nonconstant across positions and cases.
    expected=((torch.arange(host.numel(),dtype=torch.int32)%1021)-510).reshape(shape).to(torch.float16)
    expected.add_(idx % 11)
    host.copy_(expected)
    device=host.to('cuda') if gpu else None
    stream=torch.cuda.Stream()
    view=memoryview(host.numpy()).cast('B')
    expected_hash=hashlib.sha256(view).hexdigest()
    background=None;baseline=[]
    if mode.startswith('gpu_'):
        state=blocks.state(shape[1],128 if shape[1]==1 else 0,shape[0])
        background=Background(blocks,state,mode.split('_')[1])
        baseline=background.baseline()
        background.prepare_worker()
    torch.cuda.synchronize()
    setup_ms=ms()-setup
    send_json(sock,{'kind':'case_ready','case':idx})
    validate(recv_json(sock),{'kind':'case_ready','case':idx})
    rows=[]
    for sample in range(1+args.warmup+args.repeats):
        tag='first_transfer' if sample==0 else ('warmup' if sample<=args.warmup else 'steady')
        desc=dict(cfg,kind='activation',run_id=args.run_id,request_id=idx,step_id=sample,group_id=0)
        bg_started=False
        origin=ms()
        try:
            if sender:
                send_json(sock,desc)
                validate(recv_json(sock),{'kind':'ready','step_id':sample})
                # The control handshake is outside E2E timing. Background signaling/host scheduling is inside; thread was prestarted.
                origin=ms()
                anchor=event();anchor.record()
                if background:
                    background.start(origin,anchor);bg_started=True
                d2h=phase_copy(host,device,stream,anchor) if gpu else {'cuda_ms':0.,'host_ms':0.}
                sock.sendall(view)
                validate(recv_json(sock),{'kind':'ack','step_id':sample})
                elapsed=ms()-origin
                jobs=background.stop() if background else []
                bg_started=False
                peer=recv_json(sock)
                validate(peer,{'kind':'result','step_id':sample,'correct':True})
                residual=elapsed-d2h['host_ms']-peer['h2d_host_ms']
                rows.append(dict(case=idx,**cfg,sample=sample,phase=tag,e2e_ack_ms=elapsed,
                    d2h_cuda_ms=d2h['cuda_ms'],d2h_host_ms=d2h['host_ms'],d2h_gpu_interval=[d2h.get('gpu_start_ms'),d2h.get('gpu_end_ms')],h2d_gpu_interval=peer['h2d_gpu_interval'],
                    h2d_cuda_ms=peer['h2d_cuda_ms'],h2d_host_ms=peer['h2d_host_ms'],
                    network_host_ack_residual_ms=residual,receiver_receive_host_ms=peer['receive_host_ms'],
                    correct=True,background_sender_jobs=jobs,background_receiver_jobs=peer['jobs'],
                    sender_origin='local before background launch and D2H',receiver_origin='local after ready ACK',
                    sender_completion_relative_ms=elapsed,receiver_staging_start_relative_ms=peer['staging_start_ms'],
                    receiver_completion_relative_ms=peer['completion_ms']))
            else:
                validate(recv_json(sock),desc)
                send_json(sock,{'kind':'ready','step_id':sample})
                origin=ms()
                anchor=event();anchor.record()
                if background:
                    background.start(origin,anchor);bg_started=True
                receive_start=ms()
                receive_into(sock,view)
                receive_ms=ms()-receive_start
                staging_start=ms()-origin
                h2d=phase_copy(device,host,stream,anchor) if gpu else {'cuda_ms':0.,'host_ms':0.}
                complete=ms()-origin
                send_json(sock,{'kind':'ack','step_id':sample})
                jobs=background.stop() if background else []
                bg_started=False
                correct=hashlib.sha256(view).hexdigest()==expected_hash
                if gpu: correct=correct and torch.equal(device.cpu(),expected)
                if not correct: raise AssertionError(f'Payload mismatch: case {idx} sample {sample}')
                # Job details bounded to keep control messages below MAX_HEADER.
                # Full local jobs are in receiver raw JSON; wire summary only carries first 32.
                send_json(sock,dict(kind='result',step_id=sample,correct=True,h2d_cuda_ms=h2d['cuda_ms'],
                    h2d_host_ms=h2d['host_ms'],receive_host_ms=receive_ms,jobs=jobs[:32],h2d_gpu_interval=[h2d.get('gpu_start_ms'),h2d.get('gpu_end_ms')],
                    staging_start_ms=staging_start,completion_ms=complete))
                rows.append(dict(case=idx,**cfg,sample=sample,phase=tag,correct=True,
                    h2d_cuda_ms=h2d['cuda_ms'],h2d_host_ms=h2d['host_ms'],receive_host_ms=receive_ms,h2d_gpu_interval=[h2d.get('gpu_start_ms'),h2d.get('gpu_end_ms')],
                    background_receiver_jobs=jobs,receiver_staging_start_relative_ms=staging_start,
                    receiver_completion_relative_ms=complete))
        finally:
            if bg_started: background.stop()
    if background: background.close()
    return {'case':idx,**cfg,'setup_ms':setup_ms,'background_alone_cuda_ms':baseline,
        'communication_pinned_live_bytes':cfg['bytes'],'communication_gpu_live_bytes':cfg['bytes'] if gpu else 0,
        'background_weight_bytes':blocks.bytes if background else 0,
        'background_copy_destination_bytes':blocks.bytes if mode=='gpu_h2d' else 0,
        'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--role',choices=['send','receive'],required=True)
    p.add_argument('--node',choices=['A','B'],required=True)
    p.add_argument('--model',choices=['opt-125m','opt-1.3b'],required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--port',type=int,default=5211)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    if not 3<=args.repeats<=1000 or not 1<=args.warmup<=100: p.error('Invalid sample count')
    args.out.mkdir(parents=True,exist_ok=False)
    begin=ms()
    torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.4)
    cfg={'model':'facebook/'+args.model,'weights_root':'~/opt_weights','node_id':args.node,'nic':'enp4s0','gpu_index':0}
    inventory(args.out,cfg)
    write_json(args.out/'invocation.json',vars(args))
    write_json(args.out/'source_sha256.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    sock=None;listener=None
    try:
        with torch.inference_mode(),Memory() as memory:
            blocks=Blocks(cfg,1)
            init_ms=ms()-begin
            if args.role=='receive':
                listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                listener.bind((args.host,args.port));listener.listen(1);listener.settimeout(60)
                print('LISTENING',flush=True)
                sock,_=listener.accept()
            else:
                sock=socket.create_connection((args.host,args.port),timeout=30)
            sock.settimeout(30);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            values=[]
            for i,c in enumerate(cases(args.model,args.quick)):
                result=run_case(sock,c,i,args,blocks)
                values.append(result)
                write_json(args.out/'raw.json',values)
                print('case',i,c['shape'],c['mode'],'PASS',flush=True)
                gc.collect()
        write_json(args.out/'completion.json',{'status':'pass','cases':len(values),'initialization_ms':init_ms,
            'elapsed_ms':ms()-begin,'memory':memory.values,'resident_background_reference_weight_bytes':blocks.bytes,
            'resident_pinned_background_weights_bytes':blocks.bytes})
    except BaseException as exc:
        write_json(args.out/'failure.json',{'error':repr(exc)})
        raise
    finally:
        if sock: sock.close()
        if listener: listener.close()


if __name__=='__main__': main()
