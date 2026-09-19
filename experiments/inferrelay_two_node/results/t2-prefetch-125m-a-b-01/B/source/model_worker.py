"""T2/T3 shared executor for single-node, continuous and alternating OPT."""
import argparse
import gc
import hashlib
import json
import socket
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from .executor import Executor
from .activation import event,ms
from .wire import send_json,recv_json,receive_into,validate
from experiments.inferrelay_precheck.common import inventory,Memory,write_json

ATOL=.02
RTOL=.01


class Channel:
    def __init__(self,sock,batch,q,h,runid,weights):
        self.sock=sock;self.runid=runid;self.weights=weights
        self.host=torch.empty(batch*q*h,dtype=torch.float16,pin_memory=True)
        self.gpu=torch.empty_like(self.host,device='cuda');self.stream=torch.cuda.Stream(priority=-1)
        self.reset()

    def reset(self):self.records=[];self.sent_bytes=0;self.payload_bytes=0
    def send(self,obj):
        self.sent_bytes+=4+len(json.dumps(obj,separators=(',',':')).encode());send_json(self.sock,obj)
    def recv(self):return recv_json(self.sock)
    def barrier(self,key):self.send({'barrier':key});validate(self.recv(),{'barrier':key})

    def transfer(self,x,shape,request,step,group,sender):
        count=int(np.prod(shape));host=self.host[:count].view(shape);view=memoryview(host.numpy()).cast('B')
        expected=dict(kind='activation',run_id=self.runid,request_id=request,step_id=step,group_id=group,shape=list(shape),dtype='float16',bytes=count*2)
        start=ms();self.weights.pause.set()
        try:
            if sender:self.send(expected);validate(self.recv(),{'ready':step,'group':group})
            else:validate(self.recv(),expected);self.send({'ready':step,'group':group});receive_into(self.sock,view)
            stage=ms();a,b=event(),event();ready=event();ready.record()
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready);a.record()
                if sender:host.copy_(x,non_blocking=True)
                else:self.gpu[:count].view(shape).copy_(host,non_blocking=True)
                b.record()
            b.synchronize();stage_end=ms()
            if sender:
                self.sock.sendall(view);self.sent_bytes+=count*2;self.payload_bytes+=count*2
                validate(self.recv(),{'ack':step,'group':group})
            else:self.send({'ack':step,'group':group})
        finally:self.weights.pause.clear()
        self.records.append(dict(step=step,group=group,sender=sender,bytes=count*2,wall_ms=ms()-start,
            staging_host_ms=stage_end-stage,staging_cuda_ms=a.elapsed_time(b),
            local_host_start_ms=start-self.origin,local_host_end_ms=ms()-self.origin,
            gpu_start_ms=self.anchor.elapsed_time(a),gpu_end_ms=self.anchor.elapsed_time(b)))
        if not sender:return self.gpu[:count].view(shape)


def prompt_ids(batch,q):
    tokenizer=AutoTokenizer.from_pretrained('facebook/opt-30b',local_files_only=True)
    text=('The laboratory compared two computers while processing a sequence of words. '
          'Each layer transformed its input and preserved attention history for the next token. '
          'A fair experiment measures communication, memory transfers, and computation separately. ')*30
    ids=tokenizer(text,add_special_tokens=True)['input_ids']
    assert len(ids)>q+batch*7
    out=torch.tensor([ids[i*7:i*7+q] for i in range(batch)],dtype=torch.long,device='cuda')
    assert not bool((out==1).any())
    return out


def compare_kv(got,expected):
    worst=0.
    for layer,row in got.items():
        ref=expected[layer]
        assert row['length']==ref['length'] and row['last_position']==ref['last_position'],layer
        for name in ['k','v']:
            np.testing.assert_allclose(row[name],ref[name],atol=ATOL,rtol=RTOL,err_msg=f'KV layer {layer} {name}')
            worst=max(worst,float(np.max(np.abs(row[name].astype(float)-ref[name].astype(float)))))
    return worst


def run_request(exe,channel,owners,ids,steps,request,reference=None,capture=False):
    batch,q=ids.shape;exe.reset(batch,q,steps)
    if channel:channel.reset();channel.barrier('start-'+str(request));channel.reset()
    torch.cuda.synchronize()
    anchor=event();anchor.record();origin=ms()
    if channel:channel.anchor=anchor;channel.origin=origin
    exe.weights.begin(exe.owned*steps,anchor)
    counter=0;outputs=[];logits=[];snapshots=[];step_times=[];endpoint=[]
    current=ids;past=0
    for step in range(steps):
        start=ms();epstart=ms()
        x=exe.embed(current,past+current.shape[1]) if exe.first else None
        embed_submit_ms=ms()-epstart
        shape=[batch,q if step==0 else 1,exe.cfg.input_dim]
        for g,owner in enumerate(owners):
            if owner!=exe.node:continue
            if g and owners[g-1]!=owner:
                x=channel.transfer(None,shape,request,step,g,False)
            x=exe.group(g,x,past,counter);counter+=1
            if g+1<len(owners) and owners[g+1]!=owner:
                channel.transfer(x,shape,request,step,g+1,True)
        headstart=ms()
        if exe.last:
            ls=exe.logits(x);token=ls.argmax(-1,keepdim=True);token_cpu=token.cpu().numpy()
            logits.append(ls.detach().clone())
            if channel:channel.send({'kind':'token','run_id':channel.runid,'request_id':request,'step_id':step,'ids':token_cpu.tolist()})
        if exe.first and channel:
            msg=channel.recv();validate(msg,{'kind':'token','run_id':channel.runid,'request_id':request,'step_id':step})
            token_cpu=np.asarray(msg['ids'],dtype=np.int64);assert token_cpu.shape==(batch,1)
            token=torch.tensor(token_cpu,device='cuda')
        head_token_wall_ms=ms()-headstart
        outputs.append(token_cpu.copy());current=token if exe.first else current[:,:1]
        past+=q if step==0 else 1
        step_times.append(ms()-start);endpoint.append(dict(step=step,embed_host_submit_ms=embed_submit_ms,head_and_token_host_ms=head_token_wall_ms))
        if capture:snapshots.append(exe.samples())
    elapsed=ms()-origin
    copies=exe.weights.finish();compute=exe.trace(anchor)
    for row in copies:
        i=row['index']
        assert row['gpu_end_ms']<=compute[i]['gpu_start_ms']+.01,('copy before compute',i)
        prior=i-len(exe.weights.buffers)
        if prior>=0:assert row['gpu_start_ms']>=compute[prior]['gpu_end_ms']-.01,('slot reused early',i)
    logcpu=[t.cpu().numpy() for t in logits]
    kv=exe.samples()
    errors={'tokens_exact':True,'logits_max_abs':0.,'kv_max_abs':0.}
    if reference is not None:
        for s,(got,want) in enumerate(zip(outputs,reference['tokens'])):np.testing.assert_array_equal(got,want,err_msg=f'token step {s}')
        for s,(got,want) in enumerate(zip(logcpu,reference['logits'])):
            np.testing.assert_allclose(got,want,atol=ATOL,rtol=RTOL,err_msg=f'logits step {s}')
            errors['logits_max_abs']=max(errors['logits_max_abs'],float(np.max(np.abs(got.astype(float)-want.astype(float)))))
        errors['kv_max_abs']=compare_kv(kv,reference['kv_steps'][-1])
        if capture:
            for got,want in zip(snapshots,reference['kv_steps']):errors['kv_max_abs']=max(errors['kv_max_abs'],compare_kv(got,want))
    return dict(e2e_ms=elapsed,step_ms=step_times,capture_overhead_included=capture,errors=errors,correct=True,
        copies=copies,compute=compute,communication=channel.records if channel else [],endpoint=endpoint,
        sent_wire_bytes=channel.sent_bytes if channel else 0,activation_payload_bytes=channel.payload_bytes if channel else 0,
        endpoint_weight_bytes=exe.endpoint_bytes,kv_bytes=exe.kv_bytes,weight_slot_bytes=len(exe.weights.buffers)*exe.weights.slot_bytes,
        resident_block_bytes=sum(t.numel()*2 for t in exe.weights.resident.values()),
        gpu_budget_bytes=4*1024**3,global_layers=sorted(exe.kv),
        pinned_weight_bytes=sum(t.numel()*2 for t in exe.weights.host.values()),tokens=[t.tolist() for t in outputs]),dict(tokens=outputs,logits=logcpu,kv_steps=snapshots)


def save_reference(path,ref):
    data={}
    for i,t in enumerate(ref['tokens']):data[f'token_{i}']=t
    for i,t in enumerate(ref['logits']):data[f'logits_{i}']=t
    for step,rows in enumerate(ref['kv_steps']):
        for layer,row in rows.items():
            for name in ['k','v']:data[f'kv_{step}_{layer}_{name}']=row[name]
            data[f'length_{step}_{layer}']=np.asarray(row['length'])
    np.savez_compressed(path,**data)


def main():
    p=argparse.ArgumentParser()
    for k in ['role','node','model','host','run-id']:p.add_argument('--'+k,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--port',type=int,default=5211);p.add_argument('--quick',action='store_true');p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);settings=json.loads(args.config.read_text())
    torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    if free<4*1024**3:raise RuntimeError('Need 4GiB available without disturbing other tasks')
    torch.cuda.set_per_process_memory_fraction(4*1024**3/total)
    inventory(args.out,{'model':args.model,'node_id':args.node,'settings':settings})
    write_json(args.out/'invocation.json',vars(args));write_json(args.out/'config.json',settings)
    src=args.out/'source';src.mkdir()
    for f in Path(__file__).parent.glob('*.py'):(src/f.name).write_bytes(f.read_bytes())
    write_json(args.out/'source_sha256.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*.py')})
    node=0 if args.role=='send' else 1;sock=None;listener=None;exe=None;results=[]
    try:
        if node==1:
            listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind((args.host,args.port));listener.listen(1);listener.settimeout(60)
            print('LISTENING',flush=True);sock,_=listener.accept()
        else:sock=socket.create_connection((args.host,args.port),30)
        sock.settimeout(120);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        layers=12 if args.model=='opt-125m' else 24;quarter=layers//4
        groups=[(i*quarter,(i+1)*quarter) for i in range(4)]
        for wi,w in enumerate(settings['workloads']):
          ids=prompt_ids(w['batch'],w['q']);steps=w['steps']
          # Independent same-backend full-model reference on each GPU; no reference tensors remain on GPU afterwards.
          with torch.inference_mode(),Memory() as mem:
            exe=Executor(args.model,groups,[0]*4,0,'resident',quarter)
            refrow,reference=run_request(exe,None,[0]*4,ids,steps,0,capture=True)
            resident_row,_=run_request(exe,None,[0]*4,ids,steps,1,reference=reference,capture=False)
            save_reference(args.out/f'reference-{wi}.npz',reference)
            exe.weights.close();exe=None
          write_json(args.out/f'reference-{wi}.json',dict(workload=w,correct=True,tolerances={'atol':ATOL,'rtol':RTOL},diagnostic=refrow,
              uninstrumented_kv_reference_run=resident_row,memory=mem.values))
          gc.collect();torch.cuda.empty_cache()
          send_json(sock,{'reference_ready':wi});validate(recv_json(sock),{'reference_ready':wi})
          for ci,case in enumerate(settings['cases']):
            owners=[0,1,0,1] if case['layout']=='relay' else [0]*case.get('cut',2)+[1]*(4-case.get('cut',2))
            with torch.inference_mode(),Memory() as mem:
              exe=Executor(args.model,groups,owners,node,case['mode'],quarter,case.get('policy','priority'))
              channel=Channel(sock,w['batch'],w['q'],exe.cfg.input_dim,args.run_id,exe.weights)
              rows=[]
              for sample in range(settings.get('warmup',1)+settings.get('repeats',args.repeats)):
                row,_=run_request(exe,channel,owners,ids,steps,wi*10000+ci*100+sample,reference=reference,capture=settings['stage']=='t2')
                row.update(sample=sample,warmup=sample<settings.get('warmup',1));rows.append(row)
                channel.barrier('checked-'+str(sample))
              exe.weights.close()
              metadata=dict(workload=w,case=case,owners=owners,groups=groups,rows=rows)
              exe=None;channel=None
            metadata['memory']=mem.values;results.append(metadata)
            write_json(args.out/'raw.json',results)
            print('PASS workload',wi,'case',ci,case,flush=True)
            gc.collect();torch.cuda.empty_cache()
          del reference,ids
        write_json(args.out/'completion.json',{'correct':True,'cases':len(results),'stage':settings['stage']})
    except BaseException as exc:
        write_json(args.out/'failure.json',{'error':repr(exc)});raise
    finally:
        if exe:exe.weights.close()
        if sock:sock.close()
        if listener:listener.close()


if __name__=='__main__':main()
