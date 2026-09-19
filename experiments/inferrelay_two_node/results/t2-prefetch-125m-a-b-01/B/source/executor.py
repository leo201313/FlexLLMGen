"""Shared real-global-layer OPT executor and bounded weight slots, for C and R."""
import collections
import threading
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import flexllmgen.flex_opt
from flexllmgen.opt_config import get_opt_config
from flexllmgen.pytorch_backend import TorchDevice,TorchTensor
from experiments.inferrelay_precheck.components import NAMES
from .activation import event,ms


class WeightSlots:
    def __init__(self, model, groups, owned, mode, slot_layers, policy='priority',slots=2):
        self.mode=mode;self.groups=groups;self.owned=owned;self.policy=policy
        root=Path.home()/'opt_weights'/(model+'-np')
        self.host={};self.shapes={};self.sizes={}
        for g in owned:
            ts=[torch.from_numpy(np.load(root/f'decoder.layers.{layer}.{name}')).half() for layer in range(*groups[g]) for name in NAMES]
            self.shapes[g]=[t.shape for t in ts];self.sizes[g]=[t.numel() for t in ts]
            self.host[g]=torch.cat([t.flatten() for t in ts]).pin_memory()
        self.bytes_per_layer=next(iter(self.host.values())).numel()*2//(groups[owned[0]][1]-groups[owned[0]][0])
        self.slot_bytes=slot_layers*self.bytes_per_layer
        assert all(h.numel()*2<=self.slot_bytes for h in self.host.values())
        self.buffers=[torch.empty(self.slot_bytes//2,device='cuda',dtype=torch.float16) for _ in range(slots)] if mode!='resident' else []
        self.resident={g:h.cuda() for g,h in self.host.items()} if mode=='resident' else {}
        self.stream=torch.cuda.Stream();self.pause=threading.Event();self.stop=threading.Event()
        self.condition=threading.Condition();self.thread=None

    def unpack(self,g,flat):
        return [part.view(shape) for part,shape in zip(flat[:sum(self.sizes[g])].split(self.sizes[g]),self.shapes[g])]

    def begin(self,sequence,anchor):
        self.sequence=sequence;self.anchor=anchor;self.records=[];self.error=None;self.pause.clear();self.stop.clear()
        self.loaded={};self.released={};self.thread=None
        if self.mode=='prefetch':
            self.thread=threading.Thread(target=self.produce,daemon=True);self.thread.start()

    def copy(self,index,g):
        slot=index%len(self.buffers);wait=ms();prior=index-len(self.buffers)
        if prior>=0:
            with self.condition:
                if not self.condition.wait_for(lambda:prior in self.released or self.stop.is_set(),timeout=30):raise TimeoutError('Slot release deadline')
                if self.stop.is_set():raise RuntimeError('Prefetch cancelled')
                free=self.released[prior]
        else:free=None
        host_buffer_wait=ms()-wait
        source=self.host[g];dest=self.buffers[slot][:source.numel()]
        a,b,w0,w1=[event() for _ in range(4)];t=ms();call_ms=0.
        chunk=source.numel() if self.policy=='bulk' else 4*1024**2
        pending=collections.deque()
        with torch.cuda.stream(self.stream),torch.inference_mode():
            self.stream.wait_event(self.anchor);w0.record()
            if free:self.stream.wait_event(free)
            w1.record();a.record()
            for offset in range(0,source.numel(),chunk):
                cap=2 if self.policy=='window2' else 1
                if self.policy!='bulk' and len(pending)>=cap:pending.popleft().synchronize()
                while self.policy=='priority' and self.pause.is_set():
                    if self.stop.is_set():raise RuntimeError('Prefetch cancelled')
                    time.sleep(.00005)
                before=ms();dest[offset:offset+chunk].copy_(source[offset:offset+chunk],non_blocking=True);call_ms+=ms()-before
                if self.policy!='bulk':
                    chunk_done=event();chunk_done.record();pending.append(chunk_done)
            while pending:pending.popleft().synchronize()
            b.record()
        self.records.append(dict(index=index,group=g,slot=slot,bytes=source.numel()*2,host_submit_ms=ms()-t,copy_call_ms=call_ms,
            buffer_host_wait_ms=host_buffer_wait,a=a,b=b,w0=w0,w1=w1))
        with self.condition:self.loaded[index]=b;self.condition.notify_all()

    def produce(self):
        try:
            torch.cuda.set_device(0)
            for i,g in enumerate(self.sequence):
                if self.stop.is_set():return
                self.copy(i,g)
        except BaseException as exc:
            with self.condition:self.error=exc;self.condition.notify_all()

    def acquire(self,index,g):
        if self.mode=='resident':return self.unpack(g,self.resident[g]),0.,None
        if self.mode=='demand':self.copy(index,g)
        t=ms()
        with self.condition:
            if not self.condition.wait_for(lambda:index in self.loaded or self.error is not None,timeout=30):raise TimeoutError('Weight ready deadline')
            if self.error:raise self.error
            ready=self.loaded[index]
        hostwait=ms()-t
        a,b=event(),event();a.record();torch.cuda.current_stream().wait_event(ready);b.record()
        return self.unpack(g,self.buffers[index%len(self.buffers)]),hostwait,(a,b)

    def release(self,index,done):
        if self.mode!='resident':
            with self.condition:self.released[index]=done;self.condition.notify_all()

    def finish(self):
        if self.thread:
            self.thread.join(30)
            if self.thread.is_alive():raise TimeoutError('Prefetch finish deadline')
        if self.error:raise self.error
        out=[]
        for r in self.records:
            r['b'].synchronize()
            out.append({**{k:v for k,v in r.items() if k not in ['a','b','w0','w1']},
                'cuda_ms':r['a'].elapsed_time(r['b']),'gpu_start_ms':self.anchor.elapsed_time(r['a']),
                'gpu_end_ms':self.anchor.elapsed_time(r['b']),'buffer_gpu_wait_ms':r['w0'].elapsed_time(r['w1'])})
        return out

    def close(self):
        self.stop.set();self.pause.clear()
        with self.condition:self.condition.notify_all()
        if self.thread:self.thread.join(5)


class Executor:
    def __init__(self,model,groups,owners,node,mode,slot_layers,policy='priority'):
        self.cfg=get_opt_config(model);self.groups=groups;self.node=node
        self.owned=[i for i,n in enumerate(owners) if n==node]
        self.device=TorchDevice('cuda:0')
        self.weights=WeightSlots(model,groups,self.owned,mode,slot_layers,policy)
        self.first=owners[0]==node;self.last=owners[-1]==node
        root=Path.home()/'opt_weights'/(model+'-np')
        self.endpoint={}
        names=[]
        if self.first:names+=['embed_tokens.weight','embed_positions.weight']
        if self.last:names+=['embed_tokens.weight','layer_norm.weight','layer_norm.bias']
        for name in set(names):self.endpoint[name]=torch.from_numpy(np.load(root/('decoder.'+name))).half().cuda()
        self.endpoint_bytes=sum(t.numel()*2 for t in self.endpoint.values())
        self.kv={};self.compute_records=[]

    def wrap(self,t):return TorchTensor.create_from_torch(t,self.device)

    def reset(self,batch,prompt,steps):
        self.kv={};self.lengths={};self.compute_records=[];self.wait_records=[]
        capacity=prompt+steps-1
        for g in self.owned:
            for layer in range(*self.groups[g]):
                self.kv[layer]=[torch.empty(capacity,batch*self.cfg.n_head,self.cfg.input_dim//self.cfg.n_head,device='cuda',dtype=torch.float16) for _ in range(2)]
                self.lengths[layer]=0
        self.kv_bytes=sum(t.numel()*2 for pair in self.kv.values() for t in pair)

    def embed(self,ids,length):
        mask=self.wrap(torch.ones(ids.shape[0],length,device='cuda',dtype=torch.bool))
        return self.device.opt_input_embed(self.wrap(ids),mask,self.wrap(self.endpoint['embed_tokens.weight']),self.wrap(self.endpoint['embed_positions.weight']),self.cfg.pad,[False]*4).data

    def group(self,g,x,past,index):
        tensors,hostwait,gpuwait=self.weights.acquire(index,g)
        weights=[[self.wrap(t) for t in tensors[i:i+16]] for i in range(0,len(tensors),16)]
        a,b=event(),event();a.record()
        mask=self.wrap(torch.ones(x.shape[0],past+x.shape[1],device='cuda',dtype=torch.bool))
        h=self.wrap(x)
        for layer,ws in zip(range(*self.groups[g]),weights):
            assert self.lengths[layer]==past,(layer,self.lengths[layer],past)
            if past:
                h,k,v=self.device.mha_gen(h,mask,*ws[:10],self.cfg.n_head,*[self.wrap(t) for t in self.kv[layer]],[False]*14,1.,False,None)
            else:
                h,k,v=self.device.mha(h,mask,*ws[:10],self.cfg.n_head,[False]*12,False,None)
                self.kv[layer][0][:x.shape[1]].copy_(k.data);self.kv[layer][1][:x.shape[1]].copy_(v.data)
            h=self.device.mlp(h,*ws[10:],[False]*7)
            self.lengths[layer]=past+x.shape[1]
        b.record();self.weights.release(index,b)
        self.compute_records.append(dict(group=g,past=past,a=a,b=b,weight_host_wait_ms=hostwait,gpuwait=gpuwait))
        return h.data

    def logits(self,x):
        # Same operations as original output embedding, but expose all last-token logits.
        y=F.layer_norm(x,(self.cfg.input_dim,),weight=self.endpoint['layer_norm.weight'],bias=self.endpoint['layer_norm.bias'])
        return F.linear(y,self.endpoint['embed_tokens.weight'])[:,-1,:]

    def samples(self):
        result={}
        for layer,pair in self.kv.items():
            length=self.lengths[layer];values=[]
            for t in pair:
                flat=t[:length].reshape(-1);idx=torch.linspace(0,len(flat)-1,min(256,len(flat)),device='cuda').long()
                values.append(flat[idx].cpu().numpy().copy())
            result[layer]={'length':length,'last_position':length-1,'k':values[0],'v':values[1]}
        return result

    def trace(self,anchor):
        out=[]
        for r in self.compute_records:
            r['b'].synchronize();wait=r['gpuwait']
            out.append(dict(group=r['group'],past=r['past'],compute_cuda_ms=r['a'].elapsed_time(r['b']),
                gpu_start_ms=anchor.elapsed_time(r['a']),gpu_end_ms=anchor.elapsed_time(r['b']),
                weight_host_wait_ms=r['weight_host_wait_ms'],weight_gpu_wait_ms=wait[0].elapsed_time(wait[1]) if wait else 0))
        return out
