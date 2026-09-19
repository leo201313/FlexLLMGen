"""Render existing event traces as a self-contained interactive engineering timeline."""
import csv
import json
import sys
from pathlib import Path


def render(out):
    out=Path(out)
    datasets=[]
    for path in sorted((out/'e0').glob('trace_*.csv')):
        events=[]
        for r in csv.DictReader(path.open()):
            events.append(dict(track=r['kind'],start=float(r['start_ms']),end=float(r['end_ms']),
                               token=int(r['token']),name=f"L{r['layer']} token {r['token']} {r['kind']}"))
        datasets.append(dict(name='Measured E0: '+path.stem,events=events))
    model=out/'e3/example_timeline.json'
    if model.exists():
        model=json.loads(model.read_text())
        if model:
            for mode in ['contiguous','relay']:
                events=[dict(track=e['resource'],start=e['start_ms'],end=e['end_ms'],token=-1,
                             name=f"{e['kind']} request {e['request']} group {e['group']} slot {e['slot']}")
                        for e in model[mode]['events']]
                datasets.append(dict(name='Predicted E3: '+mode,events=events))
    html='''<!doctype html><meta charset="utf-8"><title>InferRelay precheck timeline</title>
<style>body{font:16px system-ui;margin:32px;max-width:1400px;background:#fafafa;color:#162030}
select,input{font:inherit;margin:8px;padding:4px} .row{display:flex;align-items:center;margin:12px 0}
.label{width:160px;flex-shrink:0}.lane{height:30px;background:#e7ebef;position:relative;flex:1}
.bar{position:absolute;height:24px;top:3px;background:#167d9a;border-radius:2px;min-width:1px}
.weight_load,.h2d0,.h2d1{background:#cd7925}.network{background:#7d4eb2}
.d2h0,.d2h1{background:#b95e68}small{color:#556}#details{white-space:pre-wrap}</style>
<h1>InferRelay event timeline</h1>
<p>E0: measured CUDA event spans, including possible host submission gaps. E3: predicted resource reservations, not two-machine measurements. Weight load in the all-GPU arm is not H2D.</p>
<label>Dataset <select id="dataset"></select></label>
<label>E0 token (−1 = all) <input id="token" type="number" min="-1" value="1" max="15"></label>
<p id="range"></p><div id="chart"></div><p id="details">Hover over a bar for exact interval and layer/group.</p>
<small>E0 token 0 is prefill; subsequent tokens are decode. No causal wait is inferred solely from an empty interval. Raw events and sampling definitions remain in the adjacent CSV/JSON files.</small>
<script>
const data=__DATA__;
const select=document.querySelector('#dataset'), token=document.querySelector('#token');
data.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.name;select.appendChild(o)});
function draw(){
 let d=data[Number(select.value)];if(!d)return;
 let es=d.events.filter(e=>d.name.startsWith('Predicted')||Number(token.value)<0||e.token===Number(token.value));
 let chart=document.querySelector('#chart');chart.replaceChildren();if(!es.length)return;
 let lo=Math.min(...es.map(e=>e.start)),hi=Math.max(...es.map(e=>e.end)),span=hi-lo||1;
 document.querySelector('#range').textContent=`${d.name} | ${lo.toFixed(3)}–${hi.toFixed(3)} ms | span ${span.toFixed(3)} ms`;
 [...new Set(es.map(e=>e.track))].sort().forEach(track=>{
  let row=document.createElement('div');row.className='row';
  let label=document.createElement('div');label.className='label';label.textContent=track;
  let lane=document.createElement('div');lane.className='lane';
  es.filter(e=>e.track===track).forEach(e=>{let bar=document.createElement('div');bar.className='bar '+track;
   bar.style.left=100*(e.start-lo)/span+'%';bar.style.width=100*(e.end-e.start)/span+'%';
   bar.title=`${e.name}: ${e.start.toFixed(4)}–${e.end.toFixed(4)} ms (${(e.end-e.start).toFixed(4)} ms)`;
   bar.onmouseenter=()=>document.querySelector('#details').textContent=bar.title;lane.appendChild(bar)});
  row.append(label,lane);chart.appendChild(row);
 });
}
select.onchange=draw;token.oninput=draw;draw();
</script>'''
    (out/'timeline.html').write_text(html.replace('__DATA__',json.dumps(datasets)))


if __name__=='__main__':
    render(sys.argv[1])
