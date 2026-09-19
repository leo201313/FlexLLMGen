# InferRelay precheck: single-machine results

Measured E0–E2; E3 is a parameterized prediction. OPT-125M forced offload is a mechanism test, not evidence of large-model necessity or two-machine speedup.

## E1: actual block-layout transfer and computation

|blocks|FP16 weights MiB|pageable P50 ms|pinned P50 ms|pinned GB/s|
|---|---:|---:|---:|---:|
|1|13.52|0.744|0.601|23.59|
|2|27.04|1.672|1.201|23.61|
|4|54.08|3.938|2.401|23.62|
|8|108.15|8.321|4.799|23.63|

Measured H2D/compute ratios span 1.005–5.811. All query/batch/history combinations are retained in summary.json; raw.csv includes every sample.

Decision: retain near-matching shapes for E2 and the network sweep. q>1 is ordinary short-prompt prefill with past=0, not incremental/chunked prefill. Decode uses real attention KV prepared from synthetic hidden activations through actual pretrained blocks; it is a kernel workload, not full-model semantic validation.

## E2: measured concurrency

56/56 shapes have >1% speedup versus serial; 0/56 regress by >1%. These thresholds summarize measurements, not statistical significance.

|blocks|query|past|batch|serial/concurrent P50 ms|speedup|copy slowdown|compute slowdown|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|16|0|1|0.698/0.618|1.129|1.006|1.055|
|1|16|0|4|0.730/0.624|1.169|1.007|1.066|
|1|32|0|1|0.709/0.619|1.145|1.007|1.070|
|1|32|0|4|0.773/0.621|1.245|1.007|1.073|
|1|64|0|1|0.721/0.618|1.166|1.006|1.076|
|1|64|0|4|0.809/0.619|1.308|1.007|1.044|
|1|128|0|1|0.762/0.619|1.232|1.007|1.092|
|1|128|0|4|0.982/0.620|1.583|1.008|1.036|
|1|256|0|1|0.813/0.618|1.314|1.007|1.046|
|1|256|0|4|1.216/0.666|1.826|1.011|1.007|
|1|1|32|1|0.713/0.618|1.155|1.005|1.121|
|1|1|32|4|0.689/0.623|1.107|1.005|1.037|
|1|1|256|1|0.716/0.620|1.154|1.005|1.108|
|1|1|256|4|0.695/0.622|1.118|1.006|1.032|
|2|16|0|1|1.397/1.224|1.141|1.007|1.047|
|2|16|0|4|1.444/1.225|1.179|1.008|1.052|
|2|32|0|1|1.402/1.227|1.143|1.007|1.042|
|2|32|0|4|1.530/1.225|1.249|1.008|1.071|
|2|64|0|1|1.427/1.224|1.166|1.007|1.060|
|2|64|0|4|1.599/1.227|1.303|1.008|1.027|
|2|128|0|1|1.494/1.227|1.218|1.007|1.040|
|2|128|0|4|1.954/1.226|1.593|1.009|1.028|
|2|256|0|1|1.609/1.225|1.313|1.008|1.049|
|2|256|0|4|2.415/1.294|1.866|1.010|1.004|
|2|1|32|1|1.406/1.220|1.152|1.004|1.055|
|2|1|32|4|1.378/1.223|1.127|1.006|1.026|
|2|1|256|1|1.412/1.221|1.156|1.004|1.063|
|2|1|256|4|1.400/1.222|1.146|1.006|1.030|
|4|16|0|1|2.817/2.430|1.159|1.006|1.035|
|4|16|0|4|2.899/2.434|1.191|1.007|1.074|
|4|32|0|1|2.834/2.431|1.166|1.007|1.043|
|4|32|0|4|3.053/2.435|1.254|1.008|1.102|
|4|64|0|1|2.867/2.431|1.179|1.006|1.084|
|4|64|0|4|3.182/2.435|1.307|1.008|1.033|
|4|128|0|1|3.009/2.435|1.236|1.007|1.106|
|4|128|0|4|3.884/2.433|1.597|1.008|1.021|
|4|256|0|1|3.198/2.433|1.314|1.007|1.053|
|4|256|0|4|4.806/2.532|1.898|1.008|1.001|
|4|1|32|1|2.797/2.424|1.154|1.003|1.032|
|4|1|32|4|2.798/2.429|1.152|1.006|1.042|
|4|1|256|1|2.805/2.426|1.156|1.004|1.043|
|4|1|256|4|2.837/2.430|1.168|1.006|1.084|
|8|16|0|1|5.608/4.844|1.158|1.006|1.034|
|8|16|0|4|5.776/4.850|1.191|1.007|1.059|
|8|32|0|1|5.646/4.847|1.165|1.006|1.026|
|8|32|0|4|6.088/4.849|1.256|1.007|1.114|
|8|64|0|1|5.716/4.846|1.179|1.006|1.082|
|8|64|0|4|6.341/4.853|1.307|1.007|1.031|
|8|128|0|1|6.002/4.849|1.238|1.007|1.109|
|8|128|0|4|7.754/4.854|1.597|1.007|1.017|
|8|256|0|1|6.376/4.849|1.315|1.007|1.055|
|8|256|0|4|9.586/5.026|1.907|1.007|1.004|
|8|1|32|1|5.651/4.842|1.167|1.006|1.060|
|8|1|32|4|5.574/4.843|1.151|1.006|1.031|
|8|1|256|1|5.670/4.842|1.171|1.006|1.053|
|8|1|256|4|5.656/4.843|1.168|1.006|1.083|

Overlap efficiency = (isolated_copy + isolated_compute - concurrent_makespan) / min(isolated_copy, isolated_compute); it is intentionally not clamped. Independent CUDA streams do not guarantee a win. Isolated references were measured earlier in E1; DVFS and temperature drift can affect slowdown ratios.

Two-slot lifecycle traces check copy→read and prior-read→reuse dependencies and exact numerical output. The pipeline repeatedly computes the same independent group workload; it is not full model inference. Original FlexGen overlap is the E0 cpu_overlap arm, not a like-for-like component makespan.

Decision: use both the isolated and measured-contention scenarios in E3. Neither a positive component speedup nor an event-ordering check establishes relay end-to-end benefit.

## Next minimum changes

1. Repeat representative configurations with longer sampling and randomized order; keep both positive and negative shapes.
2. Obtain approval before downloading OPT-1.3B; this still fits 16GB and is only a larger mechanism test.
3. Implement and numerically validate incremental prefill before describing a q sweep as chunked prefill.
4. Measure a real second endpoint and calibrate staging, H2D contention, weight reuse and queue scheduling before making a system claim.

Memory: GPU allocated/reserved peaks are allocator measurements. RSS is sampled every 10ms. Explicit pinned tensor bytes are tracked; absent host allocator statistics are null, not zero. PyTorch pinned caching can retain more physical RAM than the live tensors. No deliberate OOM stress was performed; rejected/failing cases are written to failures.jsonl when present.
