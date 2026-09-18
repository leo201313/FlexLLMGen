# InferRelay precheck: single-machine results

Measured E0–E2; E3 is a parameterized prediction. OPT-125M forced offload is a mechanism test, not evidence of large-model necessity or two-machine speedup.

## E0: original generation loops

|case|wall P50/P95 ms|prefill P50 ms|decode P50 ms|GPU peak MiB|
|---|---:|---:|---:|---:|
|gpu_p32_b1_m1|10.992/11.056|2.743|8.103|332.6|
|cpu_serial_p32_b1_m1|108.459/108.479|27.173|81.168|87.1|
|cpu_overlap_p32_b1_m1|105.105/105.184|26.178|72.513|164.1|
|gpu_p32_b1_m2|21.044/21.240|5.237|15.558|334.1|
|cpu_serial_p32_b1_m2|113.821/113.969|28.570|85.054|88.4|
|cpu_overlap_p32_b1_m2|115.589/115.687|28.851|80.314|165.4|
|gpu_p128_b1_m1|11.199/11.243|3.089|7.958|345.5|
|cpu_serial_p128_b1_m1|109.115/109.147|27.874|81.078|99.8|
|cpu_overlap_p128_b1_m1|105.013/105.021|26.156|72.482|176.8|
|gpu_p128_b1_m2|22.156/22.479|6.175|15.721|350.3|
|cpu_serial_p128_b1_m2|115.388/115.529|29.946|85.162|104.6|
|cpu_overlap_p128_b1_m2|116.234/116.290|29.371|80.436|181.6|

All successful modes require exact generated-ID equality with the all-GPU baseline. Phase times are CUDA-event spans, including host launch gaps; wall time includes generation setup/cleanup. Fixed-length throughput is not online goodput.

Trace CSV/JSON contains weight submission spans, layer compute and original CPU global-sync waits. Weight-ready gaps are a diagnostic proxy, not causal H2D wait attribution. Trace runs are separate from the baseline samples.

- gpu_p128_b1_m1: trace wall/baseline P50 = 1.236; weight-ready gap proxy = 0.000 ms. Single diagnostic sample; do not correct throughput by this ratio.
- cpu_serial_p128_b1_m1: trace wall/baseline P50 = 1.004; weight-ready gap proxy = 96.761 ms. Single diagnostic sample; do not correct throughput by this ratio.
- cpu_overlap_p128_b1_m1: trace wall/baseline P50 = 1.008; weight-ready gap proxy = 79.177 ms. Single diagnostic sample; do not correct throughput by this ratio.

Decision: the baseline and timing path are executable. Inspect the full CPU serial/overlap table before claiming remaining prefetch opportunity; do not extrapolate the small-model ratio to a memory-constrained model.

## E1: actual block-layout transfer and computation

|blocks|FP16 weights MiB|pageable P50 ms|pinned P50 ms|pinned GB/s|
|---|---:|---:|---:|---:|
|1|13.52|1.250|1.114|12.72|
|2|27.04|2.680|2.232|12.70|

Measured H2D/compute ratios span 6.634–10.146. All query/batch/history combinations are retained in summary.json; raw.csv includes every sample.

Decision: retain near-matching shapes for E2 and the network sweep. q>1 is ordinary short-prompt prefill with past=0, not incremental/chunked prefill. Decode uses real attention KV prepared from synthetic hidden activations through actual pretrained blocks; it is a kernel workload, not full-model semantic validation.

## E2: measured concurrency

8/8 shapes have >1% speedup versus serial; 0/8 regress by >1%. These thresholds summarize measurements, not statistical significance.

|blocks|query|past|batch|serial/concurrent P50 ms|speedup|copy slowdown|compute slowdown|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|16|0|1|1.222/1.140|1.071|1.006|1.065|
|1|128|0|1|1.280/1.138|1.125|1.006|1.037|
|1|1|32|1|1.230/1.134|1.084|1.006|1.177|
|1|1|256|1|1.233/1.135|1.087|1.006|1.216|
|2|16|0|1|2.432/2.259|1.077|1.004|1.095|
|2|128|0|1|2.545/2.258|1.127|1.005|1.090|
|2|1|32|1|2.438/2.255|1.081|1.003|1.151|
|2|1|256|1|2.446/2.256|1.084|1.004|1.144|

Overlap efficiency = (isolated_copy + isolated_compute - concurrent_makespan) / min(isolated_copy, isolated_compute); it is intentionally not clamped. Independent CUDA streams do not guarantee a win. Isolated references were measured earlier in E1; DVFS and temperature drift can affect slowdown ratios.

Two-slot lifecycle traces check copy→read and prior-read→reuse dependencies and exact numerical output. The pipeline repeatedly computes the same independent group workload; it is not full model inference. Original FlexGen overlap is the E0 cpu_overlap arm, not a like-for-like component makespan.

Decision: use both the isolated and measured-contention scenarios in E3. Neither a positive component speedup nor an event-ordering check establishes relay end-to-end benefit.

## E3: modeled conditions only

128 scenarios; 128 predict >1% improvement and 0 predict >1% regression. Predicted speedup range: 1.123–1.783.

These counts depend on the chosen grid and are not probabilities. predictions.csv retains losses and ties. example_timeline.json includes resource reservations, explicit network/staging and buffer dependencies.

- Block-stack model, excludes input/output embedding and LM head compute/transfer; not total serving latency.
- Remote H2D equal to local; remote compute scaled synthetically. Wire/staging parameters are assumptions.
- Every group uses measurements of the first n actual blocks; layer-to-layer variation is not calibrated.
- Prefill q>1 has past=0; these are short-prompt proxies, not chunked prefill.
- Two decode cycles hold KV length fixed; no real growing-cache workload.
- FCFS list schedule, no backfilling; queue results are synthetic and not online SLO goodput.
- No multi-request weight reuse. This can materially overstate relay advantage versus a batched baseline.
- One half-duplex wire. GPU staging competes with weight H2D; host DRAM/NIC/compute contention only bounded by constant penalty scenarios.
- Memory estimate reserves full-stack KV on each node conservatively; 512MiB workspace reserve is not a proof of fitting a larger model.
- No retained GPU weight optimization even when model fits; these are forced-offload conditions.

Decision: a second machine and real network measurements are necessary. Do not claim relay or SLO benefit from this model. Validate a predicted favorable and unfavorable shape using identical baselines and weight-reuse policies.

## Next minimum changes

1. Repeat representative configurations with longer sampling and randomized order; keep both positive and negative shapes.
2. Obtain approval before downloading OPT-1.3B; this still fits 16GB and is only a larger mechanism test.
3. Implement and numerically validate incremental prefill before describing a q sweep as chunked prefill.
4. Measure a real second endpoint and calibrate staging, H2D contention, weight reuse and queue scheduling before making a system claim.

Memory: GPU allocated/reserved peaks are allocator measurements. RSS is sampled every 10ms. Explicit pinned tensor bytes are tracked; absent host allocator statistics are null, not zero. PyTorch pinned caching can retain more physical RAM than the live tensors. No deliberate OOM stress was performed; rejected/failing cases are written to failures.jsonl when present.
