# InferRelay precheck: single-machine results

Measured E0–E2; E3 is a parameterized prediction. OPT-125M forced offload is a mechanism test, not evidence of large-model necessity or two-machine speedup.

## E0: original generation loops

|case|wall P50/P95 ms|prefill P50 ms|decode P50 ms|GPU peak MiB|
|---|---:|---:|---:|---:|
|gpu_p32_b1_m1|42.508/43.668|2.697|39.562|333.0|
|cpu_serial_p32_b1_m1|432.805/433.365|27.185|405.291|87.5|
|cpu_overlap_p32_b1_m1|418.648/419.253|26.144|386.036|164.5|
|gpu_p32_b1_m2|84.907/85.203|5.242|79.281|334.9|
|cpu_serial_p32_b1_m2|454.242/454.599|28.540|425.252|89.2|
|cpu_overlap_p32_b1_m2|462.232/463.254|28.902|426.722|166.2|
|gpu_p32_b4_m1|44.307/44.616|3.198|40.867|347.6|
|cpu_serial_p32_b4_m1|431.205/431.781|27.902|403.077|101.8|
|cpu_overlap_p32_b4_m1|418.514/419.588|26.162|385.900|178.8|
|gpu_p32_b4_m2|87.672/88.925|6.450|80.668|354.4|
|cpu_serial_p32_b4_m2|451.398/451.765|30.319|420.759|108.6|
|cpu_overlap_p32_b4_m2|463.852/464.744|29.495|427.657|185.6|
|gpu_p128_b1_m1|43.155/43.357|3.092|39.818|346.0|
|cpu_serial_p128_b1_m1|433.355/433.498|27.875|405.215|100.2|
|cpu_overlap_p128_b1_m1|418.528/419.326|26.157|385.923|177.2|
|gpu_p128_b1_m2|84.975/85.179|6.148|78.481|351.2|
|cpu_serial_p128_b1_m2|455.510/456.650|29.862|425.331|105.5|
|cpu_overlap_p128_b1_m2|462.075/463.364|29.337|426.180|182.5|
|gpu_p128_b4_m1|46.986/47.519|5.958|40.758|399.3|
|cpu_serial_p128_b4_m1|436.289/436.815|31.305|404.779|153.6|
|cpu_overlap_p128_b4_m1|419.217/419.821|26.205|386.524|230.6|
|gpu_p128_b4_m2|93.401/93.758|11.838|81.048|420.2|
|cpu_serial_p128_b4_m2|460.927/461.608|36.772|423.762|174.5|
|cpu_overlap_p128_b4_m2|466.108/467.251|32.099|427.452|251.5|
|gpu_p256_b1_m1|43.979/44.340|3.590|40.094|362.9|
|cpu_serial_p256_b1_m1|434.224/434.274|28.680|405.299|117.2|
|cpu_overlap_p256_b1_m1|418.525/418.894|26.161|385.918|194.2|
|gpu_p256_b1_m2|87.000/87.351|7.121|79.524|372.8|
|cpu_serial_p256_b1_m2|457.541/457.883|31.463|425.666|127.1|
|cpu_overlap_p256_b1_m2|466.385/472.313|29.824|429.934|204.1|
|gpu_p256_b4_m1|51.712/51.969|9.541|41.905|467.2|
|cpu_serial_p256_b4_m1|440.486/441.111|34.992|405.203|221.6|
|cpu_overlap_p256_b4_m1|419.101/419.692|26.190|386.444|298.6|
|gpu_p256_b4_m2|102.627/102.880|19.179|82.965|508.7|
|cpu_serial_p256_b4_m2|470.261/471.049|44.285|425.564|263.1|
|cpu_overlap_p256_b4_m2|471.711/473.633|35.932|429.253|340.1|

All successful modes require exact generated-ID equality with the all-GPU baseline. Phase times are CUDA-event spans, including host launch gaps; wall time includes generation setup/cleanup. Fixed-length throughput is not online goodput.

Trace CSV/JSON contains weight submission spans, layer compute and original CPU global-sync waits. Weight-ready gaps are a diagnostic proxy, not causal H2D wait attribution. Trace runs are separate from the baseline samples.

- gpu_p128_b1_m1: trace wall/baseline P50 = 1.238; weight-ready gap proxy = 0.000 ms. Single diagnostic sample; do not correct throughput by this ratio.
- cpu_serial_p128_b1_m1: trace wall/baseline P50 = 1.004; weight-ready gap proxy = 405.887 ms. Single diagnostic sample; do not correct throughput by this ratio.
- cpu_overlap_p128_b1_m1: trace wall/baseline P50 = 1.008; weight-ready gap proxy = 336.860 ms. Single diagnostic sample; do not correct throughput by this ratio.

Decision: the baseline and timing path are executable. Inspect the full CPU serial/overlap table before claiming remaining prefetch opportunity; do not extrapolate the small-model ratio to a memory-constrained model.

## E1: actual block-layout transfer and computation

|blocks|FP16 weights MiB|pageable P50 ms|pinned P50 ms|pinned GB/s|
|---|---:|---:|---:|---:|
|1|13.52|1.239|1.119|12.67|
|2|27.04|2.648|2.237|12.68|
|4|54.08|5.601|4.470|12.69|
|8|108.15|11.476|8.915|12.72|

Measured H2D/compute ratios span 1.851–10.392. All query/batch/history combinations are retained in summary.json; raw.csv includes every sample.

Decision: retain near-matching shapes for E2 and the network sweep. q>1 is ordinary short-prompt prefill with past=0, not incremental/chunked prefill. Decode uses real attention KV prepared from synthetic hidden activations through actual pretrained blocks; it is a kernel workload, not full-model semantic validation.

## E2: measured concurrency

56/56 shapes have >1% speedup versus serial; 0/56 regress by >1%. These thresholds summarize measurements, not statistical significance.

|blocks|query|past|batch|serial/concurrent P50 ms|speedup|copy slowdown|compute slowdown|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|16|0|1|1.226/1.141|1.074|1.006|1.022|
|1|16|0|4|1.250/1.140|1.097|1.007|1.052|
|1|32|0|1|1.230/1.142|1.077|1.006|1.012|
|1|32|0|4|1.297/1.143|1.135|1.007|1.094|
|1|64|0|1|1.241/1.138|1.090|1.005|1.023|
|1|64|0|4|1.330/1.143|1.164|1.006|1.075|
|1|128|0|1|1.283/1.142|1.124|1.006|1.088|
|1|128|0|4|1.508/1.144|1.318|1.009|1.048|
|1|256|0|1|1.336/1.144|1.168|1.006|1.058|
|1|256|0|4|1.741/1.143|1.523|1.010|1.026|
|1|1|32|1|1.233/1.141|1.080|1.005|1.099|
|1|1|32|4|1.209/1.140|1.061|1.005|1.003|
|1|1|256|1|1.235/1.142|1.082|1.004|1.158|
|1|1|256|4|1.216/1.141|1.065|1.005|1.031|
|2|16|0|1|2.437/2.267|1.075|1.005|1.025|
|2|16|0|4|2.483/2.268|1.095|1.006|1.033|
|2|32|0|1|2.440/2.266|1.077|1.005|1.005|
|2|32|0|4|2.573/2.266|1.135|1.006|1.071|
|2|64|0|1|2.466/2.265|1.089|1.006|1.041|
|2|64|0|4|2.649/2.272|1.166|1.008|1.086|
|2|128|0|1|2.535/2.266|1.119|1.006|1.090|
|2|128|0|4|2.998/2.272|1.320|1.008|1.042|
|2|256|0|1|2.680/2.271|1.180|1.008|1.097|
|2|256|0|4|3.458/2.273|1.521|1.009|1.015|
|2|1|32|1|2.444/2.262|1.081|1.004|1.074|
|2|1|32|4|2.405/2.260|1.064|1.004|1.003|
|2|1|256|1|2.451/2.261|1.084|1.004|1.068|
|2|1|256|4|2.437/2.262|1.077|1.005|1.004|
|4|16|0|1|4.896/4.513|1.085|1.006|1.055|
|4|16|0|4|4.983/4.514|1.104|1.006|1.073|
|4|32|0|1|4.914/4.514|1.088|1.006|1.061|
|4|32|0|4|5.141/4.519|1.138|1.007|1.134|
|4|64|0|1|4.952/4.512|1.098|1.006|1.075|
|4|64|0|4|5.288/4.518|1.170|1.007|1.068|
|4|128|0|1|5.090/4.519|1.126|1.007|1.136|
|4|128|0|4|5.972/4.521|1.321|1.007|1.031|
|4|256|0|1|5.310/4.516|1.176|1.007|1.084|
|4|256|0|4|6.888/4.519|1.524|1.008|1.006|
|4|1|32|1|4.866/4.505|1.080|1.004|1.045|
|4|1|32|4|4.876/4.511|1.081|1.005|1.047|
|4|1|256|1|4.879/4.507|1.083|1.005|1.058|
|4|1|256|4|4.918/4.511|1.090|1.005|1.085|
|8|16|0|1|9.728/8.976|1.084|1.005|1.016|
|8|16|0|4|9.907/8.988|1.102|1.006|1.046|
|8|32|0|1|9.777/8.979|1.089|1.005|1.013|
|8|32|0|4|10.227/8.987|1.138|1.006|1.134|
|8|64|0|1|9.840/8.974|1.096|1.005|1.041|
|8|64|0|4|10.520/8.985|1.171|1.006|1.053|
|8|128|0|1|10.132/8.983|1.128|1.006|1.131|
|8|128|0|4|11.886/8.983|1.323|1.006|1.017|
|8|256|0|1|10.563/8.979|1.176|1.005|1.076|
|8|256|0|4|13.717/8.991|1.526|1.007|1.005|
|8|1|32|1|9.767/8.969|1.089|1.004|1.039|
|8|1|32|4|9.693/8.972|1.080|1.005|1.004|
|8|1|256|1|9.786/8.972|1.091|1.004|1.040|
|8|1|256|4|9.779/8.974|1.090|1.005|1.053|

Overlap efficiency = (isolated_copy + isolated_compute - concurrent_makespan) / min(isolated_copy, isolated_compute); it is intentionally not clamped. Independent CUDA streams do not guarantee a win. Isolated references were measured earlier in E1; DVFS and temperature drift can affect slowdown ratios.

Two-slot lifecycle traces check copy→read and prior-read→reuse dependencies and exact numerical output. The pipeline repeatedly computes the same independent group workload; it is not full model inference. Original FlexGen overlap is the E0 cpu_overlap arm, not a like-for-like component makespan.

Decision: use both the isolated and measured-contention scenarios in E3. Neither a positive component speedup nor an event-ordering check establishes relay end-to-end benefit.

## E3: modeled conditions only

26880 scenarios; 15830 predict >1% improvement and 3850 predict >1% regression. Predicted speedup range: 0.110–1.794.

These counts depend on the chosen grid and are not probabilities. predictions.csv retains losses and ties. example_timeline.json includes resource reservations, explicit network/staging and buffer dependencies.

- Block-stack model, excludes input/output embedding and LM head compute/transfer; not total serving latency.
- Remote H2D equal to local; remote compute scaled synthetically. Wire/staging parameters are assumptions.
- Every group uses measurements of the first n actual blocks; layer-to-layer variation is not calibrated.
- Prefill q>1 has past=0; these are short-prompt proxies, not chunked prefill.
- Two decode cycles hold KV length fixed; no real growing-cache workload.
- FCFS per-node compute order with resource-gap backfilling for prefetch; queue results are synthetic and not online SLO goodput.
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
