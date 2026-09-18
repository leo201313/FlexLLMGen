# InferRelay precheck: single-machine results

Measured E0–E2; E3 is a parameterized prediction. OPT-125M forced offload is a mechanism test, not evidence of large-model necessity or two-machine speedup.

## E0: original generation loops

|case|wall P50/P95 ms|prefill P50 ms|decode P50 ms|GPU peak MiB|
|---|---:|---:|---:|---:|
|gpu_p32_b1_m1|41.945/43.129|2.666|39.050|333.0|
|cpu_serial_p32_b1_m1|432.519/432.973|27.121|405.162|87.5|
|cpu_overlap_p32_b1_m1|418.630/419.060|26.145|386.062|164.5|
|gpu_p32_b1_m2|84.272/84.485|5.240|78.649|334.9|
|cpu_serial_p32_b1_m2|454.505/454.730|28.582|425.614|89.2|
|cpu_overlap_p32_b1_m2|461.468/462.472|28.866|426.101|166.2|
|gpu_p32_b4_m1|44.076/44.459|3.186|40.608|347.6|
|cpu_serial_p32_b4_m1|431.615/432.117|28.051|403.329|101.8|
|cpu_overlap_p32_b4_m1|418.554/419.480|26.149|385.986|178.8|
|gpu_p32_b4_m2|86.902/87.705|6.362|80.130|354.4|
|cpu_serial_p32_b4_m2|451.460/451.713|30.296|420.789|108.6|
|cpu_overlap_p32_b4_m2|462.284/463.046|29.392|426.313|185.6|
|gpu_p128_b1_m1|42.709/42.951|3.066|39.401|346.0|
|cpu_serial_p128_b1_m1|433.082/433.712|27.788|405.082|100.2|
|cpu_overlap_p128_b1_m1|418.388/418.632|26.137|385.743|177.2|
|gpu_p128_b1_m2|84.859/84.977|6.134|78.352|351.2|
|cpu_serial_p128_b1_m2|455.951/456.047|29.943|425.638|105.5|
|cpu_overlap_p128_b1_m2|461.526/462.044|29.307|425.615|182.5|
|gpu_p128_b4_m1|46.460/46.706|5.957|40.245|399.3|
|cpu_serial_p128_b4_m1|436.217/436.555|31.307|404.698|153.6|
|cpu_overlap_p128_b4_m1|418.681/419.704|26.180|386.061|230.6|
|gpu_p128_b4_m2|92.117/92.537|11.831|79.847|420.2|
|cpu_serial_p128_b4_m2|459.691/462.058|36.735|422.580|174.5|
|cpu_overlap_p128_b4_m2|465.115/466.988|32.085|426.460|251.5|
|gpu_p256_b1_m1|43.330/43.674|3.564|39.512|362.9|
|cpu_serial_p256_b1_m1|434.394/434.634|28.613|405.491|117.2|
|cpu_overlap_p256_b1_m1|418.478/418.780|26.149|385.876|194.2|
|gpu_p256_b1_m2|86.511/86.756|7.049|79.088|372.8|
|cpu_serial_p256_b1_m2|457.836/458.033|31.530|425.973|127.1|
|cpu_overlap_p256_b1_m2|462.396/463.035|29.749|426.095|204.1|
|gpu_p256_b4_m1|49.872/50.213|9.494|40.107|467.2|
|cpu_serial_p256_b4_m1|440.810/440.955|35.016|405.575|221.6|
|cpu_overlap_p256_b4_m1|418.490/418.880|26.181|385.870|298.6|
|gpu_p256_b4_m2|98.886/99.703|19.102|79.377|508.7|
|cpu_serial_p256_b4_m2|469.573/470.403|44.191|425.002|263.1|
|cpu_overlap_p256_b4_m2|468.829/469.786|35.838|426.432|340.1|

All successful modes require exact generated-ID equality with the all-GPU baseline. Phase times are CUDA-event spans, including host launch gaps; wall time includes generation setup/cleanup. Fixed-length throughput is not online goodput.

Trace CSV/JSON contains weight submission spans, layer compute and original CPU global-sync waits. Weight-ready gaps are a diagnostic proxy, not causal H2D wait attribution. Trace runs are separate from the baseline samples.

- gpu_p128_b1_m1: trace wall/baseline P50 = 1.236; weight-ready gap proxy = 0.000 ms. Single diagnostic sample; do not correct throughput by this ratio.
- cpu_serial_p128_b1_m1: trace wall/baseline P50 = 1.005; weight-ready gap proxy = 406.472 ms. Single diagnostic sample; do not correct throughput by this ratio.
- cpu_overlap_p128_b1_m1: trace wall/baseline P50 = 1.007; weight-ready gap proxy = 337.031 ms. Single diagnostic sample; do not correct throughput by this ratio.

Decision: the baseline and timing path are executable. Inspect the full CPU serial/overlap table before claiming remaining prefetch opportunity; do not extrapolate the small-model ratio to a memory-constrained model.

## E1: actual block-layout transfer and computation

|blocks|FP16 weights MiB|pageable P50 ms|pinned P50 ms|pinned GB/s|
|---|---:|---:|---:|---:|
|1|13.52|1.261|1.119|12.67|
|2|27.04|2.651|2.237|12.67|
|4|54.08|5.661|4.473|12.68|
|8|108.15|11.443|8.912|12.72|

Measured H2D/compute ratios span 1.868–10.673. All query/batch/history combinations are retained in summary.json; raw.csv includes every sample.

Decision: retain near-matching shapes for E2 and the network sweep. q>1 is ordinary short-prompt prefill with past=0, not incremental/chunked prefill. Decode uses real attention KV prepared from synthetic hidden activations through actual pretrained blocks; it is a kernel workload, not full-model semantic validation.

## E2: measured concurrency

56/56 shapes have >1% speedup versus serial; 0/56 regress by >1%. These thresholds summarize measurements, not statistical significance.

|blocks|query|past|batch|serial/concurrent P50 ms|speedup|copy slowdown|compute slowdown|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|16|0|1|1.225/1.141|1.073|1.005|1.065|
|1|16|0|4|1.249/1.144|1.092|1.006|1.080|
|1|32|0|1|1.230/1.142|1.076|1.006|1.069|
|1|32|0|4|1.296/1.143|1.134|1.006|1.117|
|1|64|0|1|1.241/1.144|1.085|1.005|1.071|
|1|64|0|4|1.329/1.142|1.164|1.006|1.082|
|1|128|0|1|1.283/1.139|1.126|1.006|1.059|
|1|128|0|4|1.508/1.147|1.315|1.009|1.057|
|1|256|0|1|1.336/1.144|1.168|1.006|1.084|
|1|256|0|4|1.742/1.144|1.523|1.009|1.034|
|1|1|32|1|1.234/1.142|1.081|1.004|1.166|
|1|1|32|4|1.210/1.140|1.061|1.005|1.074|
|1|1|256|1|1.236/1.141|1.083|1.004|1.168|
|1|1|256|4|1.216/1.140|1.066|1.004|1.072|
|2|16|0|1|2.436/2.263|1.076|1.005|1.071|
|2|16|0|4|2.481/2.268|1.094|1.006|1.064|
|2|32|0|1|2.439/2.266|1.077|1.005|1.048|
|2|32|0|4|2.572/2.267|1.134|1.006|1.107|
|2|64|0|1|2.464/2.264|1.088|1.005|1.069|
|2|64|0|4|2.645/2.300|1.150|1.021|1.091|
|2|128|0|1|2.550/2.268|1.125|1.006|1.121|
|2|128|0|4|3.001/2.272|1.321|1.008|1.043|
|2|256|0|1|2.660/2.268|1.173|1.007|1.098|
|2|256|0|4|3.458/2.270|1.524|1.008|1.021|
|2|1|32|1|2.443/2.260|1.081|1.004|1.113|
|2|1|32|4|2.407/2.261|1.064|1.004|1.052|
|2|1|256|1|2.450/2.261|1.084|1.004|1.104|
|2|1|256|4|2.438/2.263|1.077|1.005|1.059|
|4|16|0|1|4.895/4.514|1.085|1.005|1.069|
|4|16|0|4|4.928/4.515|1.091|1.006|1.033|
|4|32|0|1|4.889/4.514|1.083|1.006|1.041|
|4|32|0|4|5.109/4.522|1.130|1.007|1.082|
|4|64|0|1|4.952/4.515|1.097|1.005|1.085|
|4|64|0|4|5.286/4.522|1.169|1.007|1.068|
|4|128|0|1|5.092/4.520|1.126|1.007|1.131|
|4|128|0|4|5.970/4.522|1.320|1.007|1.031|
|4|256|0|1|5.312/4.519|1.175|1.006|1.088|
|4|256|0|4|6.890/4.521|1.524|1.007|1.010|
|4|1|32|1|4.872/4.509|1.081|1.004|1.075|
|4|1|32|4|4.877/4.512|1.081|1.005|1.058|
|4|1|256|1|4.880/4.507|1.083|1.004|1.066|
|4|1|256|4|4.917/4.512|1.090|1.005|1.103|
|8|16|0|1|9.731/8.976|1.084|1.005|1.013|
|8|16|0|4|9.909/8.983|1.103|1.006|1.050|
|8|32|0|1|9.770/8.980|1.088|1.006|1.028|
|8|32|0|4|10.232/8.987|1.139|1.007|1.130|
|8|64|0|1|9.846/8.980|1.096|1.006|1.072|
|8|64|0|4|10.457/8.987|1.164|1.007|1.022|
|8|128|0|1|10.094/8.980|1.124|1.006|1.083|
|8|128|0|4|11.888/8.988|1.323|1.006|1.019|
|8|256|0|1|10.570/8.980|1.177|1.006|1.072|
|8|256|0|4|13.718/8.989|1.526|1.007|1.004|
|8|1|32|1|9.772/8.979|1.088|1.005|1.065|
|8|1|32|4|9.696/8.974|1.080|1.005|1.019|
|8|1|256|1|9.786/8.972|1.091|1.005|1.047|
|8|1|256|4|9.781/8.974|1.090|1.005|1.068|

Overlap efficiency = (isolated_copy + isolated_compute - concurrent_makespan) / min(isolated_copy, isolated_compute); it is intentionally not clamped. Independent CUDA streams do not guarantee a win. Isolated references were measured earlier in E1; DVFS and temperature drift can affect slowdown ratios.

Two-slot lifecycle traces check copy→read and prior-read→reuse dependencies and exact numerical output. The pipeline repeatedly computes the same independent group workload; it is not full model inference. Original FlexGen overlap is the E0 cpu_overlap arm, not a like-for-like component makespan.

Decision: use both the isolated and measured-contention scenarios in E3. Neither a positive component speedup nor an event-ordering check establishes relay end-to-end benefit.

## E3: modeled conditions only

26880 scenarios; 15848 predict >1% improvement and 3832 predict >1% regression. Predicted speedup range: 0.110–1.794.

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
