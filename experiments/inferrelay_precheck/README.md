# InferRelay 前期验证：离线单机最小版本

本目录独立于 `flexllmgen/` 和原始 benchmark。E0–E2 是本机实测，E3 是参数化成本模型；E4/E5 仅提供后续方案。所有正负结果都保留，不把单机结果称为两机收益或在线 SLO goodput。

首轮已完成：先看 [中文结论与阶段决策](FINDINGS.md)，完整数据采用 [smoke-02](results/smoke-02/REPORT.md)，可直接查看 [交互时间线](results/smoke-02/timeline.html)。

## 当前硬件与范围

初始检查：RTX 5060 Ti 16GB、i7-14700KF、32GiB RAM（约 28GiB available）、67GiB 可用磁盘；仓库 commit `004ffef82b46e8dc8685c55d0cdda650bdaf1269`。驱动 580.95.05，Python 3.10、PyTorch 2.7.1+cu128、Transformers 4.30.2。每次运行另行保存实际 inventory、源码副本和 SHA256。

空闲时 `nvidia-smi` 返回 PCIe Gen1/x4；不得据此认定负载带宽。E1 每项传输后读取 sysfs 链路状态，同时报告实测有效带宽。网卡枚举有 RTL8125 2.5GbE，但枚举不等于实测协商速率或端到端有效带宽。不存在第二台设备的测量。

默认仅使用已经缓存的 OPT-125M，FP16、dense attention、GPU KV/activation，无量化、磁盘 offload、CPU attention、自定义 CUDA kernel。未缓存的模型会直接报错，不自动下载。

## 运行

从仓库根目录执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmexp
python -m experiments.inferrelay_precheck.run
```

默认完成 E0–E3。GPU 操作需在可访问显卡的终端运行。可指定输出位置、重复次数和每项最小采样秒数：

```bash
python -m experiments.inferrelay_precheck.run \
  --config experiments/inferrelay_precheck/configs/smoke.json \
  --out experiments/inferrelay_precheck/results/my-run \
  --repeats 20 --min-seconds 1
```

输出目录必须不存在，防止覆盖结果。可用 `--suite e0` 或 `--suite e1 e2 e3`；E2/E3 必须使用同次 E1/E2 数据。`warmup`、所有扫描范围、显存和 RAM 预算均在 JSON 配置中。默认预热 2 次、至少 7 次采样且每项至少 0.15 秒；这些是短时预检查，P95/P99 为样本分位数，尤其 E0 的少量重复不能给出稳定尾延迟结论。

运行结束后可直接用浏览器打开输出目录中的 `timeline.html`，切换实测 E0 / 预测 E3 时间线、选择 prefill 或某个 decode token、悬停查看精确区间。页面自包含，不需要联网。已有结果可用 `python -m experiments.inferrelay_precheck.visualize <结果目录>` 生成此页面。

```bash
python -m unittest discover -s experiments/inferrelay_precheck/tests -v
```

## 实验配置与预计资源

|实验|默认扫描|主要输出|
|---|---|---|
|E0|GPU resident / CPU serial / 原版 CPU overlap；prompt 32/128/256；microbatch 1/4；microbatch 数 1/2；生成 16 tokens|36 组原始采样、输出 ID 一致性、三条独立诊断 trace|
|E1|完整 block 1/2/4/8；pageable/pinned；实际 16 张量/block 与连续 slab；prefill q=16/32/64/128/256,past=0；decode q=1,past=32/256；batch 1/4|传输与计算 P50/P95/P99、准备 pinned 的独立耗时、H2D/compute 比值|
|E2|与 E1 同一形状；串行与独立双流；代表形状再执行 4 步双缓冲|makespan、copy/compute slowdown、重叠效率、data-ready/reuse 等待、生命周期检查|
|E3|1/2.5/10/25/100 Gbps；单向 25/100/500µs；1/2 buffer；1/4 请求；0/5ms 到达间隔；远端计算倍率 1/2|连续切层与交替切层的依赖时间线和预计盈亏范围|

125M FP16 每 block 约 13.5MiB，8-block 权重约 108MiB。E0 的模型含独立输入/输出 embedding 副本，不能只按参数数目估计峰值；预留 logits、KV、workspace 和 allocator。最小配置预计 GPU < 2GiB、进程 RAM < 3GiB；硬预算配置为 GPU 4GiB、RAM 6GiB。GPU 使用 PyTorch allocator fraction 限制，RAM 使用分配前估算和实测采样（不是 OS 硬限制）。逐配置记录 GPU allocated/reserved peak、10ms RSS 采样峰值和明确持有的 pinned tensor 字节；不把缺失的 pinned allocator 统计当作零。不主动测试 OOM；发生 OOM/预算拒绝保留失败记录。

后续候选 OPT-1.3B：FP16 原始权重约 2.6GB，完整模型加 KV/临时空间预计需要约 4–6GiB GPU；下载原始文件与转换权重合计至少约 5.2GB，另留缓存和临时空间。此估计需按具体 batch/prompt 复核。它仍能放入 16GB，不代表 offload 必要性；本轮不下载。6.7B 等模型需重新核对 RAM/显存/磁盘并获得下载确认。

## 计时、公平性与已知限制

- E0 直接调用原版 `OptLM.generate`、原版策略和原版 overlap 循环。`sep_layer=True`，一层 attention/MLP 分开调度，与原始默认一致。三种模式必须生成完全相同的 token IDs。
- E0 同时保留原版 `get_test_inputs`：`Paris is the capital city of` 共 7 个非 padding token，左补齐到 32/128/256。因此这里扫描的是 **padding 后的物理序列长度**，不是不同语义长度的真实请求；dense attention 仍处理这些物理形状。后续真实负载实验必须使用无 padding 的不同长度 prompt，不能把本表解释成真实长 prompt 服务结果。
- 禁用未使用的磁盘后端：原版 `TorchDisk` 会启动 4 个 copy worker，每个申请 `1 Gi` 个 FP16 元素（2GiB）pinned 缓冲。替代后端只允许空同步，拒绝磁盘分配。此更改仅在实验入口生效，原仓库未修改；因此不把原始 CLI 的进程 RAM 与本实验直接比较。
- E0 原版 overlap 仍保留每层/每 microbatch 的全局同步；不为“好看”的并发数据改写它。正式采样只添加逐 token CUDA 标记，端到端用墙钟。原始 host timer 不作为 CUDA 阶段的唯一计时依据。
- E0 详细 trace 单独运行，用 CUDA event 记录层计算和权重提交区间，同时记录原版 CPU sync。无额外逐层同步。权重 event 区间可能包含 allocator/host launch 空隙；`weight_ready_gap_proxy_ms` 不是因果归因后的纯 H2D 等待。`causal_weight_wait_ms=null`，后续可用 Nsight/CUPTI 精确分离。trace 吞吐扰动单独保存。
- Trace JSON 可导入 Perfetto 或 Chrome tracing。CPU 与 CUDA 的起点通过一次 origin 同步近似对齐；不用于亚微秒跨时钟结论。
- E1/E2 使用前 n 个完整 block 的真实 FP16 权重，计算直接复用 FlexGen `mha/mha_gen/mlp`。隐藏输入是固定种子合成张量。decode 的历史 KV 由同组真实 prefill 生成，重复采样固定 past 并覆盖当前位置；不使用未初始化的历史 KV。
- E1 的每个计算形状单独记录 GPU/RSS 峰值，包含预备历史 KV 的过程；组级峰值取搬运及所有计算形状的最大值。PyTorch host allocator 原始计数完整保留；若出现负计数则不采用其 allocated peak，明确记录该异常。reserved peak 是本进程生命周期内的报告值，不是逐配置重置的峰值；live pinned 字节来自显式张量统计。
- q>1 的 prefill **仅是短 prompt 近似**，没有历史 KV；本版本不实现 incremental prefill。E1 中 pinned 准备（分配+CPU copy）与复用 pinned source 的 H2D 分开。首次准备不保证 allocator/OS 冷态。slab 测试为不同布局的参考上限，不据此替代实际张量传输时间。
- E2 copy 与 compute 完全不共享权重目的 buffer。所有流有起点依赖和终点 join，仅在样本边界等待。双缓冲通过 ready/done 事件管理复用，检查数值和事件依赖。常驻参考权重等诊断开销包含在实际峰值中；双缓冲字节另列，不声称这是部署时完整显存需求。
- E2 的原版 overlap 对照来自 E0：整模型结果与层组微基准分别报告，不能直接把它们的绝对 ms 放在同一个速度比中。E1 isolated 与 E2 concurrent 按固定顺序测量，尚未随机交错，应在扩展实验中控制 DVFS/热漂移。
- 不做权重 D2H 写回，CPU 中有只读副本。KV 保持在 GPU。E3 activation staging 的 D2H 与权重搬运明确分开。
- E3 按 copy-ready、activation-ready、slot-release 依赖保留具体资源区间，显式建模源 D2H、网络、目的 H2D、token 返回和请求排队；连续切层扫描所有完整层组边界以适应异构计算倍率。参数是假设，非网络实测。
- E3 为 **block-stack** 模型，尚不含 embedding/LM-head；使用同形状前 n 层测量代表其他层组。decode 两轮固定 KV 长度；prefill 只走一轮。FCFS 计算顺序下允许独立预取回填资源空隙，没有多请求权重复用或驻留优化。有限 buffer 会约束预取，但合成排队不是 SLO 结果。网络带宽用 Gbps，端点 staging 带宽用 GB/s（默认 8GB/s），不是同一单位。
- E3 使用 isolated 与持续争用倍率两种预测，不假定所有阶段可重叠；DRAM/NIC 等争用仍需两机校准。显存预算包含保守 KV 和 512MiB workspace 预留，不能保证更大模型可用。
- 所有结论限于机制与必要条件。未进行文献查新，不作新颖性声明。

## 目录与原始数据

```text
experiments/inferrelay_precheck/
  README.md                 # 运行与解释
  NEXT_STAGES.md            # E4/E5 后续计划
  configs/{smoke,pilot}.json
  run.py common.py          # 入口、资源记录、采样
  e0.py components.py e3.py # E0 / E1+E2 / 事件模型
  report.py tests/
  results/<run>/
    environment.json config.json source/ source_sha256.json
    e0/raw.{csv,json} summary.json trace_*.{csv,json}
    e1/raw.csv summary.json pin_prepare_*.json
    e2/raw.csv summary.json buffer_*.csv
    e3/predictions.csv summary.json example_timeline.json
    REPORT.md completion.json
```

部分运行会只包含选定阶段。失败会写到对应阶段的 `failures.jsonl`，非资源类异常停止运行，不静默跳过正确性问题。`source/` 保留运行时 Python 文件；实际 CLI 和配置以环境/配置记录为准。每阶段是否支持假设、限制与最小下一步，见生成的 REPORT 和人工复核的结论说明。
