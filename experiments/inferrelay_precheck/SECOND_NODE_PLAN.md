# 接入第二台机器：配置清单与实验扩展

## 已确认与待确认

2026-09-19 已完成 B 的基础配置与验收。A=`192.168.20.103` (`DAOs-5060Ti-3`)，B=`192.168.20.105` (`DAOs-5060Ti-5`)，用户均为 `leocao`。两端各自生成专用 Ed25519 密钥，仅交换公钥；A 可执行 `ssh inferrelay-b`，B 可执行 `ssh inferrelay-a`。私钥位于各自 `~/.ssh/inferrelay_ed25519`，未复制进仓库。

|项目|A|B|
|---|---|---|
|GPU|RTX 5060 Ti 16GB|RTX 5060 Ti 16GB|
|CPU / RAM|i7-14700KF / 32GB|i7-14700KF / 32GB|
|驱动|580.95.05|580.95.05|
|传输中 PCIe|Gen5 ×4|Gen5 ×8|
|OPT-125M 实际张量 pinned H2D|12.7GB/s|23.6GB/s|
|NIC|enp4s0，2.5Gbps full，MTU 1500|enp4s0，2.5Gbps full，MTU 1500|
|B 可用磁盘|—|安装前 182GiB，安装后约 171GiB|

B 新装 Miniconda，通过内网复制 A 已验证的 `llmexp` 到相同绝对路径。两端系统架构、用户名路径、GPU 和驱动匹配；此复制方式不适用于任意异构机器。环境为 Python 3.10、PyTorch 2.7.1+cu128、Transformers 4.30.2、NumPy 1.26.4；B `pip check` 和 GPU FP16 矩阵乘法均通过。仓库位于 `/home/leocao/InferRelay/FlexLLMGen`，commit `42c7919`，原版推理代码未修改。没有复制 GitHub 凭据。

已同步 `~/opt_weights/opt-125m-np`、`opt-1.3b-np` 及原版入口所用的 OPT-30B **分词器文件**。B 离线验收：125M（batch1/prompt32/gen16）推理约 0.065s、峰值显存 0.325GiB；1.3B 默认配置（batch4/prompt512/gen32）约 0.690s、峰值显存 3.248GiB、decode 237.231 token/s。这些是单次验收数据，不作稳定性能对比。

B 的原有 E1/E2 全网格完成，结果见 [node-b-smoke-01](results/node-b-smoke-01/REPORT.md)，用时 33.54s；与 A 的 `smoke-02` 同配置。1/2/4/8 层 pinned H2D 中位数分别为 0.601/1.201/2.401/4.799ms。此轮用于环境和机制验收，正式性能结论还需两端同轮复测。

10 秒单流 iperf3 接收端带宽：A→B **2.344Gbps**、B→A **2.353Gbps**；10 次 ICMP RTT 平均 0.236ms，无丢包。它们不等于 tensor staging 时延或小消息延迟。原始网络 JSON 与原版生成日志见 [node-b-setup-01](results/node-b-setup-01/README.md)。测试服务已结束。首次网络测试因服务超时中断，已重测，以保存的成功测试为准。

手动在 A 登录 B 并复验：

```bash
ssh inferrelay-b
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmexp
cd ~/InferRelay/FlexLLMGen
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m flexllmgen.flex_opt --model facebook/opt-1.3b --no-log
```

## 已执行的配置流程（供复现）

1. 只读 inventory：GPU/显存/占用/compute capability、CPU、RAM、磁盘、驱动、PCIe、NIC/MTU、系统及已有 Conda；检查远端 AGENTS.md。优先使用独立用户环境，不覆盖已有项目或修改系统驱动。
2. 创建/复用 `llmexp`，保持 Python 3.10、Transformers 4.30.2、NumPy 1.26.4 等核心版本。PyTorch CUDA 构建必须兼容 B 的 GPU 和驱动；不能盲目复制 A 的 cu128 环境。若两端构建不同，记录差异并做数值对照。
3. 经 SSH 从 A 传输 Git bundle 或项目快照，记录 commit 与源码哈希。不要复制整个用户目录、GitHub 凭据或整个缓存目录。
4. 经内网同步所需模型目录和分词器文件，首先 OPT-125M，再根据 B 空间同步已有 OPT-1.3B。只传所需权重、检查大小和 SHA256，不重新从公网下载已有模型。A 的这两份转换目录约为 314MiB 和 2.7GiB。
5. 两端验证 GPU FP16 运算、依赖检查、原版 FlexGen 离线生成。B 显存预算按实测余量设置；原版磁盘 worker 即使闲置也会申请较多 pinned RAM，RAM 不足时需先报告而非冒险运行。
6. 在 B 跑相同 E1/E2 的短配置，将结果回收至带 `node_id` 的新目录。不得把 A 的时间直接乘一个常数当成 B 实测。

## 现有代码的具体修改点

|文件/新增模块|最小必要修改|验收条件|
|---|---|---|
|`common.py`|inventory 接受 node_id、GPU index、NIC 和 PCI 地址，去掉 PCI 地址写死；区分配置值和实测值|两端输出可对应，缺失字段明确为 unknown|
|`components.py`|`Blocks` 支持 start_layer/end_layer，而非只取前 n 层；保留实际 layer_id；导出每端独立 E1/E2|能测不同真实层组，权重形状和层编号一致|
|`run.py` / `configs/two_node.json`|主机、账号、环境/权重路径、GPU budget 分别配置；默认离线，输出不覆盖|两端同一实验 ID、各自配置与哈希完整|
|新增 `network_bench.py`|TCP CPU staging 基准，按实际 activation 大小扫；再加入 GPU D2H→网络→H2D 路径|接收长度、序号、校验和正确，预热和重复可配置|
|`e3.py`|读取 A/B 实测 copy/compute 和网络消息大小曲线，替换 remote multiplier 和常数 staging；保存预测 vs 实测误差|保留旧合成模型以便回归，不把新预测当两机实测|
|新增 `relay_worker.py`、`relay_driver.py`|先做单请求阻塞四层组原型，再加 ready/done 管理的预取|每层组 hidden、最终 logits、KV 与本地参考对照|

以上是计划，不表示这些扩展已经实现。基础环境与 iperf3 已完成；下一步先实现 activation 大小相关的 CPU/GPU staging 基准，再编写真正的层组执行器。

## 网络实验：先测实际链路，而不是假设 2.5Gbps

初始扫描 4KiB、16KiB、64KiB、256KiB、512KiB、1MiB、2MiB、4MiB，并加入模型实际消息大小。OPT-1.3B 的 hidden_size=2048，FP16 activation 字节数为 `batch × query_tokens × 2048 × 2`：batch=1 的 decode 为 4KiB；batch=4、q=256 为 4MiB。

若端到端恰能达到 2.5Gbps，4MiB 单次纯序列化已约需 13.42ms；四层组交替相对连续切分多两次 activation 传输，单轮仅额外字节传输就约 26.84ms。这是线速下界计算，不含协议、排队、端点 staging，也不是网络实测。

先测 CPU→CPU ping-pong RTT 与端到端接收确认时间，单向延迟不能直接用未同步的两台时钟相减，也不能把 RTT/2 当成已测得的单向延迟。GPU 路径分别用本机 CUDA events 测 D2H/H2D，用同一进程墙钟测往返；明确控制/确认消息的开销。报告消息大小对应 P50/P95/P99、有效吞吐和校验失败。

依次比较：网络独占、与本地 H2D 并发、与层计算并发。基准仅监听指定内网地址，采用有界消息长度、超时和有限运行次数；不为了实验关闭防火墙或开放公网监听。传输使用固定头和原始张量字节，不接收任意 pickle 对象。SSH 用于部署和启动；数据面直连 TCP，不能把 SSH 管道开销默认为目标数据路径。

## 四层组原型的最小设计

125M 用 12 层分为 4×3 层；1.3B 用 24 层分为 4×6 层。接力映射为 A:G1→B:G2→A:G3→B:G4；连续基线为 A:G1,G2→B:G3,G4，并允许按两端实测速度调整切分。

分组接口要拆开权重位置、KV 生命周期和执行状态，不能直接把 E2 的重复独立 group compute 循环视为完整生成。消息头携带 run/request/step/group、query_len、past_len、shape、dtype；KV 按请求和所属层留在本机，CPU 保留只读权重。明确 embedding/LM-head 所属节点和输出 token 返回路径，不能像当前 E3 一样省略它们。

先全 GPU 驻留、单请求阻塞通信，比较单机参考的 hidden/logits/KV，再加入 CPU 权重 offload。预取用至少两个明确的 weight slot：写入等旧 reader 的 done；计算等 copy-ready；CPU 副本无需 GPU 权重 D2H 写回。额外 buffer、两端显存/RSS/pinned 和所有通信字节均需记录。

首轮四个对照：连续分组/关闭预取、连续分组/开启预取、交替分组/关闭预取、交替分组/开启预取。所有对照同样硬件、预算、batch、输入、精度及权重复用策略。由于 B 的 H2D 近乎 A 的两倍，必须加入交换起点（A→B 与 B→A）及按实测时间调优的连续切分基线；否则无法排除 PCIe 异构性带来的收益。先隔离接力和预取，再接入通过数值验收的 chunked prefill；不能把 chunking 收益归给接力。

## 阶段决策

- 125M 用于正确性和协议；1.3B 用于更大的真实层组测量，但 A 已能全 GPU 驻留，仍不能说明 offload 的部署必要性。
- 如果 B 的网络只有 1Gbps，或 staging/通信远大于减少的权重等待，优先保存负结果与修正 E3，不继续堆叠在线调度。
- 如果出现可重复的有利区间，选择一个有利和一个不利配置进行完整两机验证，同时保留单机全 GPU 基线。
- 最后才做 E4 的真实增量 prefill 与在线请求实验；使用不同有效长度的 prompt，而非现有 E0 的 7-token 输入加 padding。SLO 指标、饥饿规则和开放式到达轨迹沿用 `NEXT_STAGES.md` 中的先验方案。
