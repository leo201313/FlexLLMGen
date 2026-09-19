# T1.5、T2、T3：实现与复现

结论见 [FINDINGS_T15_T3.md](FINDINGS_T15_T3.md)。本轮使用现有环境和模型，没有修改原版 FlexGen，没有引入 chunked prefill 或在线调度。

## 文件与结果

|阶段|代码|配置/原始结果/报告|
|---|---|---|
|T1.5|`interference.py`、`report_interference.py`|`results/t15-a-b-01`、`t15-b-a-01`；[汇总](results/t15-summary-01/README.md)|
|T2|`executor.py`、`model_worker.py`、`verify_partitions.py`|`configs/t2*.json`；[基础与预取正确性汇总](results/t2-summary-01/README.md)；`t2-unequal-partitions-01`；`t2-bulk-13b-01`|
|T3|同一 `executor.py`/`model_worker.py`、`report_model.py`|`configs/t3.json`；[正式02轮次](results/t3-summary-02/README.md)；[分节点时间线](results/t3-summary-02/timeline.html)|
|补充|同一执行器|`t3-confirm-125m-a-b-01` 交错复测；`t3-control-a-b-01`、`t3-control-b-a-01` 提交粒度消融|

原始 JSON 保存到每次运行的 A/B 子目录；汇总 CSV 保留所有成功配置。每端保存硬件/软件、Git commit/status、启动参数、配置、源码快照和 SHA256。旧 T0/T1 与本轮诊断轮次保留。

## 一键运行

从 A 的仓库根目录执行，使用从未用过的标签：

```bash
bash experiments/inferrelay_two_node/run_stages.sh repeat03 all
```

也可分阶段运行同一标签：

```bash
bash experiments/inferrelay_two_node/run_stages.sh repeat03 t15
bash experiments/inferrelay_two_node/run_stages.sh repeat03 t2
bash experiments/inferrelay_two_node/run_stages.sh repeat03 t3
```

T3 脚本要求同标签 T2 全部成功，且 executor/model_worker/wire 的 SHA256 与通过门槛时一致。运行顺序125M→1.3B；不同时运行多个GPU实验。启动器使用本地文件锁拒绝并发启动。已有目录不会覆盖。

只复现一个已通过门槛的配置矩阵：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmexp
python -m experiments.inferrelay_two_node.launch \
  --worker model_worker --model opt-1.3b --direction B-A \
  --config experiments/inferrelay_two_node/configs/t3.json \
  --out experiments/inferrelay_two_node/results/manual-13b-b-a
```

单独的 launch 是低层入口，不替你检查T2历史门槛。修改数值执行器后应先运行T2。

## 停止与失败

Ctrl-C，或在另一终端指定当前运行目录：

```bash
~/miniconda3/envs/llmexp/bin/python -m experiments.inferrelay_two_node.stop \
  --run experiments/inferrelay_two_node/results/manual-13b-b-a
```

只终止记录且命令匹配的本轮PID/进程组。socket、连接启动及进程均有限时，默认进程600秒；断网时依靠对端 timeout 最终退出。失败保留日志与部分结果，不继续下一项。不能通过删除旧结果来重试，请换标签。

## 执行器与公平性

- 全局层编号直接读取对应 `.np` 文件；group支持任意真实起止层。除常规四等分，还验证了125M的 `[0,2), [2,5), [5,9), [9,12)` 不等长分组，在resident/demand/prefetch模式都与全模型参考一致。
- 计算调用原FlexGen kernels。输出head复用原版layer_norm/linear公式并暴露logits。KV按层留GPU，网络只传FP16 contiguous activation与token/控制元数据。
- 单机参考、连续、接力共用代码。真实有效prompt长度32/128/256，无padding替代；greedy输出，固定FP16；预先固定atol=0.02/rtol=0.01，token完全一致。T2每步检查所有层KV长度/位置及每层K/V均匀256元素；不声称完整KV逐元素比较。
- 同模型所有offload对照每端均分配两个等字节槽位：125M为85,054,464B，1.3B为1,208,598,528B。每端统一4GiB PyTorch CUDA分配器上限。连续cut25/50/75%改变所属层/KV容量，但不增加槽位预算。
- embedding/head常驻并计入，各端token embedding副本分别计算；block CPU副本只读，不将GPU权重写回。resident基线单列。
- 本轮有意每生成步重新搬运block，**未启用跨步权重驻留复用**。两槽位对均分小模型足以装下本端全部block，此为受控重载机制实验；不能把它称作模型自然装不下。GPU常驻方案实际更快。
- C1/R1同时具备提前预取。priority只在本机activation拷贝期间暂停新权重提交，网络等待/发送期间继续预取；window2最多两片8MiB在途。bulk作为补充消融，每组一个平坦拷贝，不限为8MiB。

## 计时与资源口径

首节点端到端含embedding、完整block、head、activation通信及token返回；初始权重准备、KV预分配、输入tokenization、诊断时钟交换在计时外。T2逐步KV检查含诊断开销，不能作正式性能对比；T3计时后核对保存的logits/token和最终KV抽样。

记录H2D片段事件、compute事件、copy-ready等待、slot复用等待、阻塞activation接收与staging、实际传输次数/字节、GPU allocated/reserved、RSS采样峰值及显式pinned字节。等待项可能重叠，不能简单相加。wire字节为应用层帧（含JSON与协议ACK），不含TCP/IP头及计时外时钟交换。

分节点时间线保持各自GPU原点。跨端H2D重叠范围来自8次控制消息交换的offset界和GPU anchor主机包络，假设短请求期间offset稳定；区间可能很宽，零下界不表示完全没有并行。这不是PTP或copy engine硬件利用率测量。

## 诊断轮次

`t3-*-01` 是旧priority过早暂停的诊断，不能与正式02混用；旧demand的host-wait仅计ready条件等待，02改为整个按需acquire阻塞。修正后重新进行了数值门槛与完整性能矩阵。

`t2-priority-fix-13b-01` 因旧诊断端口仍占用而启动失败，没有模型结果；使用02重试通过。随后加入启动互斥锁。被终止的旧运行及失败日志均保留，未据此作性能结论。
