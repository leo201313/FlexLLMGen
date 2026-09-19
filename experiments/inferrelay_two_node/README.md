# InferRelay 真实两机验证

当前已完成 **T0/T1、T1.5 干扰定位、T2 完整模型正确性、T3 最小性能对照**。最新结果与复现见 [T1.5–T3说明](README_STAGES.md) 和 [阶段结论](FINDINGS_T15_T3.md)。原版 FlexGen 和既有原始结果未修改。下文保留历史 T0/T1 的复现入口；尚未实现完整在线系统或 E3 校准。

## 结果入口

- [T0：112行两端组件数据](results/t0-01/README.md)：56个相同配置/节点，分别保留 H2D、compute、争用、字节数。
- [T1：128个配置全部结果](results/t1-summary-01/README.md)：CPU pinned、GPU staging、与权重H2D/计算并发，两个模型、两个方向、batch1/4、q1/32/128/256。
- [T1 分节点时间线](results/t1-summary-01/timeline.html)；[原始逐样本CSV](results/t1-summary-01/raw.csv)；[统计CSV](results/t1-summary-01/summary.csv)。各正式运行下 A/B 的 raw.json 保留完整背景工作事件，汇总JSON只挑代表性时间线。
- [阶段发现与下一步](FINDINGS.md)。不能把 T1 通信微基准当成 R1/C1 全模型对照。

## 一键运行与停止

环境已存在，不安装依赖、不下载权重。从 A 的仓库根目录执行；使用新标签，结果目录存在时拒绝覆盖：

```bash
bash experiments/inferrelay_two_node/run_t1.sh repeat02
```

只跑一个方向：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmexp
python -m experiments.inferrelay_two_node.launch \
  --model opt-125m --direction A-B \
  --out experiments/inferrelay_two_node/results/manual-125m-a-b
```

Ctrl-C 会停止当前运行。也可从另一终端停止指定运行：

```bash
~/miniconda3/envs/llmexp/bin/python -m experiments.inferrelay_two_node.stop \
  --run experiments/inferrelay_two_node/results/manual-125m-a-b
```

启动器记录本轮 worker PID 和命令，停止前核对 `/proc/PID/cmdline`；不进行宽泛进程清理。SSH 会话/进程组处理、30秒socket超时、60秒等待连接、默认600秒进程级超时相互兜底。失败保存日志和已有部分结果，对端被关闭/定向终止。数据端口5211只绑定目标内网IP；不支持同时运行多个使用同端口的实验。

## 可复现信息与计时

每端保存 environment/config、invocation、commit/status、软件版本、源码SHA256、完整源码快照、启动器命令、完成/失败状态、GPU allocated/reserved 和 RSS采样峰值。CPU pinned显式活跃字节记录在每配置中，allocator累计值不等于活跃内存。

T1 发送端计时至接收端完成 CPU接收/GPU拷贝后的ACK返回；包含ACK、主机处理与背景调度。准备元数据和ready握手在测量区间外；全模型端到端计时必须包括它们。不能把该时间称作纯单向网络时延，也不使用RTT/2或跨机时间戳相减。

通信buffer及背景线程按配置预分配/预启动；每配置首次传输、3次warmup、20次稳态分别标记。分配/setup包含背景独占基准，与传输时间分开。完整payload及GPU readback在ACK后校验，失败立即退出；固定容差为逐字节/逐元素完全一致。

背景负载为对应模型从layer0开始的**一个完整block**：H2D复制实际张量到独立GPU目标；compute复用原FlexGen kernel，q1配past128真实组内KV，其余q为普通prefill。背景持续到传输完成，每次完成一个工作才提交下一个；记录实际次数与CUDA区间。后台主机线程可能竞争GIL，不能把所有退化归因GPU资源。

## 开发验证记录

- `t1-pilot-01`：SSH已是会话首进程，重复setsid失败；未产生测量。
- `t1-pilot-02`：8配置通信正确，但每次传输新建背景线程，仅开发验证，不纳入正式汇总。
- `t1-pilot-03`：预启动持久线程后的反方向8配置通过。
- `test_wire.py`：分片接收、EOF、头长度/元数据错误。需在允许socket的环境运行。
- `t1-stop-check-01`：通过stop模块主动停止双方，端口均关闭；预期失败，不属于性能测量。
- `t1-timeout-check-01`：刻意设置1秒进程超时，验证退出与定向清理；预期失败，不属于性能测量。
