# 两机验证实施方案（运行前固定）

本轮先交付 T0/T1：对齐已有组件测量，并采用未来执行器可复用的 TCP FP16 contiguous 张量协议测真实 activation 通路。之后依次 T2 正确性、T3 预取、T4 公平性、T5 校准；不引入 chunked prefill 或在线调度。

## 复用与新增

- 原版 `flexllmgen` 不改动。复用 `inferrelay_precheck.components.Blocks` 的完整 OPT block、实际权重和 CUDA kernels，仅作为 T1 并发干扰负载；它的独立组 state 不作为全模型正确性参考。
- `t0.py`：对齐 A/B 的 E1/E2，保存来源 SHA256、独立 compute/H2D、争用和内存口径。
- `wire.py`：有界 JSON 元数据 + 原始 FP16 contiguous payload；recv_into 直接写入预分配 pinned buffer，校验 run/request/step/group/shape/dtype，禁止 pickle。
- `activation.py`：同一 CPU/GPU 通路，支持无背景负载、两端权重 H2D、两端实际 block 计算；冷启动与稳态分开。
- `launch.py`：只启动/停止本轮 SSH 子进程，有限超时、失败保存日志，断开连接通知对端退出。
- `report.py`：P50/P95、阶段时间、并发 slowdown、按节点的时间线，保留所有退化配置。

## 最小矩阵与计时

T0 先核对已有 125M：完整 block 数 1/2/4/8，batch 1（4 补充），q/past 与来源一致。缺少四等分原型所需 125M group3 和 1.3B group6，不用线性倍率伪造；T2/T3 前按实际分组补测。

T1：先 125M 通过再 1.3B；hidden=768/2048，batch=1/4，q=1/32/128/256；双方向，每形状 cpu/gpu/gpu_h2d/gpu_compute 四模式，共 128 配置。每配置首次传输单独保留，3 次 warmup，20 次稳态重复。125M 小配置先验证协议再执行完整矩阵。

背景负载使用对应模型一个完整 block，prefill 与传输 q/batch 一致，decode 用 past=128 的真实组内 KV；两端分别测同负载独占时间。背景线程重复投递一个工作并等其完成，记录实际次数和 CUDA 时长。该负载包含 Python 调度竞争，不能只解释为 GPU 争用；记录局部事件检验实际重叠，不能假设线程启动即实现重叠。

端到端计时：发送端 D2H 前至接收端 CPU 收到 / GPU H2D 完成后的 ACK 返回。含协议、主机处理和 ACK，非纯单向网络时间。D2H/H2D 为本机 CUDA events；发送端剩余墙钟扣除本机 D2H 主机时间及对端 H2D 主机时间，只作网络+主机+ACK残差，不是精确 wire 时延。所有 payload 完整字节校验（GPU 再拷回逐元素校验）放在 ACK 后、下一轮前，不计入传输延迟；失败立即退出，不对错误结果作性能结论。GPU readback 校验会影响下一轮缓存/时钟，须列为限制。

启动及 buffer 分配/首次传输独立记录，不称为 OS 冷启动。稳态通信 pinned/GPU buffer 预分配复用，发送完成前不可覆盖。元数据/ACK 仍存在 Python 对象分配成本。两机时间戳不相减，时间线分节点展示。

## 后续预注册

T2a：125M 四组3层、1.3B四组6层，CPU/GPU后端共享；本机全模型、连续常驻、接力常驻；真实有效 prompt 长度32/128、gen8。运行前固定 logits atol=0.02, rtol=0.01，argmax/token完全一致，KV长度/位置完全一致、抽样值同容差。失败定位，不能事后放宽。

T2b/T3：同执行器添加只读 CPU 权重和 event 管理的槽位，C0/C1/R0/R1，主比较 R1/C1。预取不依赖 activation 到达，公平性以每节点实际字节预算而非槽位数判断。

T4：交换起点；连续切分在25/50/75%先扫、最优附近细化，与接力相近调参预算；单机常驻、两机连续常驻；等字节预算下更多预取/部分驻留；最后少量 microbatch 权重复用。

T5 预选旧125M形状：group2,q128,b1（有利）；group1,q256,b4（不利）；group8,q128,b1（近边界）。四组原型与这些组大小不一致，后续须按这些 group 数显式配置，不能把四组结果伪装为同配置校准。保存旧预测、组件更新预测、真实 block stack 和含 embedding/输出/token 返回的端到端结果。不得用拟合总时间代替修正依赖。
