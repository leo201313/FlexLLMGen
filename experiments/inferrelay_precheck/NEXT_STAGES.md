# E4/E5：后续验收方案（本轮未实现）

## E4：增量 prefill 后再做服务调度

先在本目录增加独立 attention 适配器，不直接重构 FlexGen。输入显式携带 request_id、query_len、past_len、position、cache 写入范围。先保持 FP16、GPU 常驻和单请求，使用相同权重/后端。

数值验收矩阵：prompt 1/17/63/129/257，chunk 16/32/64/128/256；覆盖不整除边界、q=1、p=0、p>0、左 padding。保存完整 prefill 与逐 chunk 最后有效位置 logits 的 max_abs、mean_abs、相对误差和 argmax 一致率；初始 atol/rtol 必须在测试前固定并记录，不能根据输出来调到通过。FP16 候选 atol=0.03、rtol=0.03，仅作为需要核验的起点。必要时提供 FP32 参考，以区分数值差异与语义错误。

逐层检查 KV 长度、内容及写入位置、causal mask、position；中间 chunk 不生成 token。以两个不同 prompt 的请求交错执行并与独立执行对照，检测 KV 串用。不能只比较生成文本。

通过数值验收后，用同一后端和可复现开放式到达轨迹比较：完整 prefill、固定 chunk、decode 优先 token budget。decode 优先策略必须包含最长 prefill 等待或 aging 规则，以免饥饿。先全 GPU，再加入同样的权重 offload；所有基线均支持相同 chunking 能力。

负载：短/长混合、持续 decode、突发请求；逐档增加到达率。每请求记录 arrival/start/first-token/token timestamps/end、排队时间、prefill 等待、最大 token 间隔。记录 TTFT、逐 token 间隔 P50/P95/P99、端到端延迟、完成/超时/未完成请求、显存与主存。

预先固定 SLO 网格，例如 TTFT 100/500/1000ms × 最大 token 间隔 50/100/200ms。请求必须同时满足两项才算成功；以固定测量窗口内满足条件的完成请求数/秒报告 goodput，同时记录跨窗口未完成数和 drain 规则。数值是待评估的阈值网格，不是产品承诺。保存所有阈值而非只取有利档位。

进入 E5 的前提：增量语义正确；明确调度收益来自何处；不能把单机 chunking 作为 InferRelay 独有创新。

## E5：实测两机四层组

两端都记录 hardware/software/commit、PCIe、实际 NIC 链路和配置。先测 E1 所得到 activation 字节数对应的 CPU staging 与 GPU staging 单向传输，分别覆盖小 decode、大 prefill；与 H2D 和 GPU 计算并发再测。不能默认 GPU Direct RDMA。避免跨机未同步时钟直接相减；用 ping-pong/本地完成时间和时钟误差校准，明确单向指标的测量口径。

原型按顺序验收：

1. A:G1→B:G2→A:G3→B:G4，固定请求、阻塞消息，先比较 logits/KV 与单机基准。
2. CPU 保留所属组权重，GPU 只读 buffer 用事件管理；加入对方执行期间的下一组预取。
3. 接入已通过 E4 的 chunked prefill。
4. 最后加入多请求调度和队列；明确权重复用与公平调度策略。

同一实现底座对照：连续切层+异步 offload；连续切层+offload+chunking；接力+chunking/关闭跨窗口预取；接力+chunking/开启跨窗口预取；必要时接力不 chunking。两台机器、模型、精度、请求轨迹、SLO、预算和调参次数一致；连续基线允许按异构能力调整切分。保留全 GPU 或更少通信的配置作为可行性对照。

每阶段先固定请求校验数值，再开启并发；记录 GPU compute/H2D/idle、activation staging/通信等待、暴露权重等待、两端峰值显存/RSS/pinned、通信次数和权重复用次数。展示 E3 预测有利和不利的两类配置，校准预测误差而非只展示最佳点。

决策要求：若额外通信超过减少的等待，报告退化区间；若收益全来自 chunking，明确与连续切层+相同 chunking 的差距；若只在过大 batch 或不现实网络成立，不称为交互服务收益。部署成本只能在相同 SLO 与负载下比较。
