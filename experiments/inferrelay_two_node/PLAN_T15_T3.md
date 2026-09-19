# T1.5 → T2 → T3 本轮预注册

不修改原版 FlexGen，不重装、不下载，不覆盖旧结果。先完成干扰诊断，再通过完整模型正确性门槛，最后比较性能。

## T1.5

OPT-1.3B、b1、q1/128；双方向；none/sender/receiver/both背景。固定每个活跃端搬运两个完整block的等量权重，然后执行一个真实block计算；同时保存activation ACK延迟、整个有限工作完成时间、权重有效带宽、activation完成至计算可用的等待。不能只暂停所有预取后比较activation。

bulk（实际16张量/block整体提交）、8MiB分块单个在途、8MiB分块最多2个在途、8MiB单在途+activation请求时暂停新权重提交。到达相位为0ms、2ms、固定种子随机0–8ms，记录实际到达；每配置1次warmup、10次正式重复。拆开事件准备、copy_调用、提交完成、event等待及CUDA事件时间。线程每配置预启动，结果分节点时钟记录。优化只视为有限工作集微基准收益，不能等同完整模型收益。

## T2

共享实现调用原FlexGen的opt_input_embed/mha/mha_gen/mlp；输出层采用原版相同layer_norm与linear公式，并暴露完整last-token logits。全局layer id直接索引实际权重，KV仅在所属节点。

每模型依次：单机完整模型参考、连续常驻、四组交替常驻、连续按需、交替按需；125M先过再1.3B。真实有效长度32和128，batch1，gen4，无padding替代有效长度。固定 logits/KV/hidden **atol=0.02, rtol=0.01**，argmax/token完全一致；每层KV长度和末写入位置完全一致，并对全部层K/V按均匀位置采样256元素（非全部KV逐元素），每生成步核验。预先固定，不事后调宽。

输入embedding在首节点，输出head在末节点，token返回首节点。token embedding/LM-head共享原始权重文件，两端各一副本，明确计入两端驻留字节。端点权重在所有offload布局中常驻；只offload Transformer block，属于受控机制实验。

## T3

两种布局同执行器、同通信、同权重槽位与预取粒度；C0/R0 demand，C1/R1 prefetch。预取在请求开始即可投递，不能等待activation。默认2个可复用槽位，按全局最大组字节等量分配到每端；CPU保留只读副本。buffer reuse依赖compute_done，compute依赖copy_ready，不在每个算子插入全局sync。

三种工作负载：b1,q32,gen8（报告decode）；b1,q128,gen4（普通prefill）；b4,q256,gen4（大activation）。方向A→B/B→A均测。连续切分按25/50/75%扫描，接力四等分；同模型同负载所有布局每端相同weight-slot总字节预算（以最大连续分组为准）。明确槽位多而模型小时可能接近可驻留，保留单机与两机常驻基线。

先1次warmup+3次重复看完整矩阵，主要结果小样本只能作为机制证据。记录原始步骤时间、H2D/compute事件、host等待、显式slot容量、KV/端点驻留、GPU allocated/reserved/RSS。两端时间线分别展示，不推断无时钟校准的精确跨机H2D交叠；是否同时投递可由请求开始控制和各端本地事件证据支持，无法精确对齐部分标unknown。
