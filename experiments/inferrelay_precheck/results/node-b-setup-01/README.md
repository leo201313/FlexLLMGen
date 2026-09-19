# B 节点配置验收 · 2026-09-19

节点 A 192.168.20.103；节点 B 192.168.20.105。配置步骤、复现命令与解释见 [SECOND_NODE_PLAN.md](../../SECOND_NODE_PLAN.md)。

- `inferrelay-b-validation.log`：pip check、CUDA FP16、原版 OPT-125M 小配置和 OPT-1.3B 默认配置的完整输出，顺序执行全部成功，进程退出码 0。
- `inferrelay-b-e1e2.log`：B 的 E1/E2 执行日志，退出码 0；原始结果在相邻 `node-b-smoke-01`。
- `inferrelay-*-iperf.json`：10 秒单流 TCP 客户端结果；`*-server.json` 是对应服务端。两个方向分别执行，客户端均退出码 0，JSON 无 error。
- `weights_sha256.json`：A/B 两模型 586 个文件逐文件 SHA256 一致。
- ICMP 基础验收：10 次，间隔 0.2s，0% 丢包，RTT min/avg/max/mdev = 0.131/0.236/0.259/0.035ms（终端摘要，未保留原始日志）。

这些是配置验收与独立组件测量，没有运行跨机模型接力。iperf3 长流吞吐不能代替小消息、D2H/H2D staging 测量。首轮超时失败的 iperf 输出不在此目录；这里仅保存随后重测成功的完整 JSON。
