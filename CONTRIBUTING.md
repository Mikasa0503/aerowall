# 贡献指南

项目现状、基线证据和复现流程见 [README.md](README.md) 与 [文档索引](docs/INDEX.md)。

## 开发边界

- 以 C350 双策略方案作为当前对照基线；候选实验使用独立名称、检查点和报告。
- 保留上游 VolleyBots、JuggleRL、HCSP 等代码及模型的原名称，不把直接复用的实现写成 AeroWall 原创。
- 不在运行实验时覆盖上游源码、正式检查点或既有证据。
- 新实验记录策略检查点 SHA-256、配置与场景版本、随机种子、运行时版本、接触审计结果和自然终止／时间截断状态。
- 单次演示、开发筛选和正式批量评测分别报告，不混作同一类证据。

## 本地工作区

运行时、上游 checkout、训练日志、原始 rollout、缓存和临时 worktree 留在本机或服务器，不纳入源码审阅。它们由 .gitignore 排除。经选择用于复现和演示的两个检查点与已验收视频单独保留。

## 代码与文档

- 新的 AeroWall 环境、奖励、观测、策略路由和诊断逻辑放在 aerowall/ 或 scripts/，并说明属于 AeroWall 还是上游实现。
- 稳定配置和复现样本放在 configs/；运行输出放在 runs/。
- 结果文档明确说明样本数、随机种子、评测上限、自然终止原因及统计边界。
- 不以环境步数、训练更新数或接触事件数替代策略成功率。
- 兼容旧实验入口时注明其历史用途与当前受支持入口。

## 复核命令

在准备好锁定的 Isaac Sim 环境和上游源码后，可运行纯 CPU 合约测试：

    bash scripts/python.sh --plain -m unittest discover -s tests -p 'test_*.py' -v

GPU 仿真评测见 docs/REPRODUCIBILITY.md。大型仿真测试不应作为本机默认 CI 测试。
