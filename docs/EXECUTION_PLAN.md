# AeroWall 现行实施计划

**当前计划：** [HCSP 技能升级与能力保持实施计划](plans/2026-09-24-hcsp-skill-upgrade-implementation-plan.md)

该计划已于 2026-09-24 确认，取代本文件此前关于项目部署、CTBR／FF-PPO／GRU／Estimator 三方法研究和 6–8 周交付的旧计划。旧计划不再作为当前工作的验收依据；其内容由 Git 历史保留。

## 当前阶段
Part 1–14 仍在实施与能力验证阶段，状态详见 [Part 1–14 进度表](part1-14-progress.md)。Part 1–3、11–12 首轮范围已完成；Part 4–10、13 仍有能力验证或条件工作未闭环。Corrected `relative_v3`、Tanh Hit 三种子、Recover-only Tanh、CausalV6 Hit reward 和 Launch 截点目标保留实验均未通过 P2。新增预注册的 `AeroWallLaunchLateralMomentumCounterfactualV1` 在 12 对定向 v4 开发状态上达到局部机制 screen：intervention 6/12 合法首触 vs control 3/12，配对径向改善中位数 0.0279 m；双方首触均通过 PhysX 核验、20 项 reset 字段误差为 0。该状态干预不证明策略能通过动作实现该改变；intervention 中两球后来越过 world-y ±3 m 边界而 out-of-bounds（其中一案已有 4 次 rallies）。P2 仍 false，正式 C350 不变，不训练、不晋升、不创建最终测试集，P3–P6 继续 gated。后续若研究 action-level controllability，先单独预注册并纳入 out-of-bounds endpoint。

具体文件、函数、配置字段、依赖、课程阶段、HCSP 迁移边界、指标与停止条件均以现行详细计划和进度清单为准。实验运行成功与策略通过能力门槛分别记录，不据单一种子或开发 bank 结果晋升策略。

## 被替代计划的历史记录

- 本文件在 2026-09-20 的旧版规划了 JuggleRL/CTBR 主线与 FF-PPO、GRU、Estimator 等后续研究；其路线不再指导现阶段工作。
- [2026-09-23 连续回中与偏球恢复计划](plans/2026-09-23-aerowall-recenter-recovery.md)记录了 C350 训练、恢复门槛和相应实验过程。该实验记录仍是有效历史证据，但不再是当前后续实施计划。
- [方向审计](plans/2026-09-23-aerowall-direction-audit.md)、`docs/EXECUTION_STATUS.md` 及其他带日期报告保留各自时点的结果；其中历史状态和旧计划目标不自动成为新计划的验收要求。

本项目当前策略保留规则：正式 C350 是能力基线；新策略必须通过现行计划的固定、未见来球和安全验收后才可替换。演示成绩与训练评估窗口需按相同协议比较。
