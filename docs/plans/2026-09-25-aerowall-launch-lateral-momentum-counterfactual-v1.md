# AeroWall Launch Pre-contact Lateral-Momentum Counterfactual V1

**预注册日期：** 2026-09-25

**状态：** 机制诊断设计；不训练、不晋升、不替换正式 C350。

**作用范围：** Part 7 的 Launch handoff/reset 契约与 Part 13 的开发期配对因果诊断。

## 依据与假设

现有同一 `heldout-128-v4` 开发 bank 的 12 对匹配首触中，非法案在最后动作前约 0.40 s 的无人机 world-`v_y` 均高于匹配合法案；12/12 差值为正，中位数 `+0.136236 m/s`，范围 `+0.014532` 至 `+0.610596 m/s`。同期非法案 local-y 中位差为 `−0.144314 m`，而最终首触均由 Launch actor 执行。该关系仅来自失败富集的匹配开发样本，不能证明因果，也不能估计总体率。

**可证伪假设：** 对这些匹配的 phase-0 失败状态，过大的无人机 world-`v_y` 是导致径向接触误差的一项可干预因素。将失败状态的无人机 world-`v_y` 单独设为其匹配合法状态的 world-`v_y`，应降低首次接触径向误差，并使一部分非法首触转为合法。

## 固定配对协议

- 使用 `heldout-128-v4` 原始自然 RALLY 开发 bank，评估 seed `9524`，冻结的 V6 Launch、Tanh Hit 和 C350 Recover checkpoint、原 FSM、reward、观测版本及 PhysX 设置全部固定。
- 只纳入预先列明的 12 对（每对 1 个非法案与其 1 个合法案）。在各 case 球心按 `g=9.81 m/s²` 下落至 `z=2.18 m` 的弹道预测时间为 `0.40 ± 0.02 s` 时记录完整 Launch 状态；从非法案快照构造成对续跑。
- Control 从非法案原始快照续跑。Intervention 从同一快照续跑，**唯一改动**是把 `drone_velocity[1]`（world-y 线速度）替换为该对合法案快照中的同一字段。保留 ball/drone 其余状态、姿态、角速度、throttle、上一动作、FSM、checkpoint、seed 及环境参数；两臂均使用 deterministic actor 与自然 RALLY bank，不启用随机 perturbation。
- 对两臂逐案核验：干预字段以外的所有可恢复物理/FSM 字段最大误差不超过 `1e-4`；所选 actor、来源 bank、源轨迹和交接字段均记录 SHA256。首个动作必须能与源状态复现；否则该 pair 无效，不补抽、不换配对。
- 运行到首个身体接触或终止。逐案记录 first-contact radial/axial 误差、合法性、后续 rally、安全失败与 PhysX contact corroboration。此 12 对筛选集只作机制诊断，不作为总体率或 P2 acceptance。

## 预注册判定

**支持该机制的最低信号：** 12 个原失败状态中至少 6 个在 intervention 下形成合法首触，且首次接触 radial error 的配对中位改善至少 `0.02 m`；相对 control 不新增无人机触地、非法接触或实际 PhysX 无人机撞墙，所有 scored contact 均有完整 PhysX audit，且无 callback error。

**拒绝条件：** 合法首触转换少于 6/12、radial error 中位改善小于 `0.02 m`、安全失败增加、任一 contact audit 缺失，或非干预状态重置误差超过 `1e-4`。不因失败结果改变阈值、重选配对、扩大样本、切换 checkpoint 或追加动作偏置。

即使达到最低信号，也只支持继续研究这一状态因素；该结果不通过 P2、不授权训练、不晋升策略。任何后续策略训练仍须满足正式 P2 gate，并经用户另行明确批准。v4 保持开发集身份，独立最终测试集继续不创建。

## 预先固定的配对

| 非法案 | 匹配合法案 |
| --- | --- |
| heldout-0008 | heldout-0095 |
| heldout-0012 | heldout-0100 |
| heldout-0018 | heldout-0004 |
| heldout-0033 | heldout-0096 |
| heldout-0051 | heldout-0029 |
| heldout-0059 | heldout-0075 |
| heldout-0073 | heldout-0070 |
| heldout-0082 | heldout-0037 |
| heldout-0099 | heldout-0068 |
| heldout-0107 | heldout-0124 |
| heldout-0112 | heldout-0103 |
| heldout-0113 | heldout-0058 |

## 续跑前测量细则锁定（2026-09-25）

为使“首触径向误差”在无接触终止时有确定解释，先于 intervention 续跑补充固定：每臂每案取最早的 `body == true` 球—拍接触事件，按 `(policy_step, substep)` 排序，记录其 `radial_error` 与 `legal_cap`；若终止前没有 body 接触，该臂误差记为缺失，不插补。配对径向改善定义为 control 误差减 intervention 误差，只在两臂均有首触误差的配对上计算；至少需要 6 个完整配对，否则本筛查按拒绝/无支持信号处理。至少 6/12 的合法首触数仍独立按 intervention 的首个 body 接触 `legal_cap=true` 计数。

每个被计分的首触事件还必须能在 PhysX contacts sidecar 中以相同 `env`、step、substep 找到 drone `base_link` 与 ball 的正 contact_count；事件漏审计、callback error 或合法 cap／wall 汇总不完全 corroborate 均拒绝本轮支持信号。安全事件按 evaluator failure code 2（drone-ground）、4（illegal contact）、5（drone-wall）逐案比较；intervention 新增任一该类失败即不支持。以上补充仅规定缺失值和审计方法，不改变既有样本、干预、阈值或阶段 gate。

## 产物记录

实现、源快照 bank、双臂 handoff bank、reset audit、paired continuation、源码哈希及结果将于执行后追加。执行前不浏览或修改任何配对 case 的逐值遥测。
