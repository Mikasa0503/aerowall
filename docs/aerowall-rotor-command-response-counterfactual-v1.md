# AeroWall Rotor Command Response Counterfactual V1

**日期：** 2026-09-25

**状态：** 预注册无训练诊断已完成；不通过 P2，不授权训练或晋升。

**正式策略：** C350。

## 问题与执行

本实验检查能否用现有 AeroWall actor 的直接旋翼命令，在 Launch 接近早期把无人机 world-y 速度推向同一失败状态的匹配合法参考值。

固定 seed 9524、12 个已登记 Launch handoff 状态、V6 Launch／Tanh Hit／C350 Recover、自然 RALLY 续跑和原环境、FSM、奖励、观测配置。每个状态运行一个 control 和八个处理：在策略步 0–9 对 rotor index 0–3 中单一通道逐步加或减 0.10，第 10 步起恢复原策略。共 9 臂、108 个短续跑，没有训练或更换 checkpoint。

9 臂的 post-reset state SHA256 完全相同：`f5edcc051b0b154f66a83d53c368916bff51b60b3f5368679c46918987234280`。Torch、NumPy、CUDA RNG 指纹相同，control 第一步 actor action 与来源动作最大差 `2.38e-7`，所有处理臂第一步 raw actor action 与 control 完全一致。每臂均对 12 案施加了完整 10 步处理。每臂 12/12 首次机体接触均由同 step/substep 的 PhysX base_link—ball contact corroborate，contact callback errors 为 0。

## 结果

“朝参考”比较处理与 control 在策略步 9 后的 world-y 速度与匹配合法参考之间的绝对差距；正信号要求同一旋翼通道和符号至少 9/12 案朝参考移动，且绝对速度变化中位数至少 0.02 m/s。

| 处理 | 朝匹配参考 | world-y 速度变化绝对值中位数 | 合法首触 | 新安全失败 | 新球越界 | 被执行器限幅的处理行 | 判定 |
|---|---:|---:|---:|---:|---|---:|---|
| rotor 0 +0.10 | 0/12 | 0.0161 m/s | 0/12 | 3 | 0 | 0/120 | 无信号；拒绝候选使用 |
| rotor 0 −0.10 | 12/12 | 0.0171 m/s | 4/12 | 0 | 1（0113） | 0/120 | 未达幅度门槛；出现越界 |
| rotor 1 +0.10 | 0/12 | 0.0353 m/s | 4/12 | 0 | 1（0113） | 23/120 | 方向不符；出现越界 |
| rotor 1 −0.10 | 12/12 | 0.0361 m/s | 0/12 | 3 | 0 | 0/120 | 达局部信号门槛；新增非法接触 |
| rotor 2 +0.10 | 0/12 | 0.0263 m/s | 6/12 | 0 | 3（0033、0107、0113） | 0/120 | 方向不符；出现越界 |
| rotor 2 −0.10 | 12/12 | 0.0265 m/s | 1/12 | 2 | 0 | 0/120 | 达局部信号门槛；新增非法接触 |
| rotor 3 +0.10 | 0/12 | 0.0208 m/s | 1/12 | 2 | 0 | 58/120 | 方向不符；有显著限幅和新增非法接触 |
| rotor 3 −0.10 | 12/12 | 0.0253 m/s | 3/12 | 0 | 0 | 0/120 | 达局部信号门槛；无新增安全／越界失败 |

Control 为 3/12 合法首触、中位首触径向误差 0.21763 m，0 次球越界。`rotor 3 −0.10` 处理的首触也为 3/12，径向误差中位数 0.21644 m（仅改善约 0.00119 m），平均 rallies 为 0；它没有带来已观察到的回合能力增益。

三个方向通过局部速度门槛：`rotor 1 −0.10`、`rotor 2 −0.10`、`rotor 3 −0.10` 都有 12/12 案朝匹配参考移动，速度效应绝对值中位数分别为 0.03609、0.02651、0.02527 m/s。前两者分别新增 3 案和 2 案非法接触；按预注册规则排除候选使用。`rotor 3 −0.10` 没有新增 drone-ground、drone-wall、illegal-contact 或 world-y 边界失败，因此是唯一未被这项安全规则排除的局部信号。它仍只是在这 12 个定向开发状态里的动作敏感性证据，不是可用策略或总体效果估计。

球越界按完整 active trajectory 检查 `abs(ball_world_y) > 3 m`：control 为 0 案；rotor-0 负向和 rotor-1 正向各新增 heldout-0113，rotor-2 正向新增 heldout-0033、0107、0113。越界 arm 均予以保留，没有因获得 rallies 而忽略。

### 可复现性说明

评估器给 Torch、NumPy 设了 seed，但没有给 Python 标准库 `random` 设 seed。因此九臂的 post-reset 物理/FSM state、Torch/NumPy/CUDA RNG 指纹和首个 actor action 相同，而 Python 标准库 RNG 指纹跨进程不同。本次记录了这一点；AeroWall 环境和策略模块没有使用标准库随机调用。独立复核应给标准库 RNG 增加显式 seed，并使用预先冻结的新 case bank。

## 决策与后续

本 screen 找到一个未触发已登记安全拒绝条件的局部动作响应（rotor-3 的 -0.10），但该处理没有改善合法首触率或 rallies。不得把它直接加到 actor、训练新 checkpoint 或替换正式 C350。下一步若继续，先为未用于此实验的 case bank 预注册独立复核，给所有 RNG 显式设 seed，并以首触、续拍、安全失败和球越界作为共同端点；只有新样本也支持且任务指标提升，才重新讨论候选。

P2 仍未通过；P3–P6 继续 gated。本实验没有进行训练、晋升或正式策略变更。

## 产物

- 预注册协议：[`docs/plans/2026-09-25-aerowall-rotor-command-response-counterfactual-v1.md`](plans/2026-09-25-aerowall-rotor-command-response-counterfactual-v1.md)
- 可复算脚本：`scripts/analyze_aerowall_rotor_command_response_counterfactual.py`，SHA256 `650884dea00f62a5930f417dd470321387762390422d0e3841f6616695523703`
- 机器分析：`artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-rotor-command-response-counterfactual-v1-analysis-s9524.json`，SHA256 `52de9078969a888751493c58a2123904073a6cab50581d4a847767c88a5540cc`
- evaluator：`scripts/evaluate_aerowall_wall_rl.py`，SHA256 `c474b5905bdc0c5e7fb13035a5097131ae2c16a8e6ef6f13b9d6c4e5d41b504b`
- 九臂的逐案 report、完整 trajectory、events、PhysX contacts 和运行日志都保存在 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/`。
