# AeroWall 技能升级 v3：执行与验收记录

**日期：** 2026-09-25

**结论：** T0–T5 已执行，但 P2 gate 仍未通过。Recover-only Tanh 消融减少动作限幅却使续拍下降、安全失败增加；CausalV6 Hit reward 对照无回合能力提升。Corrected relative_v3 control 与 Launch lateral-momentum 状态反事实均已留档。最新 AeroWallRotorCommandResponseCounterfactualV1 在 12 个定向开发状态中发现 rotor 1/2/3 的负向局部速度响应各为 12/12 朝匹配参考；rotor 1/2 分别新增 3/2 案非法接触，只有 rotor-3 负向未触发本 screen 安全拒绝条件，但首触仍 3/12、rallies 为 0。正式 C350 保持，不训练、不晋升，P3–P6 gated。

## AeroWall Rotor Command Response Counterfactual V1（2026-09-25）

按冻结协议固定 seed 9524、12 个 Launch handoff 和 V6 Launch／Tanh Hit／C350 Recover；执行 fresh control 与 8 个 rotor-index 0–3、±0.10 单通道处理，共 108 个续跑。每个处理臂的 12 案都完成前 10 个策略决定，完整命令轨迹为 120 行；每臂 12/12 首次 body contact 都有 PhysX sidecar 对应审计，callback errors 为 0。九臂 post-reset physical/FSM state SHA256 相同；Torch、NumPy、CUDA RNG 和首个 actor action 指纹相同。fresh control 的首个动作相对冻结来源最大误差为 2.4e-7。

| 处理臂 | 朝匹配 world-y 参考 | 速度效应绝对值中位数 | 合法首触 | 新安全失败 | 新球越界 | 限幅命令行 |
|---|---:|---:|---:|---:|---|---:|
| rotor 0 +0.10 | 0/12 | 0.0161 m/s | 0/12 | 3 | 0 | 0/120 |
| rotor 0 -0.10 | 12/12 | 0.0171 m/s | 4/12 | 0 | heldout-0113 | 0/120 |
| rotor 1 +0.10 | 0/12 | 0.0353 m/s | 4/12 | 0 | heldout-0113 | 23/120 |
| rotor 1 -0.10 | 12/12 | 0.0361 m/s | 0/12 | 3 | 0 | 0/120 |
| rotor 2 +0.10 | 0/12 | 0.0263 m/s | 6/12 | 0 | heldout-0033/0107/0113 | 0/120 |
| rotor 2 -0.10 | 12/12 | 0.0265 m/s | 1/12 | 2 | 0 | 0/120 |
| rotor 3 +0.10 | 0/12 | 0.0208 m/s | 1/12 | 2 | 0 | 58/120 |
| rotor 3 -0.10 | 12/12 | 0.0253 m/s | 3/12 | 0 | 0 | 0/120 |

Control 为 3/12 合法首触、首触径向误差中位数 0.21763 m、0 越界。rotor-3 负向为 3/12、0.21644 m（只改善约 0.0012 m），平均 rallies 仍为 0。Rotor-1 与 rotor-2 负向分别新增 3 案和 2 案非法接触，按预注册拒绝规则不得用于候选；rotor-3 负向是唯一未被 screen 安全规则排除的局部速度信号，但没有显示任务提升。它不能用来替换 C350 或授权训练。

评估器只 seed Torch 和 NumPy，Python 标准库 random 指纹在 9 个进程间不同；这一限制已写入结果分析。独立复核应显式 seed 所有 RNG 并使用此前未触碰的 case bank。该 screen 仅适用于本次 12 个定向 development 状态，不通过 P2，不改变正式路由。

完整结果：[AeroWallRotorCommandResponseCounterfactualV1 报告](aerowall-rotor-command-response-counterfactual-v1.md)、[冻结预注册](plans/2026-09-25-aerowall-rotor-command-response-counterfactual-v1.md)、[机器分析 JSON](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-rotor-command-response-counterfactual-v1-analysis-s9524.json)。分析报告 SHA256 52de9078969a888751493c58a2123904073a6cab50581d4a847767c88a5540cc；分析脚本 SHA256 650884dea00f62a5930f417dd470321387762390422d0e3841f6616695523703。

## AeroWall Launch 横向动量反事实 screen（2026-09-25）

按预注册协议固定 seed 9524、V6 Launch／Tanh Hit／C350 Recover、`heldout-128-v4` 与 12 对失败／合法参考状态；在预测接触时间 0.40±0.02 秒捕获 24 个状态。自然源 rollout trajectory SHA `bf0f2aee9e7f065acad26b2121680487995ad01dfb58702ccc6e41dca1401c06` 与正确 Launch-observation control 基线一致。24/24 handoff 被接受且来源 callback errors 为 0。

Control 与 intervention 各有 12 案。两侧 20 项 reset audit 报告均为 passed、逐字段误差为 0；intervention reset-only 进程在报告写盘后于 `SimulationApp.close` 报 native segmentation fault。完整 intervention continuation 后续以 passed 状态结束。source→control 首动作最大差 `2.4e-7`。每侧最早 body 首触 12/12 均在同 env／step／substep 对应到 drone base_link—ball 正 PhysX contact_count，callback errors 为空，合法 cap 与 wall 汇总均完整 corroborate。唯一干预为把失败案 `drone_velocity[1]` 设为匹配合法案的实测值；其中一对速度差为负（−0.03164 m/s），按预注册保留。

| 首触指标 | Control | Intervention |
| --- | ---: | ---: |
| 合法首触 | 3/12 | 6/12 |
| 配对径向误差改善中位数（control − intervention） | — | +0.027883 m |
| code 2/4/5 新增安全失败 | — | 0 案 |
| 球越过 world-y ±3 m 终止 | 0 案 | 2 案 |

按预先固定的门槛，局部机制 screen 达标；相较 control 净增合法首触为 3/12，其中 heldout-0033、0073、0107 转为合法。两个球越界 case 为 heldout-0033、0107：0033 在 trajectory index 68 到达 `[0.8327, -3.0206, 3.4483] m` 并终止，0107 在 index 316 到达 `[0.7056, -3.0070, 0.1206] m`，此前完成 4 次 rallies。它们是 ball out-of-bounds（code 3），不是无人机触地或撞墙。所有 12 案最终仍以失败结束，因此不宣称整体任务成功。此结果只支持对 12 个定向 v4 development 状态的机制线索；它直接干预状态，不证明当前策略能通过动作实现同样改变，也不通过 P2、不授权训练或晋升。正式 C350 保持，P3–P6 gated。

完整逐案结果与 SHA256 见 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-launch-lateral-momentum-counterfactual-v1-analysis-s9524.json`（`16c005b4fa625135d9eab0c62aec38dcc4d0b1630920d118f35c59898a7bff5b`）。预注册为 `docs/plans/2026-09-25-aerowall-launch-lateral-momentum-counterfactual-v1.md`；银行构造与分析入口分别为 `scripts/build_aerowall_launch_lateral_momentum_banks.py` 与 `scripts/analyze_aerowall_launch_lateral_momentum_counterfactual.py`。


**工作区迁移：** 项目内容已迁入并核验。Isaac Sim 等共享运行时与临时缓存不属于项目源码；冲突备份属于本地忽略文件，不纳入项目分发。

迁移复核在运行 Isaac Sim 的环境完成。仓库不依赖该主机的绝对路径。

## 名称与上游边界

本次新增／修改的任务环境、技能编排、观测扩展、奖励和训练入口归属 **AeroWall**。HCSP 只保留原名用于其未修改的 `MAPPOPolicy`、物理仿真和控制。旧脚本名继续作为兼容入口。

| 实现 | 最终名称 | 来源 |
| --- | --- | --- |
| 单墙 rally 环境 | `AeroWallSingleWallRallyEnv` | AeroWall 自有环境，使用 HCSP 仿真基础设施 |
| 冻结技能链 | `AeroWallSkillChainPolicy`、`AeroWallLaunchRecoveryPolicy` | AeroWall 自有策略编排，底层 actor 使用 HCSP `MAPPOPolicy` |
| Hit 目标观测 | `aerowall_goal_v1` | AeroWall 自有 48 维视图：原 46 维特征加 2 个归一化墙目标通道 |
| 因果奖励 | `aerowall_causal_v3` | AeroWall 自有版本；本次 Hit pilot 按计划使用原样 `legacy` 奖励 |
| 训练／评估／案例入口 | `train_aerowall_wall_rl.py`、`evaluate_aerowall_wall_rl.py`、`build_aerowall_wall_cases.py` | AeroWall 规范入口 |

`train_hcsp_wall_rl.py`、`evaluate_hcsp_wall_rl.py`、`hcsp_wall_rl_env.py`、`hcsp_chained_policy.py`、`hcsp_policy_encoder.py`、`hcsp_wall_recovery_logic.py` 和 `build_wall_initial_cases.py` 仅保留兼容导入或命令转发。HCSP 固定提交为 `009961b8f5702dd0c1c943cef0e01e09dfcd138d`。历史奖励函数源码内容保持不变，SHA-256 仍为 `0ec90fda95baec92cba313b0c37511a52ef820bfbd620b9aa1ba710f9312f1d7`。

## 实施与验证

- **T0：** 固定 Launch、C350 Recover、实验 Intercept 检查点，冻结 fixed/train/heldout case banks 与配置哈希；原有未提交工作区内容保留。
- **T1–T3：** 完成相对速度坐标修正、新增 goal 观测、显式 FSM 推进、接触优先级与技能所有权、局部 reset、有限墙预测及奖励分项。`relative_v2` 与 `legacy` 语义保留供旧检查点复现。
- **T4 A/B：** 128 环境、600 步、seed 123。同权重包装器在 fixed bank 都是平均 8.0 个回合、中心回合 8.0、零安全失败；training bank 都是平均 1.8516、中心回合 1.5、安全失败率 77.34%。两个 bank 的逐步动作、无人机状态、球位置与速度、active mask 全部逐元素相同，最大差值为 0。合法接触事件均有 PhysX 报告：fixed 为 1,152/1,152，training 为 307/307；对应墙面事件分别为 1,152/1,152 与 272/272。
- **T4 C：** 只替换实验 Intercept 前序（并按其检查点使用 `relative_v2`）。fixed bank 128/128 撞无人机墙、零回合；training bank 平均 0.0078 回合、安全失败率 48.44%，157/157 合法接触与 6/6 墙面事件经 PhysX 核验。拒绝该前序策略。
- **16 环境 smoke：** 观测纯度、选择性 reset、C350 重载、有限奖励及 1+7 电机节奏检查均通过。该 smoke 没有观察到接触事件；Isaac Sim 在 `app.close()` 退出阶段仍打印 native segmentation fault，因此此 smoke 不作为接触审计证据。

本地纯逻辑与兼容奖励测试 **15/15** 通过；隔离 Isaac Python 环境中的设计、奖励与真实 HCSP 张量／PPO 测试 **28/28** 通过。新旧 CLI `--help`、源码编译和 `git diff --check` 通过。

## T5：AeroWall goal Hit pilot

seed `6201`，50 次 PPO 更新、409,600 帧；Hit 从 C350 Recover warmstart，只训练 Hit，Launch 与 Recover 冻结，奖励为 `legacy`。训练 Hit 样本占 rollout 平均 14.14%；人工 Hit reset 为 1,263/1,787（70.68%）。actor 权重确实改变，完整 checkpoint 重载精确，冻结来源核验通过。

C350 actor 从 46 维扩展为 48 维：14 个张量原样拷贝，3 个输入张量补零；source actor 与 warmstart actor 的动作最大误差为 **0.0**。训练 checkpoint SHA-256：`d6a4c52ffb71d295d29e4ce3917b28cb232f8253df4f3ae12d50e4d97abc763c`。

| 评测分布 | 策略 | 平均回合 | 合法第二拍率 | 三回合率 | 安全失败率 | 物理审计 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fixed-128-v1 | C350 基线 | 8.0000 | 100.00% | 100.00% | 0.00% | 1,152/1,152 合法接触、1,152/1,152 墙面事件 |
| fixed-128-v1 | Hit pilot | 8.9688 | 100.00% | 100.00% | 0.00% | 1,276/1,276 合法接触、1,152/1,152 墙面事件 |
| train-128-v1 | C350 基线 | 1.8516 | 38.28% | 21.09% | 77.34% | 307/307 合法接触、272/272 墙面事件 |
| train-128-v1 | Hit pilot | 2.1172 | 37.50% | 25.00% | 60.94% | 338/338 合法接触、311/311 墙面事件 |
| heldout-128-v4 | C350 基线 | 0.0000 | 0.00% | 0.00% | 3.13% | 0 次合法接触；128 个 case 均核验 |
| heldout-128-v4 | Hit pilot | 0.0000 | 0.00% | 0.00% | 3.13% | 0 次合法接触；128 个 case 均核验 |

Hit pilot 在 v4 上与正式基线具有完全相同的首次 episode 动作和无人机状态轨迹；两者都以 124 次球落地和 4 次非法接触结束。冻结 Launch 没有接到球，Hit actor 没有获得有效击球机会。训练分布的回合与安全率有有限改善，合法第二拍率略低；这不满足“留出集第二拍和三回合率必须改善”的门槛，因此不扩种子、不晋升 pilot，不调换检查点。v4 是已用于开发的挑战集，不作为新冻结最终测试集。

## 追加：真实 Hit 交接 bank 与同状态配对诊断

首次采集尝试把环境观察版本设为 `legacy`。这会关闭 AeroWall 的技能 FSM 推进，因此尽管有接球／墙面接触，`skill_step_counts` 仍是 `[22173, 0, 0]`，没有 Hit 交接候选。该诊断运行保留在 `artifacts/wall-skill-upgrade-v3/handoff-source-b-train-128-s123-frozen-goal-route.json`，不作为 handoff bank 使用。

修正后启用环境 `relative_v3` 技能状态机，同时令 Launch、C350 Recover 与基线 Hit checkpoint 读取各自 `legacy` 观察视图；trainable Hit 视图保持 `aerowall_goal_v1`。在 `train-128-v1` 上完成 128 环境自然策略链评测：均值 1.8359 rallies、3／5 连率均为 21.09%；304/304 合法拍球与 279/279 墙面事件经 PhysX 对应审计。收集到 52 个交接候选，52 个通过状态与 cap→wall 事件前缀校验，拒收 0 个。bank SHA-256 为 `b284d8ed156bed842c4d8cc6196a17fa9d9e5b0d50ecbbd63af99e9095d60ca3`。

C350 基线与 T5 Hit pilot 从完全相同的 52 个状态继续评估（seed 123，bank SHA 相同）。报告确认这是条件性交接评估，104 个继承的 source prefix 事件具有 PhysX 来源审计；每次评测的新增合法拍球和墙面事件也全部有 PhysX 记录。

| 同一交接 bank 指标 | C350 Hit 基线 | T5 Hit pilot | 差值／解释 |
| --- | ---: | ---: | --- |
| 首次合法 Hit 率 | 92.31% | 92.31% | 持平 |
| 交接后平均 rallies | 5.000 | 5.519 | +0.519 |
| 交接后至少 3 连 | 50.00% | 53.85% | +2/52 个状态 |
| 交接后至少 5 连 | 50.00% | 53.85% | +2/52 个状态 |
| 墙目标命中率 | 68.94% | 92.78% | pilot 有正向信号 |
| 安全失败率 | 40.38% | 40.38% | 持平 |

这是来自训练分布筛出的条件 bank，样本仅 52，属于诊断信号；不能代替自然整链留出评测。T5 pilot 在 v4 自然整链上仍为零首次接触，故仍不晋升、不扩展为三种子训练，正式策略保持 C350。

执行环境曾有一轮 Isaac `app.close()` native segmentation fault，但最终 JSON 报告已写为 `passed`，完整配对评测报告也正常落盘；旧式 smoke 的退出崩溃已在 T4 记录中说明。本轮没有更改 checkpoint 或正式路由。

### Isaac reset 字段级读回

新增 `--handoff-reset-audit-only` 评估模式，在执行策略动作前，将 bank 中 52 个状态与 Isaac reset 后的 live state 逐项比较。20 个字段全部在 `1e-4` 绝对容差内通过：无人机和球位置的最大误差为 `9.5367431640625e-7`，无人机／球姿态、线角速度、电机 throttle、`prev_action`、`action_before`、FSM、回合／接触计数、wall target 及历史球速度误差均为 `0`。评估 JSON 为 `handoff-reset-audit-fsm-enabled.json`。这验证了序列化状态恢复字段；接触前缀仍由来源事件日志与 PhysX 报告单独审计。

## 追加：自然 v4 Launch 首击覆盖的侧向反事实诊断

为区分 Launch 首击覆盖与 Hit 接手后的表现，固定 `AeroWallLaunchRecoveryPolicy`（Launch + C350 Recover）、`legacy` 观察、RALLY、seed 123，在原 heldout v4 bank 上分别制作 y 限幅、vy 限幅、同时限幅和仅翻转 vy 符号四个 **diagnostic-only** bank。每组 128 环境；源 bank SHA-256 为 `666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`。源 v4 的 128 个 case 全部是 y 与 vy 同号，且 y／vy 均超出 train-128-v1 的观测范围（train 的 ball y 为 `[-0.3864, 0.3972] m`、ball vy 为 `[-0.2914, 0.2989] m/s`）。

| 诊断变换 | 合法首击 case | PhysX 核验合法 cap 事件 | 平均最近径向距离 | 最小最近三维距离 | 安全失败率 | 合法第二拍 | 三连／五连 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| y 限幅至 `[-0.4, 0.4] m` | 0/128 | 0 | 0.4626 m | 0.3170 m | 27.34% | 0/128 | 0% / 0% |
| vy 限幅至 `[-0.3, 0.3] m/s` | 0/128 | 0 | 0.4550 m | 0.3291 m | 23.44% | 0/128 | 0% / 0% |
| y 与 vy 同时限幅 | 0/128 | 0 | 0.3768 m | 0.2771 m | 78.91% | 0/128 | 0% / 0% |
| 仅翻转 vy 符号 | 96/128 | 97/97 | 0.1434 m | 0.1493 m | 63.28% | 1/128 | 0% / 0% |

四组报告的 `status` 均为 `passed`，表示评估执行与审计完成；四组技能 gate 均未通过。`vy` 符号翻转逐 case 验证除 ball `vy` 外所有字段相同，且每个翻转后的 case 均变成 y 与 vy 异号。该反事实输入下 24 个 case 因球落地失败、23 个出界、68 个非法接触、13 个无人机撞墙；97 次合法 cap 事件全部经 PhysX 核验。因为 sign-flip 改变了来球方向，它不是原始 v4 策略成绩，也不是泛化或晋升证据。

### 全量轨迹配对审计与根因定位

为判断 sign-flip 的首击增益是否来自更好的 Launch 控制，对原始 v4 与 vy 翻转 bank 在相同 `AeroWallLaunchRecoveryPolicy`、seed 123、128 环境配置下各重跑一次并启用 `--record-trajectory`。两次评测的逐案 outcomes 和汇总指标均与此前无轨迹报告完全一致，说明记录选项没有改变策略行为。两份报告与压缩轨迹的哈希都记在 `analysis.json`。

两组逐案初始状态除了 ball `vy` 符号以外完全相同。冻结 Launch 第一个控制步的 4 维 action 平均 L2 差为 `0.00625`，95 分位 `0.01057`，114/128 个 case 的差值小于 `0.01`；原始 action 平均 L2 范数为 `1.2553`。`legacy` 输入为 46 维，包含世界系 ball y／vy 与当前相对目标，但不包含显式侧向拦截点。

| 条件（前 20 个策略步，0.4 s） | 球沿初始外侧方向位移 | 无人机沿外侧方向位移 | 第 20 步有符号球机侧向间距 | 合法首击 |
| --- | ---: | ---: | ---: | ---: |
| 原始 v4 | +0.1867 m | +0.1209 m | 0.6776 m | 0/128 |
| 仅翻转 vy | -0.1867 m（向内） | +0.1059 m | 0.3193 m | 96/128 |

无人机在两组中的早期外侧位移相近，球路却因速度符号反转而分别向外／向内。结合小幅首步动作差异，96/128 的 sign-flip 首击主要反映更有利的球路，不表示冻结 Launch 已能追赶原始向外球路。`relative_v2`／`relative_v3` 同 actor 配对评测中，初始状态和首个动作相同，首触和续打 gate 也相同；只换观测语义不能解决问题。完整范围、特征索引、逐项哈希、轨迹指标和解释记录见 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json`。

### 同 actor 的 relative_v2／relative_v3 观测消融

使用同一个 `v2-intercept-event-u50-s6101.pt`、C350 Hit／Recover、原始 v4 bank、seed 9524、128 env，只切换 Launch 观测视图。两次初始 drone state、ball position／velocity 和 active 数组逐元素一致；第一个 4 维动作 L2 差为 0。两组均为 113/128 合法首触、114/114 PhysX 核验合法 cap、1/128 合法第二拍、三连／五连 0。安全失败 metric 从 20/128 变为 19/128；后段动作仅小幅分化。

| Launch 观察 | 首触 | 合法第二拍 | 安全失败 | 墙事件（均经 PhysX 核验） | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `relative_v2` | 113/128 | 1/128 | 20/128 | 44/44 | 未通过 |
| `relative_v3` | 113/128 | 1/128 | 19/128 | 49/49 | 未通过 |

结果表明：对该旧 actor 单独修正相对速度坐标没有带来首击或延续改善。完整报告和 trajectory SHA 记录在 `analysis.json`；逐案 report 与 NPZ 保存在 `observation-view-ablation/`。

## AeroWallLateralInterceptV1：单技能训练 pilot

新增 AeroWall 自有训练分布配置 `configs/wall_training_distributions/aerowall-lateral-intercept-v1.json`。它只改变训练 reset 的 ball `y` 和 `vy` 范围，分别为 `[-0.85, 0.85] m`、`[-0.70, 0.70] m/s`，每次独立均匀采样；其他 RALLY train 初始状态范围不变，不从 v4 case bank 复制状态。训练 actor 用 `relative_v3`、`legacy` reward，seed 6101，128 env、50 PPO updates、409,600 frames；warmstart 是实验 Intercept checkpoint，Hit 与 Recover 均冻结为 C350。

训练完成后 17 个 warmstart actor 张量均原样匹配，未补零、未跳过；权重已更新，冻结 Launch／Hit／Recover 源精确保持，完整 chain checkpoint 重载精确。standalone actor 导出保留 HCSP 要求的 `TensorDictParams` 容器，并由后续真实 MAPPO 评估加载验证。actor checkpoint SHA-256 为 `c50fb2658639922c72d23d528181e60a75e8d22e0af4066812d627459ac3198b`。

使用原始 v4 开发挑战 bank、seed 9524、128 env，Launch 替换为新 actor，Hit／Recover 仍为 C350。结果为 110/128 合法首触、15/128 合法第二拍、平均 rallies 0.1172、最大 1、三连／五连 0；125/125 cap 与 110/110 wall events 经 PhysX 核验。安全失败为 38/128，其中非法接触 36、无人机触地 2。相同 bank／seed 下，旧 actor 的 `relative_v3` 观测结果是 113/128 首触、1/128 第二拍、安全失败 19/128。pilot 有续打信号，但安全显著退步，gate 未通过；v4 已用于开发，不能作为新冻结最终测试集。

第一次 standalone actor 导出把 `TensorDictParams` 降成普通 `TensorDict`，MAPPO 加载失败且没有启动仿真。导出逻辑已改为复用原容器并逐张量复制；随后完整 v4 评估通过，所有结果均来自修复后的 actor。

| 方案 | 合法首触 | 合法第二拍 | 安全失败 | 三连／五连 | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 旧 Intercept + `relative_v3` | 113/128 | 1/128 | 19/128 | 0% / 0% | 未通过 |
| `AeroWallLateralInterceptV1`，legacy reward | 110/128 | 15/128 | 38/128 | 0% / 0% | 未通过 |

legacy、aerowall_causal_v3、legacy actor LR=1e-5、aerowall_causal_v4、aerowall_causal_v5 和 aerowall_causal_v6 的单变量 Intercept pilots 均未通过正式 gate。CausalV5 为 118/128 首触、12/128 第二拍、6/128 三连、22/128 安全失败；CausalV6 为 110/128 首触、7/128 第二拍、0 三连、39/128 安全失败。V6 的 phase 0/1/2 非法接触为 17/2/16，且 phase-0 惩罚加重后安全反退。相较正式 C350 留出基线，两者均不安全，候选不晋升，正式路由保持 C350。下一步先做轨迹和动作诊断。

### `aerowall_causal_v3` 单变量 reward 对照

在相同 seed 6101、warmstart、训练分布、观测、50 updates、冻结 Hit／Recover 和 actor LR=1e-4 下，仅将 Intercept 训练 reward 从 legacy 改为 `aerowall_causal_v3`。训练与导出完整性检查通过。原始 v4 开发挑战、seed 9524、128 env 评估得到 100/128 首触、28/128 合法第二拍、58/128 安全失败（54 非法接触、3 无人机触地、1 撞墙），1/128 三连、0 五连；141/141 cap 和 126/126 wall events 经 PhysX 审计。相较 legacy/LR=1e-4，续打提高但首触降低、安全失败增加，gate 仍失败，不晋升。causal-v3 与 legacy eval reward 下轨迹 SHA 相同，证明评测 reward 配置不改变确定性策略执行。

以上 legacy/LR=1e-5 对照及后续 CausalV4–V6 phase 奖励单变量 pilots 均已完成；V6 结果表明单纯提高 phase 0 惩罚无效。当前下一项是分析已存轨迹、动作与接触几何，明确新的可证伪低层控制假设，再决定是否开展单变量 pilot。

### AeroWall causal-v4 safety pilot（完成，gate 未通过）

复用 `aerowall_causal_v3` 的所有奖励项，只对首次合法 cap 前的 phase 0 非法接触增加 -30，使该阶段总惩罚为 -40，phase 1/2 仍为 -10。候选 `AeroWallLateralInterceptV1-CausalV4` 使用 seed 6101、128 env、50 updates、LR 1e-4、同 warmstart／分布／relative_v3 观测及冻结 C350 Hit／Recover。第一次启动因 stats schema 缺 key 而在 0 updates 失败；修正后完整训练通过。

原始 v4 开发挑战、seed 9524、128 env：111/128 首触、15/128 合法第二拍、46/128 安全失败（43 非法接触、3 无人机触地），最大回合 3，三连 2/128、五连 0/128。132/132 合法 cap 和 130/130 wall events 经 PhysX 核验。gate=false，不晋升。phase 0/1/2 非法接触为 17/0/26，phase 2 是主要剩余失败来源。报告、轨迹、接触审计和 SHA256 见 Part 1–14 进度文档。

### AeroWall causal-v5 safety pilot（完成，gate 未通过）

候选 `AeroWallLateralInterceptV1-CausalV5` 仅在 CausalV4 上对 phase 2 非法接触额外增加 -30，使 phase 2 总惩罚为 -40；phase 0 仍为 -40、phase 1 为 -10。固定 seed 6101、128 env、50 updates、LR 1e-4、同 warmstart／分布／relative_v3 和冻结 C350 Hit／Recover。训练 50 updates／409,600 frames 完成；17 个 warmstart actor 张量精确匹配，冻结技能、checkpoint reload 和 actor export 均通过。

原始 v4 开发挑战：118/128 首触、12/128 合法第二拍、6/128 三连、0/128 五连、最大回合 4；安全失败 22/128（20 非法接触、1 无人机触地、1 撞墙）。146/146 合法 cap 与 123/123 wall events 经 PhysX 核验。终止非法接触 phase 0/1/2 为 10/1/9，20/20 事件与终止 step 匹配。相较 V4 安全失败由 46 降至 22；但正式 C350 v4 留出基线为 4/128 安全失败，候选违反安全门槛。三连增加但第二拍下降，五连仍为零；gate=false，不晋升。产物路径与 SHA256 见 Part 1–14 进度表。

### AeroWall causal-v6 safety pilot（完成，gate 未通过）

AeroWallLateralInterceptV1-CausalV6 复用 CausalV5 的 phase 0/1/2 惩罚（-40/-10/-40），只额外将 phase 0 惩罚增加 -40，使 phase 0 总惩罚为 -80；其余 warmstart、分布、seed、updates、LR、观测和冻结 C350 技能均相同。50 updates／409,600 frames 完成；17 个 warmstart actor 张量精确匹配，checkpoint reload、冻结 Launch／Hit／Recover 和 actor export 均精确。

原始 v4 heldout bank、seed 9524、128 env：110/128 首触、7/128 合法第二拍、0 三连／五连，最大 1 回合；安全失败 39/128（35 非法接触、3 无人机触地、1 撞墙；另有 65 球落地、24 越界）。118/118 cap 和 110/110 wall event 经 PhysX 核验。35/35 终止非法接触的最后 body event 与 policy_steps - 1 对齐，phase 0/1/2 为 17/2/16，均为非预期接触。评估 status=passed 但能力 gate=false。相较 CausalV5 首触 -8、第二拍 -5、安全失败 +17、非法接触 +15，最大连续回合从 4 降至 1；正式 C350 安全失败基线为 4/128，不晋升。

逐案比较 V5→V6 有 22 个 ball-ground 转为 illegal-contact（10 起在 phase 0，12 起在 phase 2），仅 5 个 illegal-contact 转为 ball-ground。新增非法碰撞横跨首次拦截与回接；phase 0 惩罚加重仍未降低整体安全失败，当前证据不支持继续单纯放大奖励惩罚。下一步先用保存的轨迹、动作、状态和接触事件定位可验证的故障机制。

训练报告 SHA256 27075469eff680493f3c3cfef825ad529eb6d5a400cc5e99a05c4228e373e564；全量 checkpoint 2e61ddf3914a21513d724ac43e8b1ba03312401dbb5590337ae2ade52135a3be；actor export 165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b。评估报告 042b6123f07b26f30e9f82a1bbde8f55e9d4a0c222aa9e96b3342ffef151c518；trajectory 208dce368068251ed9f32efefd019a102b69fcf3dcf7cacfe9498e14523a076e；events cf3ef21ec21ba80a2dca7457343a3ad62b6a9ba090096ecfcee43ffdc5cfc918；contacts 0b307abef6945e757b5456c5ccaadc9f813bfcf323e5747c6eed2169fefae097。Bank SHA256 666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80。

### AeroWallLaunchTanhMappingAblationV1（同 checkpoint 映射消融）

针对 V6 轨迹显示的动作分量限幅，固定 V6 Launch actor、C350 Hit／Recover、seed 9524、同一 v4 128-case bank、relative_v3 环境观测和 legacy eval reward，只将 Launch actor 的分布切换为 HCSP 已有的 `TanhNormalWithEntropy(tanh_loc=True)`。AeroWall 自有消融编排命名为 `AeroWallLaunchTanhMappingAblationV1`；没有复制或重命名 HCSP 的分布实现。Hit／Recover 保留 HCSP 默认 `IndependentNormal`。不训练、不晋升。

同 bank V6 默认映射对照为 110/128 合法首触、7/128 合法第二拍、最多 1 连、39/128 安全失败（35 非法接触），动作分量限幅率 30.56%。Launch Tanh 映射结果为 119/128 合法首触、9/128 合法第二拍、最多 2 连、29/128 安全失败（29 非法接触），动作分量总体限幅率 17.68%。技能分解确认 Launch 3416 步、最大 raw action 0.964、限幅率 0%；Hit 1274 步、限幅率 5.47%；Recover 7172 步、限幅率 28.27%。总限幅剩余部分来自未修改的 Hit／Recover。128 个逐案 outcomes 和 PhysX contact audit 与同一 Tanh 配置的无技能轨迹复跑完全一致；129/129 合法接触和 110/110 wall events 均经 PhysX 核验。

这是正向单变量信号，但三连／五连仍为 0，P2 gate=false；安全失败仍高于正式 C350 heldout 的 4/128。正式策略不晋升。基于此信号，随后启动单独命名的 AeroWall 有界动作训练 pilot，见下节。

机器报告 SHA256 `83aee9bf660ad8f8182e1bd8ed7bdcb8dd0819ed05077d83d98f9be7879e2af0`；`AeroWallActuatorTraceV2` 轨迹 SHA256 `a8f13ace5579876ac33c29b3244a51cbe6c7f6875e550b6f2db88ed3fbbc13f3`；events `a933bc0655eb762ac3fe608ca7a39ce591e9ad4a8d6b7faab2cc375bd99536fb`；contacts `985cdc3c6847b84784f79eee99a50063ab3a93af1ea9dbc01850bcd1e4490841`。四个产物均在 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/`，文件名为 `aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-tanhmapping-skilltrace-s9524` 加相应扩展名。

### AeroWallBoundedInterceptV1-CausalV6（有界动作训练 pilot，gate 未通过）

基于上节同 checkpoint Tanh 映射信号，训练独立候选 `AeroWallBoundedInterceptV1-CausalV6`。复用 V6 warmstart、`AeroWallLateralInterceptV1` 分布、CausalV6 reward、seed 6101、128 env、50 updates／409,600 frames 和冻结 C350 Hit／Recover，只将 trainable Intercept actor 设为 HCSP `TanhNormalWithEntropy(tanh_loc=True)`。HCSP 源未改；训练正常完成，actor 权重有变化，17 个 warmstart 张量精确，checkpoint reload、冻结技能源和独立 actor 导出均通过。

使用原始 v4 开发 bank、seed 9524、128 env 评估：116/128 合法首触、14/128 合法第二拍、3/128 三连、2/128 五连、最大 8 连；安全失败 27/128（23 非法接触、3 无人机撞墙、1 无人机触地）。146/146 合法接触、143/143 墙面事件经 PhysX 核验，128/128 outcomes 审计通过。总动作分量限幅率 20.54%，Launch／Hit／Recover 分别为 0%／6.04%／32.00%。相较 V6 默认映射，第二拍由 7 增至 14、安全失败由 39 降至 27；相较 Tanh-only 映射消融，出现 3 个三连样本和 2 个五连样本，但五连率仅 1.56%。RALLY 三连／五连 gate 要求 80%，两项 gate 均失败；安全失败也高于正式 C350 的 4/128，候选不晋升，正式路由继续使用 C350。

将每案最后活动策略步与终止原因配对后发现：91 次球落地、9 次越界和 4 次无人机触地／撞墙在冻结 Recover 步骤终止；23 次非法接触中，14 次在冻结 Hit、8 次在新 Intercept actor、1 次在冻结 Recover 终止。此统计定位 episode 终止所在技能，不足以单独证明该技能导致失败。与 Tanh-only 消融逐案配对时，20 个 case 不再出现安全失败，18 个 case 新出现安全失败；3 个达到三连的 case（heldout-0004、0075、0083）均为候选新增。剩余失败横跨多个技能阶段。

首轮评估报告的运行指标正确，但旧评估器把候选标签记成 `AeroWallLaunchTanhMappingAblationV1`。已修正评估器：可从训练报告读取 AeroWall 候选名，并校验训练报告 SHA256 与独立 actor checkpoint SHA256。修正后的身份核验复跑输出 `AeroWallBoundedInterceptV1-CausalV6`，训练报告／checkpoint 均与评估输入精确绑定；它与首轮运行的 128 个 outcomes、summary、contact audit 及 trajectory 完全一致。旧报告保留为首次运行记录，身份版报告作为正式引用。

训练报告 SHA256 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`；完整 checkpoint SHA256 `35cfb1cd4cdde021d0ec05316c0033426e986fffe5b8fc4268fd8ac155b97121`；standalone actor SHA256 `bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50`。身份版评估报告 SHA256 `832a0147e7c55cb69bd8eb26b79062f1e687d4903027a4a988922486ca5fbed1`；trajectory SHA256 `e36a81304c684fd117cd10ea1f4927d211a09a742f2b701c37ca217ec0f4129b`；events SHA256 `cbdcf73a035729d121f240187372cd7d4e3b9c292a2e5ae5b0e3a12e5b9d708a`；contacts SHA256 `c8b38541bb994ec61bed9b41ded4a81516343bdd512219a82292a9f360a05c0a`。产物均位于 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/`。

### AeroWallHitTanhMappingAblationV1（冻结 Hit 分布消融，未改善 gate）

固定 `AeroWallBoundedInterceptV1-CausalV6` candidate actor、C350 Hit／Recover checkpoint、seed 9524、相同 128-case v4 bank、relative_v3 环境和 legacy eval reward；只将冻结 Hit actor 的分布从 HCSP 默认 `IndependentNormal` 切换到 HCSP 已有 `TanhNormalWithEntropy(tanh_loc=True)`。Launch Tanh candidate 与默认 Recover 均保持不变。AeroWall 自有消融名为 `AeroWallHitTanhMappingAblationV1`；HCSP 分布及上游源码均保持原名和原样。

映射把 Hit 动作分量限幅从 6.04% 降至 0%，但非法接触仍为 23/128、安全失败仍为 27/128；116/128 首触及 14/128 第二拍不变。三连／五连从 3/128、2/128 降至 2/128、1/128，最大回合从 8 降至 5。Recover 动作限幅为 33.17%，总限幅率 20.29%。139/139 合法接触、136/136 墙面事件经 PhysX 核验，128/128 outcomes 的审计完整；评估 status=passed 但能力 gate=false。只消除 Hit 输出限幅未改善整体结果。

报告 SHA256 `3350a215853200fba14c95f1c80cc75bb5872fdb7f4b885d42a336aa58b2af2d`；trajectory SHA256 `5ddd1203e76715890caf35ef73c2df22ebf28a70daa0af69e5d5a99bc15ca2`；events SHA256 `504f466a3be281f923f84febb5525b86ab5f57f08448a585159d7f3c685385d6`；contacts SHA256 `b8b02bc2245aa88cfbcbe5ab0567030c37165004511df13bc8b66b609444a539`。候选训练报告 SHA256 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`，评估器亦核验独立 actor SHA256 `bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50`。

### AeroWallRecoverTanhMappingAblationV1（冻结 Recover 分布消融，未改善 gate）

保持同一 candidate／bank／seed／观察／reward，仅把冻结 Recover actor 改用 HCSP `TanhNormalWithEntropy(tanh_loc=True)`；Launch candidate 的 Tanh 与 C350 Hit 默认分布均保持不变。该 AeroWall 自有单变量消融针对 Recover 的 33% 限幅和 Recover 阶段的球落地／越界事件；不训练、不晋升，HCSP 分布实现与源码未改动。

Recover 限幅从 32.00% 降至 0%，总体限幅率从 20.54% 降至 0.75%；球落地由 91 降至 84、越界由 9 降至 8。与此同时安全失败从 27/128 升至 35/128（26 非法接触、5 无人机撞墙、4 无人机触地），第二拍由 14/128 降至 11/128；三连／五连仍为 3/128 和 2/128，最大回合 9。143/143 合法接触及 138/138 wall events 经 PhysX 核验，128/128 outcomes audit 完整；RALLY gate=false。这说明消除 Recover 动作限幅没有带来安全或能力改善，不晋升。

报告 SHA256 `9046276eca609f0f30ff535733ccafb96041c4f07a45a64b868f728289c416df`；trajectory SHA256 `e2bd55fd49abe276fd8b200751550596490812ef28dbe8de3dd33ef2a4b7115f`；events SHA256 `da2ee5b1b9fc140939ba5162ea38f2b690264b7dd5eff9b7dbba25ef73d95b8f`；contacts SHA256 `a9c58a65aee27ec2f06e721594e89e2fc5b2bc3f970d54530edb493e7ba84ed8`。候选训练报告 SHA256 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`，standalone actor SHA256 `bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50`。

### AeroWallGoalHitPilotT5-S6201 自然 RALLY 整链集成诊断（gate 未通过）

固定 `AeroWallBoundedInterceptV1-CausalV6` Intercept actor、seed 9524、同一原始 v4 128-case 开发 bank、`relative_v3` 观测与 C350 Recover；只把 Hit actor 替换为已有的 `AeroWallGoalHitPilotT5-S6201`。这是自然完整 RALLY 评估，不训练或晋升；评估器报告中记录了 candidate 与 Hit 候选的训练报告、checkpoint 身份和 SHA256。HCSP 上游策略及分布类名称没有改写。

| 指标 | 同 Intercept＋C350 Hit | Intercept＋T5 Hit | 变化 |
| --- | ---: | ---: | ---: |
| 合法首触 | 116/128 | 116/128 | 持平 |
| 合法第二击 | 14/128 | 14/128 | 持平 |
| 三连样本 | 3/128 | 5/128 | +2 |
| 五连样本 | 2/128 | 4/128 | +2 |
| 最大连拍 | 8 | 9 | +1 |
| 安全失败 | 27/128 | 29/128 | +2 |
| 平均 rallies | 0.2344 | 0.3984 | 上升 |
| 墙目标命中率 | 0.3776 | 0.3522 | 下降 |

T5 Hit 组的安全失败为 25 次非法接触、2 次无人机触地和 2 次无人机撞墙；球落地 86 次、越界 9 次、无失败结局 4 个。167/167 合法接触和 159/159 墙面事件经 PhysX 核验，128/128 outcomes 审计完整。总动作分量限幅率 19.85%，Launch／Hit／Recover 分别 0%／8.42%／29.63%。三连率 5/128、五连率 4/128，gate=false。相比同一 Intercept actor＋C350 Hit，T5 Hit 有有限的高连拍信号，但安全失败增加 2 个、目标墙命中率略低；这项单种子整链比较不足以把差异单独归因于 Hit actor，不晋升 T5。

身份绑定：Intercept 训练报告 SHA256 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`、actor SHA256 `bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50`；T5 Hit 训练报告 SHA256 `90d088b9cc5515ed8e46df963668822197815669f0f960ba126cfe2efc15664b`、checkpoint SHA256 `d6a4c52ffb71d295d29e4ce3917b28cb232f8253df4f3ae12d50e4d97abc763c`。评估报告 SHA256 `f40bb520bc00da064bfd7228ac4574eec312e438f64e90a4d61d5558142374c2`；trajectory SHA256 `c2013f43912c32b2e1f292f0b6fa40f89285a8ec9de5953772cb1c796607222e`；events SHA256 `c16dae06483059081fd300c2d3fe48f0b6117298c7253f03ca309855b8c2895e`；contacts SHA256 `cb3e8c5bad43dc5707378ba75a44f9e531d9a6ede7132e39f9618b5aac859d0f`。所有产物保存在 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/`。

### AeroWallBoundedGoalHitV1-Tanh 三种子自然 RALLY 诊断（训练完成，gate 未通过）

**训练状态更正：** S6201 首次运行在用户要求先做无训练消融后，于保存 update 20 状态时有意停止。报告中的 teardown TypeError 与 timeline 关闭发生在同一时刻，是停止时的仿真关闭异常，不代表独立训练失败。保存的 policy、两个 optimizer、CUDA RNG、配置 SHA 和 warmstart SHA 完整；同配置恢复后完成至 update 50／409,600 frames，训练报告 `status=passed`、权重已变化、checkpoint reload exact。S6202、S6203 分别完成独立的 50 updates／409,600 frames，三份训练报告均通过完整性检查。

S6201 第一次评估错误地把 48 维 Hit checkpoint 传给通用 46 维 `--checkpoint`，加载阶段 shape mismatch，未执行 case。该失败报告保留但不用于策略结论。移除通用参数、只通过 `--hit-checkpoint` 加载 Hit，并提供 Hit 训练报告完成 retry1；retry1 是权威报告。

三种子都采用单独的 AeroWall 名称 `AeroWallBoundedGoalHitV1-Tanh-S6201/6202/6203`。Hit 使用 HCSP 已有的 `TanhNormalWithEntropy(tanh_loc=True)`，该分布类和上游实现未修改。自然整链固定同一 `AeroWallBoundedInterceptV1-CausalV6` Launch、C350 Recover、`relative_v3` 环境、`aerowall_causal_v6` reward、128-case 原始 v4 开发 bank 和评估 seed 9524。bank SHA256 为 `666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`。该 bank 是开发 challenge 集，不是独立冻结最终测试集。

| 方案 | 合法首触 | 合法第二击 | ≥3 连拍 | ≥5 连拍 | 安全失败 | 非法接触 | 墙目标命中率 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 同条件 C350 Hit 基线 | 116/128 | 14/128 | 3/128 | 2/128 | 27/128 | 23/128 | 0.3776 | 未通过 |
| S6201 | 116/128 | 14/128 | 5/128 | 4/128 | 28/128 | 23/128 | 0.3671 | 未通过 |
| S6202 | 116/128 | 14/128 | 5/128 | 3/128 | 29/128 | 25/128 | 0.4204 | 未通过 |
| S6203 | 116/128 | 14/128 | 5/128 | 3/128 | 29/128 | 24/128 | 0.3484 | 未通过 |
| 三种子描述均值 | 116/128 | 14/128 | 5/128 | 3.33/128 | 28.67/128 | 24/128 | 0.3786 | 未通过 |

逐案与同条件 C350 Hit 基线比较，三个 seed 各有 7 个 case rallies 增加、0 个下降；三连样本均净增 2 个，五连样本分别净增 2、1、1 个，没有失去原有三连／五连 case。合法第二击仍为 14/128，没有增加。安全失败则分别净增 1、2、2 个。`heldout-0023`、`heldout-0079` 和 `heldout-0117` 在三个 seed 中都从非安全终止变成非法接触；S6202 另在 `heldout-0083` 出现非法接触。此处报告的是相同开发 case 的重复配对诊断，不能外推为独立测试可靠性。

动作追踪显示三种子 Hit 动作分量限幅均为 0%，Recover 限幅仍为 29.96%–30.33%，总限幅约 18.89%–19.11%。整体安全失败和非法接触没有改善。每份评估的 outcome 接触审计完整；合计 489/489 legal events、470/470 wall events 经 PhysX 核验，callback errors 均为空。训练与评估进程通过不代表策略 gate 通过：三种子的三连／五连能力门槛均未过，安全失败也高于同条件 C350 Hit 的 27/128；不晋升，正式 C350 不变。

| Seed | checkpoint SHA256 | training report SHA256 | natural RALLY report SHA256 |
| --- | --- | --- | --- |
| 6201 | `d970228e9c2fdc662fc6fa9b0aead0672628ffd0da947dd04a93685a31968dd7` | `f223a9fd72260f1683edd4371fa32e865b891c51e16a1444374cfd33f686f630` | `e90ea3fb933a6b696747a6f402abc9b4d65b6ecc80f92f4c5b72b2ab8a558e4c` |
| 6202 | `bcb58eb4b1101533e0b9eb2ec3fc812828d3297ac76dd225ef609a223d5a85b7` | `cf656aff93c5150b9e439962f066279a51b1cbc06b0bcda102a7107a7f48ede3` | `417496f8710a76d3a6fac14f5bb38198fc5a3316b3700b96f42bc5470eed9d9b` |
| 6203 | `d8b7a01efb5d9abc8c0b33f849ba4d8a908da697c27fec1b74acdfd74b7d53a6` | `ca1c10c125640356aba68e477c14f061182bdd7896fc96df97ccd9a88dfd5b9e` | `8d48ecf70d99be4d7c2448c3bdd10f1d3d139e942a43d6a325bcd83f89bb68f7` |

下一步先对重复非法接触 case 的完整轨迹、接触几何和对应状态／动作做根因诊断，再提出可证伪的单变量 pilot。P3–P6 保持 gated；不能把本三种子开发 bank 诊断当作计划中的正式最终验收。

### 非法接触阶段归因

从正确配置评测的事件记录中，每个 failure=4 case 的最后 contact event 都与终止 policy step 对齐；全部是非预期球—机身接触。按 phase 0/1/2（首次合法 cap 前／出球至墙／碰墙后回接）统计：legacy/LR=1e-4 为 18/1/17，causal-v3/LR=1e-4 为 28/3/23，legacy/LR=1e-5 为 51/2/17，causal-v4 为 17/0/26，causal-v5 为 10/1/9，causal-v6 为 17/2/16。CausalV6 的 phase 0 惩罚升至 -80 后 phase 0 非法接触反升，整体非法接触也从 V5 的 20 增至 35；停止单纯放大奖励惩罚，先做轨迹与动作层诊断。

### legacy actor LR=1e-5 学习率对照

复用 `AeroWallLateralInterceptV1`、相同 warmstart／seed／分布／updates／legacy reward 与 C350 冻结技能，只将 actor LR 从 1e-4 改为 1e-5。50 updates／409,600 frames 完成，17 个 warmstart actor 张量精确匹配，冻结源、checkpoint reload 和 standalone actor export 均通过。

原始 v4 开发挑战、seed 9524、128 env：77/128 合法首触、13/128 合法第二拍、70/128 安全失败（70 非法接触），最大回合 2，三连／五连均为 0；93/93 记录到的 legal-cap contacts 与 84/84 wall events 经 PhysX 核验。相比 legacy/LR=1e-4 的 110/128 首触、15/128 第二拍、38/128 安全失败，降 LR 未改善安全或续打，候选不晋升。第一次评估遗漏环境级 `--observation-version relative_v3`，使 FSM 未推进；该执行记录保留但不纳入比较。修正后评估 `skill_step_counts=[3400,1299,4641]`，确认三技能链生效。

事件阶段归因显示 phase 0（首次合法 cap 前）非法接触分别为 legacy/LR=1e-4 的 18/36、causal-v3/LR=1e-4 的 28/54、legacy/LR=1e-5 的 51/70、CausalV4 的 17/43、CausalV5 的 10/20、CausalV6 的 17/35；phase 2 回接阶段依次为 17/36、23/54、17/70、26/43、9/20、16/35。CausalV6 的 phase 0 惩罚升至 -80 后并未减少 phase-0 非法接触，整体安全也更差；停止惩罚升级，改为轨迹与动作诊断。正式 C350 路由保持不变。

## 可复核产物

完整报告、事件／接触审计、A/B 轨迹数组、训练 checkpoint 和训练状态保存在被 Git 忽略的 `artifacts/wall-skill-upgrade-v3/`。主要入口：

- [T5 训练报告](../artifacts/wall-skill-upgrade-v3/t5-aerowall-goal-hit-pilot-s6201.json)
- [T5 Hit checkpoint](../artifacts/wall-skill-upgrade-v3/t5-aerowall-goal-hit-pilot-s6201.pt)
- [fixed 评测报告](../artifacts/wall-skill-upgrade-v3/t5-hit-fixed-s6201.json)
- [training 分布评测报告](../artifacts/wall-skill-upgrade-v3/t5-hit-train-s6201.json)
- [v4 留出评测报告](../artifacts/wall-skill-upgrade-v3/t5-hit-heldout-v4-s6201.json)
- [C350 v4 基线报告](../artifacts/wall-skill-upgrade-v3/t5-baseline-heldout-v4.json)
- [AeroWallBoundedInterceptV1-CausalV6 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-s6101.json)
- [AeroWallBoundedInterceptV1-CausalV6 standalone actor](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-s6101.launch.pt)
- [AeroWallBoundedInterceptV1-CausalV6 身份核验 v4 评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-v4-relv3-legacyeval-identity-verified-s9524.json)
- [AeroWallBoundedInterceptV1-CausalV6 v4 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-v4-relv3-legacyeval-identity-verified-s9524.trajectory.npz)
- [AeroWallBoundedInterceptV1-CausalV6 v4 contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-v4-relv3-legacyeval-identity-verified-s9524.contacts.json)
- [AeroWallBoundedInterceptV1-CausalV6＋AeroWallGoalHitPilotT5-S6201 自然 RALLY 评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-t5-goalhit-s6201-v4-relv3-legacyeval-s9524.json)
- [T5 Hit 整链 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-t5-goalhit-s6201-v4-relv3-legacyeval-s9524.trajectory.npz)
- [T5 Hit 整链 PhysX contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-t5-goalhit-s6201-v4-relv3-legacyeval-s9524.contacts.json)
- [AeroWallHitTanhMappingAblationV1 评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-hit-tanh-mapping-v4-relv3-legacyeval-s9524.json)
- [AeroWallRecoverTanhMappingAblationV1 评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-recover-tanh-mapping-v4-relv3-legacyeval-s9524.json)
- [修正后的交接来源评测](../artifacts/wall-skill-upgrade-v3/handoff-source-b-train-128-s123-fsm-enabled.json)
- [52 状态 handoff bank](../artifacts/wall-skill-upgrade-v3/handoff-source-b-train-128-s123-fsm-enabled.bank.json)
- [C350 同状态条件基线](../artifacts/wall-skill-upgrade-v3/handoff-c350-hit-baseline-s123.json)
- [T5 pilot 同状态条件评测](../artifacts/wall-skill-upgrade-v3/handoff-t5-hit-pilot-s6201-s123.json)
- [Isaac reset 字段级审计](../artifacts/wall-skill-upgrade-v3/handoff-reset-audit-fsm-enabled.json)
- [v4 Launch 侧向反事实诊断汇总与哈希](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)
- [原始 v4 全轨迹重跑报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/recorded-trajectories/v4-original-128.json)
- [原始 v4 全轨迹 NPZ](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/recorded-trajectories/v4-original-128.trajectory.npz)
- [vy 翻转全轨迹重跑报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/recorded-trajectories/v4-flip-vy-sign-128.json)
- [vy 翻转全轨迹 NPZ](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/recorded-trajectories/v4-flip-vy-sign-128.trajectory.npz)
- [relative_v2／relative_v3 消融汇总](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)
- [AeroWallLateralInterceptV1 分布配置](../configs/wall_training_distributions/aerowall-lateral-intercept-v1.json)
- [AeroWallLateralInterceptV1 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-s6101.json)
- [AeroWallLateralInterceptV1 actor checkpoint](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-s6101.launch.pt)
- [AeroWallLateralInterceptV1 v4 评估报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-v4-s9524.json)
- [AeroWallLateralInterceptV1 v4 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-v4-s9524.trajectory.npz)
- [legacy/LR=1e-5 Intercept training report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-s6101.json)
- [legacy/LR=1e-5 standalone actor](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-s6101.launch.pt)
- [legacy/LR=1e-5 v4 evaluation, relative_v3](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-relv3-s9524.json)
- [legacy/LR=1e-5 superseded evaluation with observation mismatch](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-s9524.json)
- [legacy/LR=1e-5 v4 contacts](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-relv3-s9524.contacts.json)

本次没有提交或推送改动。2026-09-25 更正已复现 116/128 首触结果并定位先前 16/128 结果来自 Launch 观测错配；下一步分析正确配置下重复非法接触及 Recover 阶段回接失败，不启动新的训练种子。

### AeroWallBoundedGoalHitV1-Tanh-CausalV6-S6201 与复现性审计（2026-09-25）

> 本节最初记录的 16/128 fresh 对照与“root cause 未知”判断已被文末 2026-09-25 更正 supersede；正确 Launch 观测版本和配对结果见更正章节。原始 reports 保留供审计。

- 训练：seed 6201、50 updates／409,600 frames，唯一变化为 Hit 训练 reward `legacy` → `aerowall_causal_v6`；候选名为 `AeroWallBoundedGoalHitV1-Tanh-CausalV6-S6201`。training report SHA256 `b868c586da8078fe88efdd6ea3b8b7b02e7cfdd804e648aa7d858e990c748dd4`，checkpoint SHA256 `09f40d3c06407bf1c8afd0419d0de60fb38635832137c3467c0291034e57d395`。
- 评估配对固定 V6 bounded Intercept Launch、C350 Recover、Tanh Launch／Hit、同一 `heldout-128-v4` bank（SHA256 `666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`）、seed 9524、relative_v3 公共观测及 aerowall_goal_v1 Hit 观测。2 次 Legacy Hit fresh control 和 CausalV6 Hit candidate 均为 16/128 首触、0 第二拍、0 连拍、115/128 安全失败（全部非法接触，另 13 球落地）；gate=false。PhysX 核验 16/16 legal contacts、13/13 wall events，128/128 outcome audits 完整。训练 reward pilot 不显示优势，不扩 seed、不晋升。
- 两次 fresh legacy rerun 与 CausalV6 Hit candidate reset 物理/FSM 状态 hash 均为 `c81363f3d8860bb682b1a6a8296b23f218d8883e8a4ef1b12a5b5b4f2d8ec9e9`。Legacy control 两次的初始所有技能 actor observation/action、采样诊断轨迹与全量 trajectory 完全相同；两个 trajectory SHA256 都是 `a399377aac7d2e99c6492e91871e02be92524035cbcd50c1d09ce92a1a2a4510`。CausalV6 Hit 与 legacy Hit 的初始 actor 输入/action 相同；首个动作分歧发生于 4 个 Hit invocation（env 29/31/81/86，policy step 92/89/88/88），在分歧前 trajectory 相同，128 个 case 的 rallies、failure 和 policy_steps 分类完全一致。
- 较早 S6201 legacy 报告当时没有记录分角色 actor input、post-reset 物理状态和源码 SHA，因此首步动作来源未能直接审计。后续按训练报告指定的 `relative_v3` 重跑得到 116/128 首触、14/128 第二击、28/128 安全失败，完整 trajectory SHA256 与历史成功报告一致；该历史能力结果已复现。另一个 16/128 fresh run 的 Launch observation mismatch 是独立问题，原因已确认并标记 superseded；所有原始报告仍保留作审计记录。
- evaluator 新增 opt-in `--record-reproducibility`，记录 post-reset 物理/FSM 状态、CPU/CUDA/NumPy/Python RNG 指纹、源码和输入文件哈希、每角色 observation version、全部环境首个实际 actor input/action，以及 env 0 采样步的 actor 输入。两次 legacy rerun 的 Python RNG 状态指纹不同，但 Torch CPU/CUDA、NumPy、物理/FSM reset、初始 actor 输入和完整轨迹一致；没有证据显示 Python RNG 差异影响本组轨迹。
- 三次新 instrumented eval 均先写出 `status=passed` 报告并完成 PhysX audit，然后 Isaac Sim teardown 打印 native segmentation fault。此为 app.close() 退出期问题；评估产物完整，运行日志保留，报告仍不能作为能力 gate 通过证据。

| 复现产物 | SHA256 |
| --- | --- |
| legacy control 1 report | `8182524037da33274036fe16cfb0d3e14d75a400e00f2adc9bbd753fea794abc` |
| legacy control 2 report | `765c823f8cdf2edbec1fa07ab0676e0f09804d48a74dfe75e538c09c7b7e8fe1` |
| CausalV6 Hit report | `92c4a27475d12794605f7b04153333e555f8767999737af76d1458fe3c701efb` |
| CausalV6 Hit trajectory | `717bf82068fcc9ffd9cf4e01ac0f27f76fe66db36322fe700424ee1aa1344d7b` |
| instrumented evaluator source | `e74809901f9240f96983da30f26939afe81cdf14ea6c97e4899b7635eaf4ec8a` |

- [CausalV6 Hit instrumented report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/causal-v6-hit-repro-s9524.json)
- [CausalV6 Hit instrumented trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/causal-v6-hit-repro-s9524.trajectory.npz)
- [Legacy fresh control reproduction 1](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/legacy-control-repro1-s9524.json)
- [Legacy fresh control reproduction 2](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/legacy-control-repro2-s9524.json)

### 更正：Launch 观测版本与 Recover TanhNormal 消融（2026-09-25）

审计证明此前 fresh control 与 CausalV6 Hit reward pilot 把 V6 Launch actor 的评测观测设为 `legacy`；训练报告和训练时实际观测为 `relative_v3`。这些 16/128 首触、0 第二拍、115/128 安全失败结果保留为历史记录，但标记为 observation-mismatch superseded，不用于判断 checkpoint 能力或 reward 效果。评测器现在会从训练报告继承观测版本、拒绝未声明的不匹配，并记录每角色训练／评测观测和动作分布。只有显式 `--allow-observation-version-ablation` 才能运行观测错配消融。

使用 `relative_v3` 重跑 legacy Hit control 后，首触 116/128、合法第二击 14/128、三连 5/128、五连 4/128、安全失败 28/128。其完整 trajectory SHA256 `bf0f2aee9e7f065acad26b2121680487995ad01dfb58702ccc6e41dca1401c06` 与历史成功报告逐字节一致。正确配对的 CausalV6 Hit reward candidate 为 116/128 首触、14/128 第二击、5/128 三连、3/128 五连、安全失败 27/128；逐案没有回合数增加、2 案下降，reward pilot 不提升回合能力，也不晋升。

用户批准先做 Tanh 映射消融。校正后的动作追踪显示 V6 Launch 和 Tanh Hit 均无执行器裁剪；约 29.96% 的 Recover/C350 原始动作分量被限幅，整链约 19.01%。`AeroWallRecoverTanhMappingAblationV1` 固定 Launch、Hit、Recover checkpoint、heldout bank、seed 和其余观测／奖励／FSM 设置，唯一改变是将冻结 Recover 从 HCSP 默认 `IndependentNormal` 改为未修改的上游 `TanhNormalWithEntropy(tanh_loc=True)`。不训练。

| 指标 | 默认 Recover 映射 | Recover TanhNormal |
| --- | ---: | ---: |
| 合法首触 | 116/128 | 116/128 |
| 合法第二击 | 14/128 | 11/128 |
| 三连 case | 5/128 | 3/128 |
| 五连 case | 4/128 | 2/128 |
| 平均 rallies | 0.3750 | 0.2109 |
| 安全失败 | 28/128 | 36/128 |
| Recover 动作分量裁剪 | 29.958% | 0% |
| 总动作分量裁剪 | 19.006% | 0% |

配对初始 physics/FSM state SHA256 相同（`c81363f3d8860bb682b1a6a8296b23f218d8883e8a4ef1b12a5b5b4f2d8ec9e9`）；116 个执行到 Recover 的 case 在第一次 Recover 动作前的动作轨迹逐步相同，另 12 个在进入 Recover 前结束且活动 rollout 前缀相同。128 个 case 中 9 个 rallies 下降、119 个持平、0 个增加；安全失败新增 9 案、消失 1 案。控制组 164/164 合法接触和 158/158 墙事件、Tanh 组 143/143 和 138/138 均经 PhysX 核验；两组 128/128 outcomes audit 完整。P2 gate 仍失败，不训练有界动作候选、不晋升，正式 C350 不变。

| 产物 | SHA256 |
| --- | --- |
| Recover Tanh report | `8956a272c56736411a2944d7b9e4d3ad99859003517893a7a92b963ddf4721e1` |
| Recover Tanh trajectory | `b4d30024ded2ce6e29384e2a29680d391ff2bfbbebb00633f50b26099aa4ed67` |
| Recover Tanh events | `60a85afd8bf9c727da37941c1a232f26cefb70e7a80588daca8517dce9b3962f` |
| Recover Tanh contact audit | `ef757a432dc0b75788911bc9bc4e99446496e041a9065d1f276c6bb8c69c19b4` |
| Evaluator with observation guard | `cfb6e79b69a76e8b3a3f4d62a64cbcf160459511fc76960ff9da3082b854be10` |

Artifacts live under `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/`:

- [Recover TanhNormal report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-recover-tanh-mapping-ablation-v1-s9524.json)
- [Recover TanhNormal trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-recover-tanh-mapping-ablation-v1-s9524.trajectory.npz)
- [Recover TanhNormal events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-recover-tanh-mapping-ablation-v1-s9524.events.jsonl)
- [Recover TanhNormal PhysX contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-recover-tanh-mapping-ablation-v1-s9524.contacts.json)

### Recover 配对轨迹与非法接触阶段诊断（2026-09-25）

按 case 对齐默认 Recover 与 TanhNormal 的 trajectory。两边初始 physics/FSM hash 相同；116 个抵达 Recover 的 case 都在第一次 Recover 动作处出现首个动作差异，之前的动作轨迹完全相同；另外 12 个未进入 Recover 的 case，其活动轨迹前缀一致。结果为 9 个 case rallies 减少、119 个持平、0 个增加。安全失败由 28/128 增至 36/128；9 个 case 从非安全失败转为安全失败（`heldout-0026/0037/0044/0053/0060/0083/0091/0119/0124`），`heldout-0006` 从无人机撞墙转为球落地。

对每个 `illegal_contact` outcome 取最后一次 body-contact event，并读取该物理事件发生时 FSM `skill_id`；该字段不等于产生动作的 actor，后者由 action-aligned actor observation role 决定。phase 0 首次接触前非法碰撞两组均为相同 12 个 case；FSM skill_id 为 Launch 8／Hit 4，但 `caps==0` 时实际 policy route 均为 Launch actor（12/12）；phase 2 回球接触前非法碰撞由 11 增至 14（默认映射时 Hit 10、Recover 1；Tanh 映射时 Hit 6、Recover 8）；Tanh 组另有 1 个 phase 1、由 Recover 执行的非法接触。各非法事件的事件回调径向误差均超过环境 `racket_radius=0.20 m`，支持“球体在拍面平面内偏离有效半径”作为直接失败描述，但不能据此判定更上层根因。phase 0 径向误差中位数为 0.215 m；phase 2 对照为 0.292 m、Tanh 为 0.329 m。

HCSP `Serve` task 配置中 `ball_radius=0.10 m`，环境几何要求轴向距离在 0–0.20 m。以 contact substep 在前后策略帧之间线性插值 drone position、球面插值 attitude 后，phase 0 的 12 案轴向距离均为 0.148–0.160 m，phase 2 control 的 11 案均为 0.109–0.160 m；Tanh phase 2 为 12/14 落在轴向区间，另 2 案位于拍面后侧。对重复 case `heldout-0023/0079/0117`，插值局部径向／轴向距离依次为 control `0.422/0.109`、`0.320/0.160`、`0.243/0.153 m`，Tanh `0.248/0.155`、`0.284/0.151`、`0.264/0.152 m`。插值径向值相对事件回调精确 radial error 的平均绝对差不超过 `0.00016 m`；轴向仍是近似估计。三个重复 case 的主要共同症状是球体越出拍面径向有效半径。

下一步保持正式 C350 与 checkpoint 不变，分别检查首次接触和墙后回球两个阶段；优先重放 `heldout-0023/0079/0117`，记录接触前 drone／ball 位姿、姿态、径向和轴向距离、相对速度、当前 actor role 及 motor action。证据足够后再设计可证伪的单变量诊断；不启动 bounded-action 训练候选、不晋升，P2 gate 仍失败。

对照报告 SHA256 `0cf75b7c7ef0868ad894619c3cc0e1f824ae6af85a059859f1f88f2661479265`，对照 trajectory SHA256 `bf0f2aee9e7f065acad26b2121680487995ad01dfb58702ccc6e41dca1401c06`，对照 events SHA256 `f644bb9d8729081530ae61b720c4b50ef5bd6da162a079a606d0400fd155a5cf`。Tanh report／trajectory／events SHA256 已列于本节上方。

### Exact contact-substep failure replay（2026-09-25）

针对预先选出的 33 个失败 case，用相同 checkpoint 分别重跑默认 Recover 与 Tanh Recover。仅新增 evaluator 事件遥测；未训练、未晋升，正式 C350 和策略链保持原状。事件现在精确记录物理子步时的球／无人机位姿和速度、FSM phase、实际 actor、policy action、motor throttle 与 per-rotor thrust。

| 组别 | 全量结局复现 | 终止非法接触 | PhysX 合法／墙面核验 |
| --- | --- | ---: | --- |
| 默认 Recover | 33 案的 failure/rallies/reason 均复现；仅 heldout-0124 终止步差 1 | 23 | 41/41、37/37 |
| Tanh Recover | 33/33 outcome 和 policy steps 一致 | 27 | 29/29、27/27 |

两组所有终止非法接触均径向超出 0.20 m 球拍半径；Tanh 组另有 heldout-0025、heldout-0037 两次落到拍面后侧。phase 0 的 12 个非法接触在两组中为相同 case；FSM skill_id 为 Launch 8／Hit 4，但 `caps==0` 使实际 active actor 全是 Launch。接触状态、动作和 active actor observation 逐字段相同；local-y 都落在负侧（-0.269 至 -0.113 m，中位 -0.204 m）。21 个合法首触的 local-y 中位数为 -0.093 m；Launch 输入中相对球位置特征 y 中位数从合法首触 -0.116 m 移到非法首触 -0.208 m。这说明存在同侧偏差的可检验症状，尚未证明根因。33 案为失败定向子集，不用于推断总体率。

下一步把失败与相近的合法首触按 actor 和来球条件配对，核查方向偏差来自初始状态／观测还是动作响应；phase 2 Recover 失败分开分析。仅在提出可证伪的单变量假设后再做不训练的干预评测。P2 gate=false，正式 C350 保持，不晋升，不推进 P3–P6。

产物及 SHA256 收录于 [analysis.json](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)，两组完整报告、trajectory、events、PhysX contacts 和 log 均位于 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/`。

### Event-time active actor observation audit（2026-09-25）

新增 evaluator 记录把每条 contact/wall event 对齐到产生该动作的 actor observation，包含 role、观测版本与实际输入向量。control 101/101、Tanh 83/83 事件均有记录；role/version 为 Launch `relative_v3` 46D、Hit `aerowall_goal_v1` 48D、Recover `legacy` 46D。两组 trajectory SHA256 与 contact-substep-only run 相同，说明观测记录未改变策略输出。

该记录纠正 phase 0 actor 归因：`executed_skill_id` 表示 FSM skill，不代表动作来源。首个合法 cap 前策略路由由 `caps==0` 决定，因此 12 个 phase 0 非法接触全部来自 Launch actor，其中 FSM skill_id 计数 Launch 8、Hit 4。phase 2 active role 与技能一致：control 的 11 案为 Hit 10／Recover 1，Tanh 的 14 案为 Hit 6／Recover 8；Tanh 另有 1 案 phase 1 Recover。

同一失败定向 33 案包含 21 个合法首触和 12 个非法首触。球拍 local-y 中位数分别为 -0.093 m 和 -0.204 m；Launch `relative_v3` 输入特征 26–28 的球相对位置中位数分别为 `[0.035,-0.116,0.227]` 和 `[0.047,-0.208,0.226]` m。两组事件均由 Launch actor 执行动作。该结果支持继续检查来球位置输入和接触前动作响应；33 案经过失败富集，不能用于估计总体率，也未证明因果。

- [control actor-input 报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-control-runtime-s9524.json)
- [control actor-input events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-control-runtime-s9524.events.jsonl)
- [Tanh actor-input 报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-tanh-runtime-s9524.json)
- [Tanh actor-input events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-tanh-runtime-s9524.events.jsonl)
- [机器分析与 SHA256](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)

新增 evaluator SHA256 `915d8ef817a983bb338b5b4cba839d3b5ac736866601546647acae802de9b31d`。下一步固定 active Launch actor，配对检查相近来球条件下的合法／非法首触输入、动作和接触前状态；phase 2 按 Hit／Recover actor 分开分析。不训练、不晋升，C350 保持，P2 gate=false。

### 全量 phase 0 接近段与 phase 2 actor-action 对齐（2026-09-25）

- 新增 `scripts/analyze_aerowall_fullbank_contact_approach.py`，基于已经保存的全量控制组／Recover-Tanh 128-case reports、events、trajectories 与 PhysX contacts 重算 phase-0 首次 body contact；接触位姿按相邻策略帧线性位置插值、WXYZ quaternion slerp，再与 callback 球位置对齐。两组 case 和合法标签完全一致，均为 116 个合法、12 个非法。失败组中 9/12 最后动作前 radial 已 >0.20 m，另 3 个在最后动作周期越界。插值 radial 相对 callback 最大误差 0.000347 m。
- 在离最后一次动作 0.40 s 的 ball world position／velocity 上，用 pooled-SD 标准化欧氏距离作 12 对 12 最小总成本匹配；7 对距离 ≤1、11 对 ≤1.5。非法减合法的 local-y 配对中位差在距最后动作 0.40/0.30/0.20/0.10/0.04/0 s 时为 -0.144/-0.094/-0.043/-0.028/-0.044/-0.057 m；world relative-y 从 -0.022 m 扩大至 -0.077 m，失败组 drone world-vy 在最早 0.4–0.1 s 约高 +0.12–0.14 m/s（即向负 y 运动更慢）。这是同一个开发 bank 上的状态关联，无法区分初态影响与较早控制响应。
- 对既有 failure-enriched 33 案 phase-2 events，最早非法接触 observation 的 ball position、ball-minus-drone world position 与动作前 trajectory 完全对齐；event policy action 与策略 trajectory action 也完全一致。control 的 11 案为 Hit 10／Recover 1，Tanh 的 14 案为 Hit 6／Recover 8；两条件共享 11 案，另有 3 个 Tanh-only 案。control/Tanh 最后动作前径向偏差中位数分别 0.309/0.318 m。
- 更正 action-path（2026-09-25）：AeroWall evaluator 的 `TransformedEnv` 只加入 `InitTracker`，四维 actor 输出直接进入 `drone.apply_action`。HCSP `RotorGroup.forward` 限幅各路命令、用平方根映射生成目标油门并应用电机滞后；这条 wall-rally 路径没有 `PIDRateController_flightmare`。之前将 AeroWall 轨迹解码为机体角速度／推力目标的分析不适用，已按实际 rotor-command 路径重算。
- 当前决定：Recover-only Tanh 消融未提升正式能力／安全指标，P2 gate=false；正式 C350 不变。当前不训练、不晋升、不创建冻结最终测试集。Launch 横向速度状态干预已完成所选 12 对筛查，但不证明 actor 能通过 rotor action 产生同一速度修正；其中两案球越过 world-y ±3 m。下一步预注册并执行一个无训练的直接 rotor-command response screen，逐案记录四路动作效应、接触结果和球越界终点。
- 可复算脚本 SHA256 `937630f3158c740741a3505cb3711b1df4810cf0332613e0f433705effd44f52`；机器报告 SHA256 `ae741d65616330242625e4ce84fcadfc87f37a5f655f3ff016b486e1afff075e`，路径为 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-fullbank-contact-approach-alignment-v1-s9524.json`。汇总已写入 `analysis.json` 的 `fullbank_contact_approach_alignment_v1`。

### Phase 0 pre-action / exact-contact alignment（2026-09-25）

Added `scripts/analyze_aerowall_contact_alignment.py` to replay the existing 33-case control and Recover-Tanh telemetry as a read-only, hash-backed analysis. It selects the first phase-0 body contact per case and aligns its active actor input to the state before the action, then aligns the callback action to the saved action at that policy step. The active Launch observation position matched the reconstructed pre-action state within `7.3e-8 m`; event action matched the saved action exactly. Contact occurred 2.5–20 ms after the action started (median 12.5 ms).

| First-contact label | Cases | Pre-action / contact radial median | Pre-action / contact local-y median | Already outside 0.20 m before final action |
| --- | ---: | ---: | ---: | ---: |
| Legal | 21 | 0.130 / 0.112 m | -0.116 / -0.093 m | 1 |
| Illegal | 12 | 0.220 / 0.215 m | -0.208 / -0.204 m | 9 |

The other three illegal cases (`heldout-0012`, `heldout-0051`, and `heldout-0099`) crossed the radial threshold during that final control interval. Eight of 12 failures had pre-action local-y below -0.18 m, compared with 1 of 21 legal contacts. The illegal-vs-legal local-y median difference was already visible 0.2–0.4 s before the final action; the trajectories partly converge and then diverge again, so this does not identify whether initial case conditions or earlier Launch behavior caused the miss. The 33 cases were selected to cover failure modes and are not a population sample. Control and Recover-Tanh phase-0 rows were identical, so they are not independent replications.

Next: trace Launch actions and lateral error through the earlier 0.2–0.4 s approach, match legal/illegal cases on incoming height and velocity, and analyze phase 2 by the actual Hit/Recover actor. If the sample lacks overlap, record that limitation. Do not train or promote; C350 remains formal and P2 remains false.

Report: [phase-0 alignment JSON](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-phase0-preaction-contact-alignment-v1-s9524.json), SHA256 `4b27a61ca51afbd38ca738ed199726303189f791f0081337628b3508b521eb85`. Script SHA256 `deae56adaceed93629d53756c7d600be84d56a1ff1fd352a445624219a1321dc`.

### Phase-2 shared-case contact comparison（2026-09-25）

Compared the earliest phase-2 illegal body-contact event in each diagnostic case. All 11 control failures also occurred under Recover-Tanh; Tanh had three additional phase-2 cases (`heldout-0037/0053/0124`) and no control-only case. It also had one phase-1 Recover illegal contact (`heldout-0083`).

Active actor roles were Hit 10 / Recover 1 in control and Hit 6 / Recover 8 in Tanh. Six of the 11 shared failures switched from Hit to Recover. For those 11 cases, Tanh-minus-control radial error had median `+0.015 m` (range `-0.173` to `+0.107 m`): six increased and five decreased. The phase-2 sample shows mixed case-level effects. It does not identify one uniform Recover mapping mechanism; the full 128-case safety failure increase from 28 to 36 remains the paired outcome for the mapping ablation.

Next: align action-time Hit/Recover inputs to exact phase-2 contact substeps, focusing on the six role-switch cases and three Tanh-only failures. Keep this separate from tracing the phase-0 Launch approach. No training or promotion; formal C350 remains and P2 is false. Summary and input event hashes are in `analysis.json` at `phase2_shared_case_actor_contact_comparison_v1`.

### AeroWallLaunchInterceptTargetRetentionV1 ablation (2026-09-25)

Added the independently named observation view `aerowall_intercept_target_retention_v1`. During phase 0, it retains the bounded predicted racket-height intercept in Launch observation features 0–2 when crossing time and court bounds are valid, even if the conservative drone-reachability check fails. It leaves features 3–45, the feasibility flag, checkpoints, action distributions, FSM, rewards, critic view, and Hit/Recover views unchanged.

The inference-only run used seed 9524 and the same 128-case v4 development bank, V6 Launch actor, Tanh Hit actor, C350 Recover actor, and evaluation settings as control. Both reports passed and all 128 outcomes were contact-audited. Initial actor views matched for all 128 reset cases. Later Launch actions diverged in 14 cases, with 714 of 13,547 shared-active action rows differing. First divergence occurred at trajectory indices 14–33.

Both routes had 116/128 legal first contacts, 12/128 illegal contacts, 14/128 legal second hits, 5/128 three-rally episodes, and 4/128 five-rally episodes. Mean rallies stayed 0.375. Safety failures rose from 28 to 29; `heldout-0060` changed from ball-ground termination in control to the environment world-x < 0.5 m task threshold in the ablation. This threshold is not evidence of physical wall impact; neither PhysX contact sidecar records a drone/single_wall contact. No previously unsafe case recovered. P2 remains false, so this is retained as a negative inference-only result; no training, promotion, or formal-route change follows.

For `heldout-0060`, the first action divergence was at trajectory index 16 (maximum component difference 0.1334). Both routes made legal phase-0 contact at policy step 36/substep 3; the ablation had the smaller radial error (0.0341 m vs 0.0635 m). At the first wall event, step 45, the ball y positions were 1.1849 m (ablation) and 1.0887 m (control). The ablation first crossed x < 0.5 m at trajectory index 75 (x=0.4958 m); control minimum active x was 0.5111 m and ended by ball ground. This is a descriptive paired timeline, not a causal explanation.

The custom implementation is in `aerowall/wall_rl/observations.py`, `aerowall/wall_rl/trajectory.py`, and `scripts/aerowall_wall_rally_env.py`, with routing in `scripts/aerowall_skill_policies.py` and the evaluator. Analysis script SHA256 `c5ff64d9fade88056525da96b2491b93b708a656f20eb3bd55b3cc0faca2b170`. Paired analysis report SHA256 `b1a1af915143f89c44d5543321b5318fbbfd30aab44ce2c25190b784138f11a1`; raw run report SHA256 `345c9968e6b58d8b562d9bc7ea0915e7377d0f032f383f72e482e08a5b4b0380`. Full trajectories, event records, PhysX contacts, logs, and hashes are stored beside those reports under `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/`.

### AeroWallPhaseSpecificFailureModeSynthesisV1 (2026-09-25)

Added a read-only synthesis over existing seed-9524 reports. The analyzer verified ten source artifact hashes and rechecked state equality immediately before each of the 14 first Launch action divergences. Across all 14, recorded drone state, ball position/velocity, actuator command, motor throttle, and rotor thrust match bitwise before the changed action. This verifies an immediate observation-to-action response on common recorded states; the paired full rollout still shows no contact-label or rally gain and one additional safety failure.

The evidence separates three routes: (1) phase-0 first-contact labels stay 116 legal / 12 illegal across control and Recover-Tanh, with all 12 illegal contacts routed through Launch and nine already beyond 0.20 m radial error before the final action; (2) the selected phase-1 replay has one Recover-attributed event and is too small to estimate a phase rate; (3) phase-2 failures are heterogeneous, with 11 shared and three Tanh-only misses and mixed paired radial changes. The 128-case Recover-Tanh mapping removes clipping but lowers second/three/five-rally counts and raises safety failures from 28 to 36. The active AeroWall path uses direct four-rotor commands. Its re-decoded paired command differences vary across approach lead and channel, so these data do not justify a fixed rotor-command offset or another trained candidate.

Keep C350 formal and P2 false; P3-P6 remain gated. A future policy change requires a named phase-specific mechanism and pre-registered paired safety/ability rejection criteria. No simulation, training, or promotion was performed for this synthesis.

Reproducibility: [synthesis note](aerowall-phase-specific-failure-synthesis-v1.md), [JSON report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-phase-specific-failure-mode-synthesis-v1-s9524.json), script SHA256 `15665229b3f24b3774148be996b6ca50c8233e093d128568e6832f7d00052241`, report SHA256 `d591ae0cdcba0e27e7423dafd40178f85dce0cfeb1a35353e6ff16d02aa9b57f`.

### AeroWallRecoverHandoffBankV1：真实 Recover 交接与条件续跑（2026-09-25）

扩展 `evaluate_aerowall_wall_rl.py` 的 handoff 契约，保留原 Hit state v1 和 `aerowall_hit_handoff_v1` 银行；Recover state 使用 AeroWall 专项 schema v2，显式保存 `handoff_actor_role=recover`，要求 `skill_id=2`，并复用同一套完整物理／FSM 字段和 PhysX 审计 cap→wall 前缀。冻结来源为 seed 9524、`heldout-128-v4` 开发 bank、V6 Launch、Tanh Hit 与 C350 Recover；不训练、不改 checkpoint。

128 案中 114 案实际到达 Recover；114/114 候选均通过状态与来源事件前缀校验，0 拒收。来源 natural RALLY 报告 SHA256 `5982e67f6a7773a91236ed4b4e6ad07b2f7e1cecaf01101c50579f53a9645bab`；合法 cap 164/164、墙事件 158/158 由 PhysX 核验，callback errors 为 0。Recover bank SHA256 `539d8999aa91f232b86ee58d852e8d0cc8ce5c5e4e159d3691cc1307381a0f4c`。

在 114 个状态上重建 Isaac scene 并逐项审计 20 个 reset 字段：所有误差不超过 `1e-4`；最大为 drone／ball position 的 `1.9073486328125e-6`，其余 18 个字段均为 0。reset 报告 SHA256 `a3137d1d28e4d8d1a83b2a8406b5e2a24190e9224e8dbb9a4e1562f1c3755da5`。

从相同 bank 做默认 C350 Recover 与 `AeroWallRecoverTanhMappingAblationV1` 续跑：

| 指标（114 个 Recover 状态） | 默认 Recover | Tanh Recover |
| --- | ---: | ---: |
| handoff 后首次合法 cap | 14/114 | 14/114 |
| 平均 handoff 后 rallies | 0.4123 | 0.2807 |
| 至少 3 连 | 5/114 | 4/114 |
| 至少 5 连 | 3/114 | 2/114 |
| 安全失败（drone ground／illegal contact／drone wall） | 18/114 | 21/114 |

逐案 rallies：107 持平、7 下降、0 上升；6 案新增安全失败、3 案解除，安全失败净增 3。两组均保留 228 个 source-prefix events 的 PhysX 来源审计；新增接触均通过 PhysX：默认组 47/47 legal cap、41/41 wall，Tanh 组 32/32、29/29，callback errors 均为 0。配对分析 JSON SHA256 `05da068f3410866bcc2f45e320db94549911c430f54ec066aa836e191a45a6df`。

这是按默认 C350 Recover 成功到达阶段后做的条件续跑，且来自已用于开发的 v4 challenge bank；结果不能替代自然整链 P2 gate，也不是独立冻结测试集。负向结果不晋升 Tanh，不改变正式 C350；P2=false，P3–P6 继续 gated。

产物：[Recover source report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-source-v1-s9524.json)、[114-case Recover bank](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-bank-v1-s9524.json)、[reset audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-reset-audit-v1-s9524.json)、[default continuation](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-continuation-control-v1-s9524.json)、[Tanh continuation](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-continuation-tanh-v1-s9524.json)、[paired summary](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-continuation-comparison-v1-s9524.json)。配对汇总 SHA256 `ceacfa37791e142712a47d389c8377c6661b0757cbe1686c4005e0ba76dcd49f`。复算入口为 `scripts/analyze_aerowall_recover_handoff_continuation.py`，SHA256 `5c134b93ee88e56b58b50cada5932c836e4623e7833e4e676cd522b56b79fef4`。第一次用仅供 inventory 的 `scripts/python.sh --plain` 启动模拟，因无 PyTorch 在模拟前退出；失败报告保存在 `reproducibility/failed-attempts/`，不纳入评估证据。
