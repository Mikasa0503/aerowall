# Part 1–14 执行进度

**更新日期：** 2026-09-25

**目标：** 按实施计划完成 Part 1–14，并保留可核对的阶段进度。

**当前结论：** Part 1–3、11–12 首轮分析与工程落地已完成；Part 4–10、13 的能力验证或条件工作仍未闭环，Part 14 能力验收未通过。近期 Recover Tanh、Launch 截点目标保留、三种子 Hit、Tanh Launch 独立复核和后接触球线速度恢复均未通过 P2。RotorCommandResponse 局部 screen 仍没有任务级能力提升。新同 bank V6 Launch-Tanh 可复核配对将限幅率 29.96% 降至 16.19%、首触 110/128 升至 119/128、安全失败 39 降至 33；第二拍与三连未提升，未过预注册任务级门槛。Launch-Tanh 的 13 案预注册速度恢复准确改变了接触后的状态，但将 12 个非法接触结局转成球落地失败：全 bank 非法接触 47→35、球落地 67→79；合法第二击及回合分布不变，单变量恢复 gate 失败。没有训练或晋升，正式 C350 保持，P2=false，P3–P6 gated。

**项目目录：** 本仓库根目录。旧副本迁移与核验已完成；冲突备份属于被忽略的本地工作区内容，不纳入项目分发。

## 状态定义

- **完成**：该 Part 中要求的分析、代码或记录已有直接证据；若设计要求保留条件门槛，也以此为准。
- **部分完成**：已有实现或实验，但至少一个明确验收条件仍未满足。
- **条件待启动**：计划规定须先通过前置门槛；当前证据不支持跳过门槛启动。
- **未开始**：尚无对应实现或实验。

## Part 清单

| Part | 状态 | 已有证据 | 未闭环工作／判定 |
|---|---|---|---|
| 1. 当前 RL 系统与数据流 | **完成** | 主线、控制频率、状态／观测、PPO、事件、reset 和终止协议已分析并写入计划。 | 后续若改变实现，需同步更新架构描述。 |
| 2. 问题、证据与修复优先级 | **完成（首轮问题）** | 坐标系、采样／更新观测一致性、FSM 读副作用、reset 隔离、奖励归属、有限墙和保存恢复接口均已实现修复；纯逻辑、张量与真实 HCSP 测试有记录。 | 若后续诊断发现新的故障，追加根因与证据，不覆盖原报告。 |
| 3. HCSP 迁移边界 | **完成** | 上游 `MAPPOPolicy`、物理和控制保留；AeroWall 自有环境、策略编排、观测、奖励与入口使用 AeroWall 名称；旧入口有兼容 shim。 | 无。 |
| 4. 最小 Skill Library | **部分完成** | Launch／Hit／Recover 串联、参数化 `aerowall_goal_v1` Hit 与冻结技能已实现；52 个真实 Hit handoff 与新增 114 个真实 Recover handoff 均通过 20 项状态 reset 审计。新增有界动作 Intercept 与独立命名的 `AeroWallBoundedGoalHitV1-Tanh-S6201/6202/6203` 三种子自然整链诊断，所有接触均经 PhysX 核验。 | 三个训练种子在同一 v4 开发 bank 上均为 116/128 首触、14/128 第二击、5/128 三连；五连为 4、3、3/128，安全失败为 28、29、29/128，均未过能力／安全 gate。三个种子重复出现 heldout-0023、0079、0117 非法接触。独立冻结最终测试集仍未建立。 |
| 5. Action Space | **部分完成** | PRT 主线、旧 actor 兼容和 AeroWallActuatorTraceV2/V3 的 raw／限幅 rotor command、throttle、推力追踪已保留；当前 evaluator 的 actor 输出直接作为四路 rotor command 进入 HCSP RotorGroup。Tanh 映射消融、AeroWallRotorCommandResponseCounterfactualV1 及新的 128 案 Launch-Tanh 独立复核已完成；新复核限幅率 30.18%→15.77%，但第二击 3→2、安全失败 44→47；13 案首触后对齐显示接触球速与 Recover 动作有配对差异；后续预注册 13 案 post-cap ball-linear-velocity restore 中，12 个非法接触改为 ball-ground failure，全 bank ball-ground 67→79、illegal contact 47→35；legal second hit 与 rally histogram 不变，严格 gate 失败。 | rotor-1/-2/-3 的负向 0.10 干预均在 12/12 案把 step-9 world-y 速度推向匹配参考；rotor-1/-2 分别新增 3/2 案非法接触，只有 rotor-3 信号未触发本 screen 的安全拒绝条件，但首触仍 3/12、平均 rallies 为 0。Recover Tanh 虽消除约 30% 裁剪，安全失败仍上升；velocity restore 只改变失败类别、没有回合收益。同 bank 可复核 Launch-Tanh 复测确认限幅、首触和总安全失败有改善，但第二击为 5→5、三连 0→0，未达到训练触发条件；已有 AeroWallBoundedInterceptV1-CausalV6 候选及评估，不重复训练。CTBR 未做，P2 未通过。 |
| 6. 因果 Reward 与尺度 | **部分完成** | 有限墙、有效预测、奖励版本与分项路径已实现；历史 reward SHA 保持不变。 | `legacy` 与 `aerowall_causal_v3` 的同种子单变量训练对照已完成（LR=1e-4）：causal-v3 的第二拍由 15/128 增至 28/128，但首触从 110/128 降至 100/128，安全失败从 38/128 增至 58/128；两者都未过 gate，不能认定 causal-v3 整体更优。legacy actor LR=1e-5 对照也已完成：首触与第二拍均下降，安全失败为 70/128。后续 CausalV4/V5/V6 的单变量惩罚试验均未过 gate；CausalV6 首触 110/128、第二拍 7/128、安全失败 39/128，phase 0 惩罚加至 -80 后安全反退。正确观测版本下的 CausalV6 Hit reward pilot 与 legacy Hit control 均为 116/128 首触、14/128 第二拍、5/128 三连；五连为 3 vs 4，安全失败为 27 vs 28，没有回合能力提升，不支持 reward 晋升。先前 16/128 的配对结果因 Launch 观测错配已 supersede。 |
| 7. Policy Chaining 与交接契约 | **部分完成** | A/B 同权重包装、52 个 Hit、114 个 Recover handoff 及 Launch／Recover reset 审计均有证据。新 9 臂动作 screen 的 post-reset state hash 相同，control 第一步 source-action 最大误差 2.4e-7；每个处理臂完整执行 120 条命令记录，12/12 首触有 PhysX corroboration。 | 新 screen 只覆盖 12 个定向 development 状态。rotor-3 负向速度信号未带来首触／rally 提升，不证明一般化可控性；Tanh Launch 复核显示分布映射会改变首触后球速及下一步 Recover 动作；单变量恢复试验改变下一步 Recovery 输入，但 12 个 illegal-contact 改为 ball-ground failure、没有救回回合；C350 保持。 |
| 8. 课程与阶段依赖 | **部分完成** | P0 接口修复、P1 无训练对照完成；P2 Hit pilot 与 52 状态配对评估完成。Launch／Hit／Recover Tanh 映射消融、有界动作候选和独立命名的 Tanh Hit S6201/6202/6203 三种子自然整链诊断均完成。 | P2 gate 仍失败：Tanh Hit 种子为 5/128 三连、3–4/128 五连；新的 Recover-only Tanh 消融为 3/128 三连、2/128 五连且安全失败 36/128。v4 是开发 challenge bank，不是最终冻结测试集。同 bank Launch-Tanh 复测未增加第二击或三连；保持 C350；P3–P6 不启动。 |
| 9. FSM → Learned High-Level | **部分完成** | FSM 单次推进、接触优先级、驻留／回滞和 reset 所有权已落实。 | 高层 Actor 与 Semi-MDP buffer 未做；须先证明低层技能可靠且剩余失败主要由技能／目标选择导致。 |
| 10. 共同适应与条件性增强 | **条件待启动** | 已记录 Stage III、域随机化及其他增强的触发条件。 | 低层完整链尚未通过，因此按计划不启动共同适应、随机化或高层学习。触发条件到达后逐项实验。 |
| 11. 文件、类、函数与配置落点 | **完成（首轮范围）** | aerowall/wall_rl、AeroWall 环境／策略／训练／评估／案例入口和配置已落地；Intercept-only 更新、独立 actor 导出与 stats schema 已有运行证据。 | 新增 trainable Tanh 动作分布选择、冻结 Launch／Hit／Recover 分技能分布消融 CLI、AeroWall 有界动作候选元数据，以及评估器的候选身份／训练报告 SHA256 绑定；新增 opt-in `--record-reproducibility`，记录 reset 物理/FSM 状态、RNG 指纹、运行源码／checkpoint 哈希、每个技能角色的实际 actor 输入和动作；HCSP 上游分布类未修改。 |
| 12. 推荐目录与重构范围 | **完成（首轮范围）** | AeroWall 规范文件名已建立，旧 HCSP 风格路径转为兼容入口；未修改的 HCSP 上游保留原名。 | 无。 |
| 13. 实验计划、统计与验收 | **部分完成** | A/B/C、T5、fixed/train/v4、CausalV5/V6、Tanh 消融、三种子评估、Launch 状态反事实和 rotor-command 响应 screen 均有预注册／哈希／PhysX 审计产物。新 screen 完成 control+8 treatment、108 个续跑；三种负向处理通过局部速度阈值，只有 rotor-3 负向未触发预注册安全拒绝条件。 | P2 能力／安全 gate 仍未通过；rotor-3 负向的合法首触仍为 3/12、平均 rallies 0。v4 与新 seed-260925 devrep 均是开发 bank，不是冻结最终测试集。Tanh Launch 的 13 案后接触对齐与 ball-velocity restore 为预注册但定向 development screen；restore 试验因新增 12 个 ball-ground failure 未通过安全门。Rotor 局部 screen 的 Python stdlib RNG fingerprint 跨进程不同；评估器已补 random.seed(args.seed)。新 Tanh 独立复核的 Python、NumPy、Torch、CUDA RNG 与 reset state 均跨臂一致；新增同 bank 复核中三技能 relative_v3、bank/checkpoint/source hashes、初始状态与 RNG 指纹均配对一致，但第二击与三连门槛未通过。不得据局部信号训练或晋升。 |
| 14. 第一轮 MVP 与执行清单 | **T0–T5 已执行；能力验收未通过** | 原 T0–T5、正确 relative_v3 control、三种子 Hit、reward/Tanh/截点目标消融、分 phase 诊断、Launch 状态 screen、AeroWallRotorCommandResponseCounterfactualV1、128 案 Tanh Launch 独立复核、13 案首触后对齐和 13 案 post-cap velocity-restore screen 均有可核对产物；最新机制 screen 的安全 gate 失败。 | Rotor-3 负向干预满足局部速度响应规则但没有任务提升；Launch post-cap ball-velocity restore 把 12 个 illegal contact 替换为 12 个 ball-ground failure，且无回合收益。两者都不替代 P2 整链 gate。同 bank Launch-Tanh 复测通过可复现性与 PhysX 审计，但未增加第二击或三连；不得据此新增训练或晋升，正式 C350 不变，P3–P6 继续 gated。 |

### 2026-09-25：AeroWall Launch 横向动量反事实 screen

- 按预注册协议固定 seed 9524、V6 Launch／Tanh Hit／C350 Recover、`heldout-128-v4` 和 12 对失败／合法参考状态；在预测接触时间 0.40±0.02 秒捕获 24 个快照。自然源轨迹与修正后的历史 control trajectory SHA 完全一致（`bf0f2aee…1401c06`），24/24 handoff 接受且来源接触 callback errors 为 0。配对续跑银行唯一的 case 字段差异是 `drone_velocity[1]`。有一对速度差为负（−0.03164 m/s），仍按预注册“替换为匹配合法案实测值”保留，没有过滤。
- Control 与 intervention 两边各 12/12 的 20 项状态 reset audit 报告均为 passed、所有字段误差为 0；intervention reset-only 进程在报告写盘后于 `SimulationApp.close` 报 native segmentation fault。source→control 首动作最大绝对误差为 2.4e-7。两边最早 body 首触各 12/12 都能对应到同 env／step／substep 的 drone base_link—ball 正 PhysX contact_count；callback errors 均为空，合法 cap 和 wall 汇总全部 corroborate。
- Control 首触 3/12 合法；intervention 6/12，净增 3/12，其中 heldout-0033、0073、0107 从 control 非法转为 intervention 合法。12 对首次径向误差改善（control 减 intervention）的中位数为 +0.027883 m，超过预注册 0.02 m 门槛。按预先指定的 code 2/4/5，未新增无人机触地、非法接触或撞墙失败。另有 out-of-bounds 终止从 0 增至 2（heldout-0033、0107），作为重要 adverse outcome 保留；每案仍以失败结束，不能称为整体任务成功。
- 该结果达到“机制 screen 支持”门槛，但只限于 12 对定向 development 样本；intervention 直接改了状态，尚不能说明策略可通过动作实现同样效果，不能估总体率、通过 P2 或授权训练。另有 heldout-0033、0107 后续越过球 world-y ±3 m 边界而 out-of-bounds；0107 在终止前完成 4 次 rallies。正式 C350 保持，P3–P6 gated。下一项若继续研究，须单独预注册 action-level controllability 与球越界 endpoint。
- 分析产物：`artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-launch-lateral-momentum-counterfactual-v1-analysis-s9524.json`（SHA256 `16c005b4fa625135d9eab0c62aec38dcc4d0b1630920d118f35c59898a7bff5b`）。预注册文件 `docs/plans/2026-09-25-aerowall-launch-lateral-momentum-counterfactual-v1.md`；银行构造与分析入口分别为 `scripts/build_aerowall_launch_lateral_momentum_banks.py`、`scripts/analyze_aerowall_launch_lateral_momentum_counterfactual.py`。

### 2026-09-25：观测版本更正与 Recover TanhNormal 配对消融

- 先前 16/128 fresh control 与 CausalV6 Hit 评估把训练报告声明为 `relative_v3` 的 V6 Launch actor 传入 `legacy` 观测，结果不适用于该 checkpoint；改为 `relative_v3` 后控制组为 116/128 首触、14/128 第二击，并与历史成功报告 trajectory SHA256 `bf0f2aee…1401c06` 完全一致。正确配对的 CausalV6 Hit 为 116/128 首触、14/128 第二击、5/128 三连、3/128 五连、安全失败 27/128；legacy Hit control 为 116、14、5、4 和 28，reward pilot 未改善回合能力。
- `AeroWallRecoverTanhMappingAblationV1` 固定 V6 Launch checkpoint、Tanh Hit checkpoint、C350 Recover checkpoint、seed 9524、128-case `heldout-128-v4` bank、所有观测／奖励／FSM 设置；唯一变量是冻结 Recover 从默认 `IndependentNormal` 切换到 HCSP `TanhNormalWithEntropy(tanh_loc=True)`。reset state hash 相同；116 个执行到 Recover 的 case 在映射切换前动作完全一致，另 12 个在进入 Recover 前结束且 rollout 前缀一致。
- Recover 原始动作裁剪从 29.958% 降至 0%，全策略从 19.006% 降至 0%。合法首触持平 116/128；第二击 14→11，三连 5→3，五连 4→2，平均 rallies 0.375→0.211，安全失败 28→36（新出现 9 案，消失 1 案）。rally 数 9 案下降、119 案持平、0 案增加；P2 gate=false，不训练候选、不晋升。
- 两侧全部 128 个 outcome 都有审计；控制组 164/164 合法接触、158/158 墙面事件，Tanh 组 143/143、138/138 均经 PhysX 核验，callback errors 均为空。旧 fresh 16/128 报告保留为 superseded 历史证据。
- 新评测报告：`artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-recover-tanh-mapping-ablation-v1-s9524.json`（SHA256 `8956a272c56736411a2944d7b9e4d3ad99859003517893a7a92b963ddf4721e1`）；trajectory SHA256 `b4d30024ded2ce6e29384e2a29680d391ff2bfbbebb00633f50b26099aa4ed67`。正式 C350 保持，P3–P6 继续 gated。

### 2026-09-25：Recover 配对轨迹与非法接触阶段诊断

### 2026-09-25：exact contact-substep 遥测复跑

- 仅扩充评估事件遥测，策略权重、HCSP 动作分布、reward、FSM、观测、case 初态和正式 C350 路由均未改。评估回调现在在每个 body/wall contact physics substep 保存 drone 位姿、方向向量、线/角速度、球相对位移、FSM phase、动作前 active actor 的 observation role/version/input、策略动作、throttle 与各旋翼 z 推力。
- 用相同 V6 Launch、Hit 和 C350 Recover checkpoints、seed 9524、`relative_v3` Launch 观测与筛出的 33 个已知失败／代表 case 分别重跑默认 Recover 和 HCSP Tanh Recover。两组通过真实 Isaac Sim／PhysX contact audit。对照组 41/41 legal 与 37/37 wall events 核验；Tanh 组 29/29 与 27/27 核验；callback errors 均为 0。
- 复跑结果与全量 bank 对齐：对照 33 案 failure class、rallies、failure reason 全一致，仅 `heldout-0124` 终止步为 190 vs 全量 191；Tanh 组 33/33 的上述结局字段和 policy steps 均一致。子集按失败模式选出，事件数量不能解释成总体率。
- 精确事件几何：对照 23/23 终止非法接触径向误差大于 0.20 m，轴向全部在 0–0.20 m；Tanh 27/27 径向误差均越过半径，另有 `heldout-0025/0037` 两次落在拍面后侧。phase 0 两组均有相同 12 案；FSM skill_id 计数为 Launch 8、Hit 4，但在首个合法 cap 前（`caps==0`）策略路由实际由 Launch actor 执行 12/12。事件状态、动作和 active actor observation 逐项相同；phase 2 active actor 为对照 Hit 10／Recover 1、Tanh Hit 6／Recover 8，Tanh 另有 1 案 phase 1 Recover。
- 将精确 event 球心偏移按记录的 WXYZ 姿态转到无人机 local frame 后，12 个 phase 0 非法接触的 local-y 为 -0.269 至 -0.113 m、中位 -0.204 m；21 个合法首触为 -0.174 至 0.110 m、中位 -0.093 m。两组事件 active actor 都是 Launch；FSM skill_id 的 8/4 计数不是 actor 分组。Launch relative_v3 输入的球相对位置特征 (26:29) 中位数由合法 `[0.035,-0.116,0.227]` 变为非法 `[0.047,-0.208,0.226]`。姿态重建 up-vector 最大分量误差小于 4.2e-7。样本按失败定向挑选，该关联不是因果或总体率证据。
- 后续 `AeroWallContactSubstepActorInputAuditV1` 为每条事件附上产生该动作时的 active actor observation role/version/vector。策略轨迹与上一轮相同 SHA256，控制组 101/101、Tanh 组 83/83 事件都有观测，角色维度分别为 Launch `relative_v3` 46D、Hit `aerowall_goal_v1` 48D、Recover `legacy` 46D。此核验纠正 FSM／actor 混淆：首次合法 cap 前所有 phase 0 事件都由 Launch actor 执行；“FSM Launch 8／Hit 4”只是 FSM skill_id 统计。
- 同一 33 案中 21 次合法 phase 0 首触 local-y 中位数为 -0.093 m（径向误差中位数 0.112 m）；12 次非法首触为 -0.204 m（径向中位数 0.215 m）。Launch 输入的 relative_v3 球相对位置 (26:29) 中位数为合法 `[0.035,-0.116,0.227]`、非法 `[0.047,-0.208,0.226]`；实际 actor 都是 Launch。该比较为描述性证据，样本按失败筛选，不表示因果或总体率。
- [control actor-input 报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-control-runtime-s9524.json)、[control actor-input events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-control-runtime-s9524.events.jsonl)、[Tanh actor-input 报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-tanh-runtime-s9524.json)、[Tanh actor-input events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-tanh-runtime-s9524.events.jsonl)。
- [对照 33 案报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-audit-v1-control-runtime-s9524.json)、[对照 exact events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-audit-v1-control-runtime-s9524.events.jsonl)、[Tanh 33 案报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-audit-v1-tanh-runtime-s9524.json)、[Tanh exact events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-audit-v1-tanh-runtime-s9524.events.jsonl)、[机器分析与哈希](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)。
- 下一步：在同一 active Launch actor 条件下，对照 local-y 偏差不同的合法与非法首触输入、动作和接触前轨迹；phase 2 分别核对 Hit／Recover actor 的输入和回球状态。只有确认可证伪的单变量假设后再做无训练干预评测。P2 gate=false，不训练、不晋升，P3–P6 继续 gated。


- 对照与 Tanh 组的 reset hash 相同；比较 128 个按 case 对齐的 trajectory，116 个抵达 Recover 的 case 首个实际动作差异都发生在 Recover，第一个 Recover 动作以前的轨迹完全一致；另 12 个在抵达 Recover 前结束且活动 rollout 前缀相同。因此映射变化只从 Recover 动作开始影响这组 rollout。
- 9 个 case 的 rallies 下降、119 个持平、0 个上升。安全失败由 28 增至 36：9 案从非安全失败变为安全失败，1 案从无人机撞墙转为球落地。新增安全失败为 `heldout-0026/0037/0044/0053/0060/0083/0091/0119/0124`，移除的安全失败为 `heldout-0006`。
- 对每案最后一个非法接触事件按 FSM phase 与发生该接触时实际 actor 归因：phase 0 两组均为相同 12 案；FSM skill_id 计数为 Launch 8／Hit 4，但 active actor 因 `caps==0` 实际均为 Launch（12/12）；phase 2 从 11 案增至 14 案；Tanh 组另有 1 案发生在 phase 1。phase 2 事件发生 actor 分布由 Hit 10／Recover 1 变为 Hit 6／Recover 8。所有这些终止非法接触的观测径向误差都超过 0.20 m 球拍半径；phase 0 中位数为 0.215 m，phase 2 对照为 0.292 m、Tanh 为 0.329 m。该证据把高频问题定位到首次拦截和回球阶段的球拍几何对准，不证明单一根因。
- 上游 `Serve` task 的球半径为 0.10 m，因此几何判定的轴向允许区间为 0–0.20 m。接触子步用相邻 trajectory frame 插值后，phase 0 的 12 个接触轴向值均在 0.148–0.160 m，phase 2 对照 11 个均在 0.109–0.160 m；Tanh phase 2 有 12/14 在区间内，另 2 案落到拍面后侧。重复失败 `heldout-0023/0079/0117` 的近似局部径向／轴向偏差（m）分别是对照 0.422/0.109、0.320/0.160、0.243/0.153，Tanh 为 0.248/0.155、0.284/0.151、0.264/0.152。子步插值的径向值与事件回调精确记录的径向值平均绝对差不超过 0.00016 m；轴向值仍为基于 trajectory pose 插值的估计。三个重复案共同越出 0.20 m 拍面半径，轴向大多在范围内，支持径向对准是直接失败症状。
- 下一步保持 C350 和 checkpoint 不变，先补查代表 case `heldout-0023/0079/0117` 的接触前球／机位置、姿态、径向与轴向距离、相对速度、actor role 及 motor action；把 phase 0 首击和 phase 2 回接分开诊断，再基于证据提出单变量非训练或训练 pilot。P2 gate=false，不训练、不晋升。

### `aerowall_causal_v3` reward 单变量对照

固定 seed 6101、warmstart、`AeroWallLateralInterceptV1` 训练分布、50 updates、409,600 frames、relative_v3 观测及 C350 Hit／Recover，只将训练 reward 从 `legacy` 改为 `aerowall_causal_v3`，actor LR 均为 1e-4。训练冻结源、checkpoint reload 和 standalone `TensorDictParams` actor 导出核验通过。

原始 v4 开发挑战、seed 9524、128 env：causal-v3 候选为 100/128 合法首触、28/128 第二拍，平均 rallies 0.2969、最多 3、仅 1/128 达到三连、0 五连；安全失败 58/128（54 非法接触、3 无人机触地、1 撞墙）。141/141 cap 与 126/126 wall events 经 PhysX 审计。相对 legacy 训练候选，第二拍增多，但首触减少、安全失败由 38 增至 58，仍未通过 gate。使用 causal-v3 与 legacy eval reward 评估得到完全相同 trajectory hash，确认评测奖励不影响这组确定性轨迹。候选不晋升，正式 C350 路由不变。

| 训练 reward | 合法首触 | 合法第二拍 | 安全失败 | 三连／五连 | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `legacy`, actor LR=1e-4 | 110/128 | 15/128 | 38/128 | 0/128 / 0/128 | 未通过 |
| `aerowall_causal_v3`, actor LR=1e-4 | 100/128 | 28/128 | 58/128 | 1/128 / 0/128 | 未通过 |
| `legacy`, actor LR=1e-5 | 77/128 | 13/128 | 70/128 | 0/128 / 0/128 | 未通过 |
| `aerowall_causal_v4`, actor LR=1e-4 | 111/128 | 15/128 | 46/128 | 2/128 / 0/128 | 未通过 |
| `aerowall_causal_v5`, actor LR=1e-4 | 118/128 | 12/128 | 22/128 | 6/128 / 0/128 | 未通过 |
| `aerowall_causal_v6`, actor LR=1e-4 | 110/128 | 7/128 | 39/128 | 0/128 / 0/128 | 未通过 |

### AeroWall causal-v4 safety pilot（完成，gate 未通过）

复用 `aerowall_causal_v3` 全部奖励，仅对首次合法 cap 前的 phase 0 非法接触额外增加 -30（该阶段总惩罚 -40；phase 1/2 仍为 -10）。固定 seed 6101、128 env、50 updates、LR 1e-4、同 warmstart／分布／relative_v3 观测及冻结 C350 Hit／Recover。训练完整性核对通过。原始 v4 开发挑战、seed 9524：111/128 首触、15/128 合法第二拍、平均 rallies 0.1641、最大 3、三连 2/128、五连 0/128；安全失败 46/128（43 非法接触、3 无人机触地）。132/132 合法 cap 与 130/130 wall events 经 PhysX 核验。phase 0/1/2 非法接触为 17/0/26。CausalV4 相较 CausalV3 首触增加、安全失败减少，但第二拍下降；相较 legacy/LR=1e-4 安全失败仍更高。gate=false，不晋升，正式 C350 路由不变。

### AeroWall causal-v5 safety pilot（完成，gate 未通过）

仅在 CausalV4 上对 phase 2（合法碰墙后、回接前）的非法接触额外增加 -30，使 phase 2 总惩罚为 -40；phase 0 保持 -40、phase 1 为 -10。固定 seed 6101、128 env、50 updates、LR 1e-4、同 warmstart／分布／relative_v3 和冻结 C350 Hit／Recover。50 updates／409,600 frames 完成；17 个 warmstart 张量精确匹配，冻结来源、checkpoint reload 和 actor export 均精确。

原始 v4 开发挑战、seed 9524：118/128 首触、12/128 合法第二拍、6/128 三连、0/128 五连、最大回合 4；安全失败 22/128（20 非法接触、1 无人机触地、1 撞墙）。146/146 合法 cap 与 123/123 wall events 经 PhysX 核验。20 起终止非法接触的 phase 0/1/2 为 10/1/9，事件步与 policy termination 全部匹配。相较 CausalV4，首触 +7、安全失败 -24、三连 +4，但第二拍 -3；相较正式 C350 原始 v4 留出基线（4/128 安全失败），安全仍退步。gate=false，不晋升。

### AeroWall causal-v6 safety pilot（完成，gate 未通过）

候选 AeroWallLateralInterceptV1-CausalV6 复用 CausalV5 的 phase 0/1/2 惩罚（-40/-10/-40），只额外将 phase 0 非法接触惩罚增加 -40，使 phase 0 总惩罚为 -80。其余 warmstart、分布、seed 6101、128 env、50 updates、LR 1e-4、relative_v3、冻结 C350 Hit／Recover 保持相同。训练完成 50 updates／409,600 frames；17 个 warmstart actor 张量精确匹配、无 padding／skip，checkpoint reload、冻结 Launch／Hit／Recover 和 standalone actor export 均精确。

同一原始 v4 heldout bank（seed 9524，128 env）：110/128 合法首触、7/128 合法第二拍、0/128 三连或五连，最多 1 回合；安全失败 39/128（35 非法接触、3 无人机触地、1 撞墙）。另有 65 次球落地和 24 次越界。118/118 合法 cap 与 110/110 wall events 经 PhysX 核验。35/35 终止非法接触均为非预期球—机身接触；最后事件 policy step 全与 policy_steps - 1 匹配，phase 0/1/2 为 17/2/16。评估进程状态为 passed，指评估完成；能力 gate 为 false。相较 CausalV5，首触 -8、第二拍 -5、安全失败 +17、非法接触 +15，最大连拍从 4 降至 1；正式 C350 heldout 安全失败基线是 4/128，因此不晋升。

同一 bank 逐案转换：V5→V6 有 22 个 ball_ground → illegal_contact（其中 phase 0 为 10、phase 2 为 12）、2 个 ball_ground → drone_ground、1 个 ball_ground → drone_wall；另有 5 个 illegal_contact → ball_ground，10 个非法接触仍为非法接触。新增非法碰撞横跨首次拦截和回接阶段；只加重 phase 0 惩罚仍未降低整体安全失败，不支持继续单纯放大奖励惩罚。

主要产物与哈希：

- 训练报告 SHA256：27075469eff680493f3c3cfef825ad529eb6d5a400cc5e99a05c4228e373e564
- 全量 checkpoint SHA256：2e61ddf3914a21513d724ac43e8b1ba03312401dbb5590337ae2ade52135a3be
- 独立 Launch actor SHA256：165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b
- 训练配置 SHA256：009e825166c710e7ce7fb19084701e238c61b80f2dad264e127da5f8c38b376f
- 评估报告 SHA256：042b6123f07b26f30e9f82a1bbde8f55e9d4a0c222aa9e96b3342ffef151c518
- 轨迹 SHA256：208dce368068251ed9f32efefd019a102b69fcf3dcf7cacfe9498e14523a076e
- 事件 SHA256：cf3ef21ec21ba80a2dca7457343a3ad62b6a9ba090096ecfcee43ffdc5cfc918
- 接触审计 SHA256：0b307abef6945e757b5456c5ccaadc9f813bfcf323e5747c6eed2169fefae097
- heldout bank SHA256：666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80

- [V6 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-s6101.json)
- [V6 全量 checkpoint](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-s6101.pt)
- [V6 独立 Launch actor](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-s6101.launch.pt)
- [V6 heldout 评估报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-s9524.json)
- [V6 评估轨迹](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-s9524.trajectory.npz)
- [V6 接触事件](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-s9524.events.jsonl)
- [V6 PhysX 接触审计](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-s9524.contacts.json)
- [Launch Tanh 映射消融报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-tanhmapping-skilltrace-s9524.json)
- [Launch Tanh 技能级执行器轨迹](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-tanhmapping-skilltrace-s9524.trajectory.npz)
- [Launch Tanh events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-tanhmapping-skilltrace-s9524.events.jsonl)
- [Launch Tanh PhysX contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-v4-relv3-legacyeval-tanhmapping-skilltrace-s9524.contacts.json)

### 非法接触阶段归因

从正确配置评测的 events.jsonl 中，对每个 failure=4 case 取最后 contact event；事件步均与 outcome 的 policy_steps - 1 一致。事件中 body=true，且 expected_cap=false、expected_wall=false。按 phase 0/1/2，legacy/LR=1e-4 为 18/1/17；causal-v3/LR=1e-4 为 28/3/23；legacy/LR=1e-5 为 51/2/17；CausalV4 为 17/0/26；CausalV5 为 10/1/9；CausalV6 为 17/2/16。phase 0 为首次合法 cap 前，phase 1 为出球至墙，phase 2 为真实碰墙后的回接阶段。V6 将 phase 0 总惩罚提高到 -80 后，phase 0 非法接触并未减少（V5 10 起、V6 17 起），总非法接触由 20 增至 35，故停止单纯惩罚升级；下一步检查可复现失败轨迹中的动作、状态和碰撞几何，再提出可证伪的单变量假设。

### legacy actor LR=1e-5：学习率对照

复用 `AeroWallLateralInterceptV1` 分布、seed 6101、原始 Intercept warmstart、50 updates、409,600 frames、`relative_v3` 观测、legacy reward 及冻结 C350 Hit／Recover；唯一变量是 actor LR 从 1e-4 降到 1e-5。训练更新数、17 张量 warmstart、冻结来源、checkpoint reload 和 standalone `TensorDictParams` actor 导出均通过完整性核对。

原始 v4 开发挑战、seed 9524、128 env：77/128 合法首触、13/128 合法第二拍、平均 rallies 0.1094、最大 2、三连／五连 0；安全失败 70/128（70 非法接触）。93/93 记录到的合法 cap 接触与 84/84 墙面事件经 PhysX 核验。相较 legacy/LR=1e-4 的 110/128 首触、15/128 第二拍和 38/128 安全失败，降低学习率使首触、续打均下降，安全失败升至 70/128。首次评测漏设环境级 `--observation-version relative_v3`、未推进 FSM，已另存为配置不匹配记录；修正配置后报告三段技能步数为 [3400, 1299, 4641]。候选不晋升。

| 训练 reward／actor LR | 合法首触 | 合法第二拍 | 安全失败 | 三连／五连 | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `legacy` / 1e-4 | 110/128 | 15/128 | 38/128 | 0/128 / 0/128 | 未通过 |
| `aerowall_causal_v3` / 1e-4 | 100/128 | 28/128 | 58/128 | 1/128 / 0/128 | 未通过 |
| `legacy` / 1e-5 | 77/128 | 13/128 | 70/128 | 0/128 / 0/128 | 未通过 |

首个 CausalV4 启动尝试因新增奖励统计 key 未登记于 env stats schema，在第 1 个训练更新前退出（0 updates）；该失败报告单独保留于 launch-diagnostics/failed-attempts/。补齐 schema 后已重启正式 pilot。

> 下列早期 Launch／Hit／Recover Tanh 消融报告形成于评测器绑定训练报告与分角色观测版本之前。它们作为历史诊断留档；涉及 V6 Launch 的因果比较当前不依赖这些报告，决策以本页上方记录的 role-aligned Recover-only 配对评测为准。

### AeroWallLaunchTanhMappingAblationV1（同 checkpoint 消融，gate 未通过）

固定 V6 Launch actor（SHA256 `165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b`）、C350 Hit／Recover、seed 9524、同一 128-case v4 bank（SHA256 `666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`）、`relative_v3` 观测和 `legacy` 评估 reward。唯一变化是为 Launch 单独选择 HCSP 原有 `TanhNormalWithEntropy(tanh_loc=True)`；Hit／Recover 仍为 HCSP 默认 `IndependentNormal`。AeroWall 实现名为 `AeroWallLaunchTanhMappingAblationV1`。只评估、不训练、不晋升；带技能字段的复跑逐案结果与前一次 Tanh 评估及 PhysX contact audit 完全一致。

| 指标 | V6 默认映射 | Launch Tanh 映射 |
| --- | ---: | ---: |
| 合法首触 | 110/128 (85.94%) | 119/128 (92.97%) |
| 合法第二拍 | 7/128 (5.47%) | 9/128 (7.03%) |
| 最大连拍 | 1 | 2 |
| 三连／五连 | 0/128 / 0/128 | 0/128 / 0/128 |
| 安全失败 | 39/128 (30.47%) | 29/128 (22.66%) |
| 非法接触 | 35 | 29 |
| 总动作分量限幅率 | 30.56% | 17.68% |

按技能拆分后，Launch 为 3416 个控制步、最大绝对动作 0.964、限幅率 0%；Hit 为 1274 步、限幅率 5.47%；Recover 为 7172 步、限幅率 28.27%。这确认 Launch 映射按预期将其 raw action 限制在执行器范围，但 Hit／Recover 的原有限幅仍在。该消融有正向诊断信号，但三连 gate 仍 false；相对正式 C350 heldout 的安全失败基线 4/128 仍有明显差距，因此不晋升。

报告 SHA256 `83aee9bf660ad8f8182e1bd8ed7bdcb8dd0819ed05077d83d98f9be7879e2af0`；轨迹 SHA256 `a8f13ace5579876ac33c29b3244a51cbe6c7f6875e550b6f2db88ed3fbbc13f3`；events SHA256 `a933bc0655eb762ac3fe608ca7a39ce591e9ad4a8d6b7faab2cc375bd99536fb`；contacts SHA256 `985cdc3c6847b84784f79eee99a50063ab3a93af1ea9dbc01850bcd1e4490841`。产物位于 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/`，文件名含 `tanhmapping-skilltrace-s9524`。

### AeroWallBoundedInterceptV1-CausalV6（有界动作训练 pilot，gate 未通过）

在 Tanh-only 同 checkpoint 消融后，训练了单独命名的 `AeroWallBoundedInterceptV1-CausalV6`。固定 V6 warmstart、`AeroWallLateralInterceptV1` 分布、CausalV6 reward、seed 6101、128 env、50 updates／409,600 frames 和冻结 C350 Hit／Recover；唯一训练变量是 trainable Intercept actor 使用 HCSP `TanhNormalWithEntropy(tanh_loc=True)`。HCSP 库本身未改动。17 个 warmstart actor 张量精确匹配，无 padding／skip；训练权重确有变化，checkpoint reload、冻结 Launch／Hit／Recover 来源及 standalone actor export 均精确。

同一原始 v4 开发 bank、seed 9524、128 env 的身份核验评估：116/128 合法首触、14/128 合法第二拍、3/128 三连、2/128 五连、最大 8 连；安全失败 27/128（23 非法接触、3 无人机撞墙、1 无人机触地）。146/146 合法接触和 143/143 墙面事件经 PhysX 核验，128/128 outcomes 均完成接触审计。相较 V6 默认映射，安全失败从 39 降至 27、合法第二拍从 7 增至 14；相较 Tanh-only 消融，首触从 119 降至 116，安全失败从 29 降至 27，出现 3 个三连样本。总动作分量限幅率 20.54%；Launch 为 0%，Hit 为 6.04%，Recover 为 32.00%。

将每案末个活动策略步按 `skill_id` 与终止原因配对（这是终止位置归因，不单独证明因果）：91 次球落地、9 次越界和 4 次无人机触地／撞墙均在冻结 Recover 执行期间终止；23 次非法接触中，14 次终止于冻结 Hit、8 次终止于新 Intercept actor、1 次终止于冻结 Recover。与 Tanh-only 消融逐案比较，20 个 case 的安全失败消失、18 个 case 新出现安全失败，净少 2 个；3 个三连 case（heldout-0004、0075、0083）均为新出现。主要后续瓶颈分布于多个技能，不能把整体差异归因于单一 actor。

能力 gate 仍为 false：3/128（三连 2.34%）、2/128（五连 1.56%），远低于报告要求的 80%；安全失败仍明显高于正式 C350 heldout 的 4/128。候选不晋升，正式 C350 路由不变。身份核验复跑与首轮候选评估的逐案 outcomes、汇总、PhysX contact audit、轨迹完全一致；身份版评估通过训练报告 SHA256 和 actor checkpoint SHA256 绑定到候选名，避免把训练候选误标成单纯映射消融。

| 指标 | V6 默认映射 | Tanh-only 映射消融 | `AeroWallBoundedInterceptV1-CausalV6` |
| --- | ---: | ---: | ---: |
| 合法首触 | 110/128 | 119/128 | 116/128 |
| 合法第二拍 | 7/128 | 9/128 | 14/128 |
| 三连／五连 | 0/128 / 0/128 | 0/128 / 0/128 | 3/128 / 2/128 |
| 最大连拍 | 1 | 2 | 8 |
| 安全失败 | 39/128 | 29/128 | 27/128 |
| 总动作分量限幅率 | 30.56% | 17.68% | 20.54% |
| gate | 未通过 | 未通过 | 未通过 |

训练报告 SHA256 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`；完整训练 checkpoint SHA256 `35cfb1cd4cdde021d0ec05316c0033426e986fffe5b8fc4268fd8ac155b97121`；独立 actor SHA256 `bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50`。身份版评估报告 SHA256 `832a0147e7c55cb69bd8eb26b79062f1e687d4903027a4a988922486ca5fbed1`；trajectory SHA256 `e36a81304c684fd117cd10ea1f4927d211a09a742f2b701c37ca217ec0f4129b`；contacts SHA256 `c8b38541bb994ec61bed9b41ded4a81516343bdd512219a82292a9f360a05c0a`；events SHA256 `cbdcf73a035729d121f240187372cd7d4e3b9c292a2e5ae5b0e3a12e5b9d708a`。

### AeroWallHitTanhMappingAblationV1（冻结 Hit 分布消融，未改善 gate）

固定 `AeroWallBoundedInterceptV1-CausalV6` checkpoint、C350 Hit／Recover 来源、seed 9524、相同 128-case v4 bank、relative_v3 环境和 legacy actor observations／reward；仅将冻结 Hit actor 改用 HCSP `TanhNormalWithEntropy(tanh_loc=True)`。训练候选、Launch Tanh 映射及 Recover 默认映射保持不变。这里的 AeroWall 名称标记消融设计；HCSP 分布实现保留原名，HCSP 源码未修改。

该消融把 Hit 动作分量限幅由 6.04% 降至 0%，但非法接触仍为 23/128，安全失败仍为 27/128；三连／五连由 3/128、2/128 降至 2/128、1/128，最大回合由 8 降至 5。116/128 首触和 14/128 第二拍不变。Recover 限幅仍为 33.17%，总体限幅率 20.29%。139/139 合法接触、136/136 墙面事件经 PhysX 核验，128/128 outcomes 审计完整；gate=false。映射限幅变化没有带来本轮能力或安全改善。

身份评估报告 SHA256 `3350a215853200fba14c95f1c80cc75bb5872fdb7f4b885d42a336aa58b2af2d`；trajectory SHA256 `5ddd1203e76715890caf35ef73c2df22ebf28a70daa0af69e5d5a99bc15ca2`；contacts SHA256 `b8b02bc2245aa88cfbcbe5ab0567030c37165004511df13bc8b66b609444a539`；events SHA256 `504f466a3be281f923f84febb5525b86ab5f57f08448a585159d7f3c685385d6`。评估使用的候选训练报告 SHA256 为 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`。

### AeroWallRecoverTanhMappingAblationV1（冻结 Recover 分布消融，未改善 gate）

仍使用同一 `AeroWallBoundedInterceptV1-CausalV6` 候选 actor、C350 Hit／Recover checkpoint、seed 9524、v4 128-case bank 和 CausalV6 环境 reward；与候选默认评估相比，唯一变化是冻结 Recover actor 使用 HCSP `TanhNormalWithEntropy(tanh_loc=True)`。Launch 使用训练好的 Tanh actor、Hit 保持默认 HCSP 分布。

Recover 动作分量限幅由 32.00% 降至 0%，全链限幅由 20.54% 降至 0.75%；但安全失败由 27/128 升至 35/128（26 非法接触、5 无人机撞墙、4 无人机触地），合法第二拍由 14/128 降至 11/128。三连／五连仍为 3/128、2/128，最大回合 9。球落地由 91 降至 84、越界由 9 降至 8，但增加的无人机接触抵消了改善。143/143 合法接触与 138/138 墙面事件经 PhysX 核验，128/128 outcomes 审计完整，gate=false。Tanh 限幅显著减少截断，却没有改善整链安全或续打；不晋升。

评估报告 SHA256 `9046276eca609f0f30ff535733ccafb96041c4f07a45a64b868f728289c416df`；trajectory SHA256 `e2bd55fd49abe276fd8b200751550596490812ef28dbe8de3dd33ef2a4b7115f`；contacts SHA256 `a9c58a65aee27ec2f06e721594e89e2fc5b2bc3f970d54530edb493e7ba84ed8`；events SHA256 `da2ee5b1b9fc140939ba5162ea38f2b690264b7dd5eff9b7dbba25ef73d95b8f`。评估绑定的候选训练报告 SHA256 为 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`。

### AeroWallGoalHitPilotT5-S6201 自然 RALLY 整链诊断（gate 未通过）

在同一 seed 9524、原始 v4 128-case 开发 bank、`relative_v3` 观测下，将有界动作 Intercept 候选 `AeroWallBoundedInterceptV1-CausalV6` 的 Hit 阶段由 C350 Hit 替换为 T5 参数化 Hit pilot `AeroWallGoalHitPilotT5-S6201`，Recover 仍使用 C350。此次只做评估，不训练或晋升；目标是观察 T5 Hit 接入自然策略链后的表现。候选训练报告／actor SHA256 分别为 `78a807ee6de19a316dd89b1e9f36b711a9a7ab95a09d3a82afc81cd352d3e927`／`bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50`；T5 Hit checkpoint／训练报告 SHA256 分别为 `d6a4c52ffb71d295d29e4ce3917b28cb232f8253df4f3ae12d50e4d97abc763c`／`90d088b9cc5515ed8e46df963668822197815669f0f960ba126cfe2efc15664b`。

| 指标 | Intercept 候选＋C350 Hit | Intercept 候选＋T5 Hit | 变化 |
| --- | ---: | ---: | ---: |
| 合法首触 | 116/128 | 116/128 | 持平 |
| 合法第二击 | 14/128 | 14/128 | 持平 |
| 三连样本 | 3/128 | 5/128 | +2 |
| 五连样本 | 2/128 | 4/128 | +2 |
| 最大连拍 | 8 | 9 | +1 |
| 安全失败 | 27/128 | 29/128 | +2 |
| 平均 rallies | 0.2344 | 0.3984 | 上升 |
| 墙目标命中率 | 0.3776 | 0.3522 | 下降 |

T5 整链共有 25 次非法接触、2 次无人机触地、2 次无人机撞墙；另有 86 次球落地、9 次越界和 4 个无失败结局。167/167 合法接触、159/159 墙面事件经 PhysX 核验，128/128 结局接触审计完整。动作分量总限幅率 19.85%；Launch／Hit／Recover 分别为 0%／8.42%／29.63%。三连率 5/128、五连率 4/128，P2 gate=false；安全失败也比同候选＋C350 Hit 多 2 个。此处只说明单种子整链诊断出现有限续打信号，尚不能据此判断 Hit 策略单独带来因果改善。T5 Hit 不晋升，正式 C350 路由保持。

报告 SHA256 `f40bb520bc00da064bfd7228ac4574eec312e438f64e90a4d61d5558142374c2`；trajectory SHA256 `c2013f43912c32b2e1f292f0b6fa40f89285a8ec9de5953772cb1c796607222e`；events SHA256 `c16dae06483059081fd300c2d3fe48f0b6117298c7253f03ca309855b8c2895e`；contacts SHA256 `cb3e8c5bad43dc5707378ba75a44f9e531d9a6ede7132e39f9618b5aac859d0f`。完整报告和轨迹位于 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/`，文件名前缀为 `aerowall-bounded-intercept-v1-causal-v6-t5-goalhit-s6201-v4-relv3-legacyeval-s9524`。

## Part 8 阶段进度

- [x] **P0** 接口、FSM、reset、奖励归属与评估协议修复。
- [x] **P1** 正式基线、同权重包装、替换前序诊断；A/B 完全一致，C 被拒绝。
- [ ] **P2** 固定来球参数化 Hit：已做单种子筛选与 T5 Hit 自然整链诊断；Intercept＋T5 Hit 达到 5/128 三连、4/128 五连，但安全失败为 29/128（同 Intercept＋C350 Hit 为 27/128），仍未同时达到续打和安全门槛。冻结 Hit Tanh 无收益。早先 bounded Intercept 配置下的 Recover Tanh 与本轮正确观测版本、冻结 V6／Hit／C350 Recover 的消融是不同配对；本轮 Recover 裁剪由 29.958% 降至 0%，但安全失败 28→36/128，第二击和三／五连均下降。所有 gate 仍不通过，不训练、不晋升。
- [ ] **P3** 小幅位置／速度变化；需先确定 P2 有效信号并冻结能力保持协议。
- [ ] **P4** 真实 Intercept→Hit 状态分层与交接成功率。
- [ ] **P5** Hit→Recover→下一击的连续回合门槛。
- [ ] **P6** 更广分布、三训练种子和新的冻结最终测试集。
- [ ] **P7** 仅在低层稳定且有具体瓶颈证据后启动条件性增强。

## Launch 首击覆盖：v4 侧向反事实诊断

- v4 的 128/128 来球初始侧向位置 `y` 与侧向速度 `vy` 同号；其观测范围超出 train-128-v1 的侧向位置／速度范围。冻结 Launch 在原始 v4 上首击为 0/128。
- 对 `y` 限幅、`vy` 限幅及两者同时限幅，合法首击均为 0/128，且没有 PhysX 审计通过的合法 cap 事件。
- 只翻转来球 `vy` 符号（其余 case 字段逐项不变）后，96/128 个 case 有合法首击，97/97 次合法 cap 事件经 PhysX 核验；但安全失败率为 63.28%，只有 1/128 有合法第二拍，三连／五连均为 0。
- 对原始 v4 与 sign-flip 各补采了 128 环境全轨迹，两个重跑的逐案结果和汇总指标均与先前报告完全一致。两组初态只差 ball `vy`；首个 4 维动作的平均 L2 差仅 `0.00625`，114/128 小于 `0.01`。
- 前 20 个策略步（0.4 s），原始 v4 平均球向外移动 `0.1867 m`、无人机向外移动 `0.1209 m`；翻转 vy 后球改为向内移动 `0.1867 m`，无人机侧移仍近似 `0.1059 m`。这说明 sign-flip 的 96/128 首击主要来自更有利的球路，而不是冻结 Launch 已学会追赶原始向外球路。
- 当前根因假设是冻结 Launch 的早期侧向预判／可达性不足。`relative_v2`／`relative_v3` 同 actor 消融中，初态与首个动作完全相同，首触和续打 gate 不变；仅改观测语义不能解决问题。反事实输入仍仅作诊断，四组原始侧向诊断 gate 都是 `false`。
- 详细指标、数据范围及 bank／报告哈希见 `artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json`。

## AeroWallLateralInterceptV1：单技能训练 pilot

- 训练配置：`AeroWallSkillChainPolicy` 只更新 `Skill.INTERCEPT` actor，warmstart 来自实验 checkpoint `v2-intercept-event-u50-s6101.pt`；Hit 与 Recover 均冻结在 C350。训练使用 `relative_v3`、`legacy` reward、seed 6101、128 env、50 updates／409,600 frames。
- 新训练分布 `AeroWallLateralInterceptV1` 独立随机生成状态，只扩展 ball `y` 到 `[-0.85, 0.85] m`、`vy` 到 `[-0.70, 0.70] m/s`，其余 RALLY train 范围保持不变；不重放 v4 case。actor warmstart 17 个张量均精确匹配，没有补零或跳过，训练后独立 actor export 保持 HCSP `TensorDictParams` 并可加载。
- 原始 v4 开发挑战、seed 9524：110/128 有合法首触，125/125 合法 cap 与 110/110 墙面事件经 PhysX 核验；15/128 有合法第二拍，但三连、五连均为 0，最大回合 1。安全失败 38/128，其中非法接触 36、无人机触地 2。
- 同条件旧 Intercept actor + `relative_v3` 观测为 113/128 首触、1/128 第二拍、安全失败 19/128。新 actor 的续打有提高，但非法接触由 19 增至 36，安全回退，gate 不通过。v4 仍是已用于开发的挑战集，不是新的冻结最终测试集。
- 初始 actor export 曾因普通 `TensorDict` 不满足 MAPPO 的 `TensorDictParams` checkpoint 接口而未能加载；已修正为原容器内逐张量替换，并通过后续完整 v4 评估。

## 下一工作项

1. CausalV6 已完成：phase 0/1/2 非法接触 17/2/16；35/35 非预期 body contacts 的最后事件与终止 step 对齐，118/118 cap 与 110/110 wall event 经 PhysX 核验；gate=false。
2. Tanh-only 消融、`AeroWallBoundedInterceptV1-CausalV6` pilot 和冻结 Hit／Recover Tanh 消融均已完成。当前正确观测版本的 Recover-only TanhNormal 将裁剪率降至 0%，但安全失败 28→36，第二拍 14→11，三／五连 5/4→3/2；不训练、不晋升。现已完成配对轨迹与 phase 0/1/2 非法接触阶段审计，下一步针对首击与回球阶段的接触几何和 Recover 动作做定向诊断。
3. 正式 C350 路由保持不变；P2 未过门槛前不推进 P3–P6，也不创建新的冻结最终测试集。三种子正式训练与独立最终测试要等 pilot 呈现同时满足能力与安全门槛的信号后再启动。
4. 每次实验后更新本表、执行记录、analysis.json 和产物哈希；P7 每类增强单独建版本与消融。

## 主要证据

- [原始实施计划](plans/2026-09-24-hcsp-skill-upgrade-implementation-plan.md)
- [T0–T5 执行与验收记录](wall-skill-upgrade-v3-execution.md)
- [AeroWallLateralInterceptV1 分布配置](../configs/wall_training_distributions/aerowall-lateral-intercept-v1.json)
- [AeroWallLateralInterceptV1 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-s6101.json)
- [AeroWallLateralInterceptV1 actor checkpoint](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-s6101.launch.pt)
- [AeroWallLateralInterceptV1 v4 整链评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-v4-s9524.json)
- [causal-v3 Intercept 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v3-s6101.json)
- [causal-v3 Intercept v4 评估报告（causal eval）](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v3-v4-s9524.json)
- [causal-v3 Intercept v4 评估报告（legacy eval 配对）](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v3-v4-legacy-eval-s9524.json)
- [legacy/LR=1e-5 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-s6101.json)
- [legacy/LR=1e-5 standalone actor](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-s6101.launch.pt)
- [legacy/LR=1e-5 v4 评测报告（relative_v3）](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-relv3-s9524.json)
- [legacy/LR=1e-5 首次配置不匹配的评测记录](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-s9524.json)
- [legacy/LR=1e-5 v4 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-relv3-s9524.trajectory.npz)
- [legacy/LR=1e-5 contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-legacy-lr1e-5-v4-relv3-s9524.contacts.json)
- [CausalV4 Intercept 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v4-s6101.json)
- [CausalV4 v4 relative_v3 评测](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v4-v4-relv3-legacyeval-s9524.json)
- [CausalV4 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v4-v4-relv3-legacyeval-s9524.trajectory.npz)
- [CausalV4 contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v4-v4-relv3-legacyeval-s9524.contacts.json)
- [CausalV5 Intercept 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v5-s6101.json)
- [CausalV5 v4 relative_v3 评测](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v5-v4-relv3-legacyeval-s9524.json)
- [CausalV5 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v5-v4-relv3-legacyeval-s9524.trajectory.npz)
- [CausalV5 contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v5-v4-relv3-legacyeval-s9524.contacts.json)
- [侧向诊断、观测消融与候选哈希](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)
- [AeroWallBoundedInterceptV1-CausalV6 训练报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-s6101.json)
- [AeroWallBoundedInterceptV1-CausalV6 独立 actor](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-s6101.launch.pt)
- [AeroWallBoundedInterceptV1-CausalV6 身份核验 v4 评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-v4-relv3-legacyeval-identity-verified-s9524.json)
- [AeroWallBoundedInterceptV1-CausalV6 v4 trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-v4-relv3-legacyeval-identity-verified-s9524.trajectory.npz)
- [AeroWallBoundedInterceptV1-CausalV6 v4 contact audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-v4-relv3-legacyeval-identity-verified-s9524.contacts.json)
- [AeroWallHitTanhMappingAblationV1 冻结 Hit 映射评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-hit-tanh-mapping-v4-relv3-legacyeval-s9524.json)
- [AeroWallRecoverTanhMappingAblationV1 冻结 Recover 映射评估](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-recover-tanh-mapping-v4-relv3-legacyeval-s9524.json)
- 机器报告与轨迹：`artifacts/wall-skill-upgrade-v3/`（Git 忽略目录；不纳入提交）

### CausalV6 Hit reward pilot 与评估复现性审计（2026-09-25）

> 本节中的 16/128 对照为后续确认的 Launch observation mismatch 结果，已 supersede；正确评估见本页前部的观测版本更正记录。原始报告保留供审计。

> **此节中的 16/128 表格和 115/128 安全失败结论已被后续观测版本更正 supersede。** 后续更正确认 V6 Launch 训练观测为 `relative_v3`，fresh 评测误传 `legacy`；正确配对结果见上方“2026-09-25：观测版本更正与 Recover TanhNormal 配对消融”。本节原始产物保留为审计记录。

用户批准的 Tanh 映射后续候选 `AeroWallBoundedGoalHitV1-Tanh-CausalV6-S6201` 固定 seed 6201、训练 50 updates／409,600 frames；只将 Hit 训练 reward 从 `legacy` 改为 `aerowall_causal_v6`，其余训练配置不变。训练权重有变化、训练报告与 checkpoint reload 完整。与已训练的 `AeroWallBoundedGoalHitV1-Tanh-S6201` legacy actor 用同一个 `AeroWallBoundedInterceptV1-CausalV6` Launch、C350 Recover、seed 9524、heldout-128-v4 bank（SHA256 `666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`）、Tanh Launch／Hit 映射、`relative_v3` 公共观测、`aerowall_goal_v1` Hit 观测及 CausalV6 eval reward 做自然整链评估。

| 评估 actor | 合法首触 | 合法第二拍 | 平均 rallies | 安全失败 | 非法接触 | 球落地 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy Hit fresh control | 16/128 | 0/128 | 0 | 115/128 | 115 | 13 | 未通过 |
| CausalV6 Hit candidate | 16/128 | 0/128 | 0 | 115/128 | 115 | 13 | 未通过 |

两次 legacy control fresh rerun 与 CausalV6 Hit rerun 的 reset 物理/FSM 状态 hash 都是 `c81363f3d8860bb682b1a6a8296b23f218d8883e8a4ef1b12a5b5b4f2d8ec9e9`；新 legacy 重跑的初始 actor 输入、采样诊断轨迹和完整 trajectory hash 均一致。CausalV6 Hit 相对 legacy Hit 的实际首次动作分歧只出现在 4 个 case 的 Hit 技能（env 29／31／81／86，policy step 92／89／88／88）；分歧前所有轨迹逐元素一致，最终 128 个 case 的 rallies／failure／policy_steps 分类完全相同。因此该 reward 单变量没有正向能力信号，不扩成三种子、不晋升。

较早 S6201 legacy 报告当时缺少分角色 actor input、post-reset 状态和源码 provenance，首步动作来源未能直接审计。后续按训练报告指定的 `relative_v3` 复跑得到 116/128 首触、14/128 第二击、28/128 安全失败，且完整 trajectory SHA256 与历史成功报告一致，因此该历史能力结果已复现。另一个曾报告 16/128 的 fresh run 是独立的 Launch observation mismatch，原因已确认并标记 superseded；原始报告均保留供审计。

评估器新增 opt-in `--record-reproducibility`。启用时输出 post-reset 物理/FSM 快照与哈希、CPU/CUDA/NumPy/Python RNG 状态指纹、运行源码与输入 checkpoint 哈希、每个技能角色的观测版本及初始实际 actor 输入／动作，并按采样诊断步记录实际技能输入。两次重复评估的 Python RNG 指纹不同，但物理状态、Torch CPU/CUDA 与 NumPy 指纹、actor inputs 和完整轨迹一致；目前没有证据表明此差异影响策略执行。三次评估均在 16/16 legal 和 13/13 wall 接触事件 PhysX 核验以及报告写入后，Isaac Sim teardown 出现 native segmentation fault；报告 `status=passed`，此退出期崩溃单独保留于运行日志。

| 产物 | SHA256 |
| --- | --- |
| CausalV6 Hit 训练报告 | `b868c586da8078fe88efdd6ea3b8b7b02e7cfdd804e648aa7d858e990c748dd4` |
| CausalV6 Hit full checkpoint | `09f40d3c06407bf1c8afd0419d0de60fb38635832137c3467c0291034e57d395` |
| 当前复现 evaluator | `e74809901f9240f96983da30f26939afe81cdf14ea6c97e4899b7635eaf4ec8a` |
| 技能策略输入记录 | `2b383ccdcb85ed07f21299ce2e2dc0b6e1476aa81127c4c0b7de0df017855274` |
| Legacy fresh control 1 trajectory | `a399377aac7d2e99c6492e91871e02be92524035cbcd50c1d09ce92a1a2a4510` |
| Legacy fresh control 2 trajectory | `a399377aac7d2e99c6492e91871e02be92524035cbcd50c1d09ce92a1a2a4510` |
| CausalV6 Hit reproduction report | `92c4a27475d12794605f7b04153333e555f8767999737af76d1458fe3c701efb` |
| CausalV6 Hit trajectory | `717bf82068fcc9ffd9cf4e01ac0f27f76fe66db36322fe700424ee1aa1344d7b` |

- [CausalV6 Hit 训练报告](../artifacts/wall-skill-upgrade-v3/aerowall-bounded-goal-hit-v1-tanh-causal-v6-s6201.json)
- [CausalV6 Hit instrumented reproduction report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/causal-v6-hit-repro-s9524.json)
- [CausalV6 Hit instrumented trajectory](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/causal-v6-hit-repro-s9524.trajectory.npz)
- [Legacy fresh control reproduction 1](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/legacy-control-repro1-s9524.json)
- [Legacy fresh control reproduction 2](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/legacy-control-repro2-s9524.json)

下一步：针对正确配置下的非法接触和 Recover 阶段回接失败做完整状态／动作／接触几何诊断；若形成有证据的假设，再设计不改变正式 C350 的单变量测试。P2 gate=false，正式 C350 不变，P3–P6 保持 gated。

### 2026-09-25：phase 0 Launch 动作前—精确接触对齐

- 用控制组与 Recover-Tanh 组各自的 33 案 actor-input telemetry，逐案选取最早 `phase_before == 0` 的 body contact。两组 case 和合法性标签完全相同；由于 Tanh 仅替换 Recover 映射，这 33 个 phase 0 对照不是两个独立样本。
- 对每一案，事件 observation 的相对球位 (26:29) 与 trajectory 的前一帧 `policy_step - 1` 按该帧 WXYZ 姿态反旋转所得位置最大绝对误差为 `7.3e-8 m`；事件 `policy_action` 与 `trajectory[policy_step].action` 完全一致。物理接触发生在该动作开始后 2.5–20 ms，中位 12.5 ms。
- 21 个合法首触中，动作前径向误差中位数为 0.130 m，接触时 0.112 m；local-y 中位数从 -0.116 m 变为 -0.093 m。12 个非法首触中，动作前径向误差中位数为 0.220 m，接触时 0.215 m；local-y 中位数从 -0.208 m 变为 -0.204 m。9/12 个非法案在最后动作决策时已越过 0.20 m 拍面半径；其余 `heldout-0012`、`heldout-0051`、`heldout-0099` 在接下来一个动作周期内越界。8/12 个非法案动作前 local-y 小于 -0.18 m，合法组为 1/21。
- 按接触前 policy step 对齐 trajectory 的 local-y 中位数：距最终动作 0.40 s 时合法／非法为 +0.476／+0.267 m；0.30 s 为 +0.521／+0.408 m；0.20 s 为 +0.296／+0.250 m；0.10 s 为 -0.010／-0.046 m；0.04 s 为 -0.123／-0.170 m；最终动作决策时为 -0.116／-0.208 m。分组差异在进入最后 20 ms 之前已出现，且轨迹并非单调分离；该方向性不代表控制动作造成失败。
- 当前能支持的结论：非法首触多数在最后动作决策时已是偏心来球，且偏差于接触前数百毫秒已存在；末端 Tanh 映射不是 phase 0 的改变量。它尚不能区分初始条件、较早的 Launch 控制响应或来球动力学。
- 下一步按时间回放 Launch 的 0.2–0.4 s 接近段，找出横向误差开始扩大的动作／状态变量；然后以相近来球状态配对，并独立分析 phase 2 实际 Hit／Recover actor。提出可证伪的单变量假设后先做不训练的配对评估。P2 gate=false，正式 C350 保持，不训练／晋升候选。
- 可复核产物：[可复现分析脚本](../scripts/analyze_aerowall_contact_alignment.py)、[详细机器报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-phase0-preaction-contact-alignment-v1-s9524.json)、[汇总与哈希](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json)。详细报告 SHA256 `4b27a61ca51afbd38ca738ed199726303189f791f0081337628b3508b521eb85`；分析脚本 SHA256 `deae56adaceed93629d53756c7d600be84d56a1ff1fd352a445624219a1321dc`。

### 2026-09-25：Recover-Tanh phase 2 共同 case 接触比较

- 按每个 case 最早的 phase-2 非法 body contact 事件比较同一批 33 案里的 control 和 Recover-Tanh。对照 11 案在 Tanh 中全部重现，Tanh 另有 3 个新增非法 case：`heldout-0037/0053/0124`；没有 control-only case。Tanh 另有一个 phase-1 Recover 非法接触 `heldout-0083`。
- control phase 2 active actor 为 Hit 10、Recover 1；Tanh 为 Hit 6、Recover 8。11 个共同 case 中有 6 个 actor role 从 Hit 切换为 Recover。共同 case 的事件 radial error 差值（Tanh − control）中位数 `+0.015 m`，范围 `-0.173` 至 `+0.107 m`；6 案增大、5 案减小。差异并非同方向，不能把 aggregate safety 退化解释为所有 Recover-Tanh 接触都变差。
- 全量 128 案 safety failure 28→36 仍是该 mapping intervention 的正式配对结局；当前 33 案只是定向诊断子集，用来定位异质接触案例。该分析未训练、未晋升，C350 保持、P2 gate=false。
- 下一步把 phase-2 action-time Hit／Recover observation 与 contact-substep 状态逐案对齐，并优先审查共同 case 中角色切换和新增的 3 案；与 phase-0 Launch 0.2–0.4 s 接近段追踪分开，不混为同一根因。
- 汇总位于 [analysis.json](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json) 的 `phase2_shared_case_actor_contact_comparison_v1`；原始来源为 [control events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-control-runtime-s9524.events.jsonl) 和 [Recover-Tanh events](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-contact-substep-actor-input-audit-v1-tanh-runtime-s9524.events.jsonl)。

### 2026-09-25：全量 phase 0 接近段与 phase 2 actor-action 对齐

- 在完整 v4 development bank 的 128 案 control 与 Recover-Tanh trajectory 上重算最早 phase-0 body contact。两条件事件标签与 case 完全相同，均为 116 个合法、12 个非法。非法组中 9 案在最后动作前径向距离已 >0.20 m，`heldout-0012/0051/0099` 在最后动作周期内越界；全部 12 个精确回调 radial 均 >0.20 m。由相邻 policy frame 插值出的 contact pose 对 callback radial 的最大差为 0.000347 m。
- 全量合法／非法组在距最后动作 0.40 s 时，local-y 中位数分别为 +0.405／+0.267 m，world ball-minus-drone y 为 -0.521／-0.642 m，drone world-vy 为 -0.606／-0.518 m/s。按该时刻球世界位置与速度的 pooled-SD 距离做最小代价 12 对 12 匹配，7 对距离 ≤1、11 对 ≤1.5。配对非法−合法 local-y 差中位数随 lead 20/15/10/5/2/0 帧为 -0.144/-0.094/-0.043/-0.028/-0.044/-0.057 m；world relative-y 差从 -0.022 m 增至 -0.077 m，失败组 drone-vy 在 lead 20–5 帧的配对中位差约 +0.123 至 +0.136 m/s。描述了较早横向跟随偏差，但这是同一个开发 bank 的观测性比较，不证明动作因果。
- 在 failure-enriched 的 33 案 exact-event actor-input rerun 中，最早 phase-2 非法接触的 actor observation 球世界位置／球相对无人机位置与 policy-step-1 trajectory 对齐误差均为 0；event policy action 与该 step 的 trajectory action 也完全一致。control 的 11 案最后动作前径向误差中位数 0.309 m（actor Hit 10／Recover 1）；Recover-Tanh 的 14 案为 0.318 m（Hit 6／Recover 8）。11 案在两条件共享，另 3 案仅在 Tanh 出现；所有这些 phase-2 miss 在最后动作前已径向越过 0.20 m。
- 更正 action-path：当前 AeroWall evaluator 使用四路直接 rotor command，HCSP RotorGroup 实际映射为 `target_throttle=sqrt(clamp((clip(action,-1,1)+1)/2,0,1))` 后经电机滞后产生推力。全量配对分析已从错误的 `PIDrate_FM` 角速度解码改为四个限幅 rotor command 和目标油门。非法−合法命令中位差在 lead 20/10/0（约 0.40/0.20/0 s）分别为 `[+0.099,+0.154,+0.093,-0.017]`、`[-0.057,-0.017,-0.056,+0.000]`、`[+0.089,+0.004,+0.021,-0.001]`；这些仅是匹配开发样本中的描述性关联，不能解释为世界 y 控制轴或因果策略。
- 结论：Recover Tanh 映射消融没有改善回合或安全 gate，P2=false、C350 保持；本轮没有训练或晋升。下一步先起草一个具备反证条件的 AeroWall Launch 早期接近干预，并在不改变 checkpoint／正式路由的前提下定义配对评估；不从这些相关性添加固定 actor-action 偏置。
- 可复算脚本：[全量接近与 actor-action 对齐分析](../scripts/analyze_aerowall_fullbank_contact_approach.py)，SHA256 `937630f3158c740741a3505cb3711b1df4810cf0332613e0f433705effd44f52`；机器报告：[JSON](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-fullbank-contact-approach-alignment-v1-s9524.json)，SHA256 `ae741d65616330242625e4ce84fcadfc87f37a5f655f3ff016b486e1afff075e`。报告输入包含完整 128-case control/Tanh 原始报告、events、trajectory、PhysX contacts 哈希，以及 33-case actor-input 对照与 Tanh 遥测哈希。

### 2026-09-25：AeroWall Launch 截点目标保留消融

- 按接近段诊断实现独立命名的 `AeroWallLaunchInterceptTargetRetentionV1`。只在 phase 0 且预测来球的高度平面交点时间与场地边界有效时，把 Launch actor 观测前三个 goto 差值保留为预测截点目标（无人机可达性判定不再触发退回当前球跟踪目标）；`relative_v3` 的其余 43 个输入、可行标记、critic 公共观测、checkpoint、action mapping、FSM、reward 和 Hit/Recover 观测均不变。正式路由和 C350 不变。
- Control 与消融使用同一 seed 9524、同一个 heldout-128-v4 development bank（SHA256 `666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`）、相同 V6 Launch／Tanh Hit／C350 Recover checkpoint 和评估配置；只有 Launch 观测版本不同。两个 evaluator 报告均 `passed`，128/128 结局均通过接触审计。
- 首次 reset 时 128 个 Launch actor observation 逐元素相同；实际 rollout 中有 14/128 个 case 的 Launch action 首次在 trajectory index 14–33 发生分歧，active action 行共 714/13,547 行不同。由此确认干预改变了运行时输入和策略动作。
- 能力结果未改善：两组均为 116/128 合法首触、12/128 非法；合法第二拍 14/128、三连 5/128、五连 4/128、平均 rallies 0.375 均相同。安全失败由 28 增至 29/128；`heldout-0060` 从 control 的球落地结束，变为消融组越过环境无人机 world-x < 0.5 m 任务阈值（failure code 5）。该阈值终止不表示发生物理撞墙；PhysX sidecar 未记录无人机／单墙接触，也没有恢复失败 case。P2 gate 仍 false。
- `heldout-0060` 路径已复盘：首次动作分歧在 trajectory index 16；两组均于 step 36/substep 3 合法触球，消融径向误差更小（0.0341 m vs 0.0635 m），但在 index 75 首次越过 x=0.5 m 阈值；control 最小 active x 为 0.5111 m，最终球落地。两组均无 PhysX 无人机／单墙接触。这是描述性配对时间线，不是因果结论。
- 判定：保留为负向的 inference-only 消融证据；不训练、不晋升，不把目标保留策略用于正式 C350。下一步汇总现有配对材料中的分角色失败模式；不启动训练，P2=false 且正式 C350 保持。
- 实现：[观测构造](../aerowall/wall_rl/observations.py)、[轨迹可行性](../aerowall/wall_rl/trajectory.py)、[环境视图](../scripts/aerowall_wall_rally_env.py)、[技能策略](../scripts/aerowall_skill_policies.py)、[评估入口](../scripts/evaluate_aerowall_wall_rl.py)。复算脚本：[消融分析](../scripts/analyze_aerowall_intercept_target_retention_ablation.py)，SHA256 `c5ff64d9fade88056525da96b2491b93b708a656f20eb3bd55b3cc0faca2b170`。
- 机器报告：[配对分析](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-launch-intercept-target-retention-ablation-v1-s9524.json)，SHA256 `b1a1af915143f89c44d5543321b5318fbbfd30aab44ce2c25190b784138f11a1`；[消融原始报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-launch-intercept-target-retention-v1-s9524.json)，SHA256 `345c9968e6b58d8b562d9bc7ea0915e7377d0f032f383f72e482e08a5b4b0380`。完整 events、PhysX contacts、trajectory、日志及输入哈希一并保留在同一 reproducibility 目录。


### 2026-09-25：分角色、分 phase 失败模式合成

- 合成脚本只读核对 10 个现有报告／轨迹 SHA256，没有启动模拟或训练。phase 0 两条件首触均 116 合法、12 非法且标签相同；12 个非法接触都由 Launch actor 产生，9 个在最后动作前已超出 0.20 m radial limit。
- phase 2 的失败定向复跑为 control 11 案、Recover-Tanh 14 案（11 案共有、3 案 Tanh 新增）；共享样本中 6 案从 Hit actor 路由变为 Recover，径向误差有增有减。不能归因为单一 Recover 动作机制。
- 截点目标保留实验的 14 个首次动作分歧，在动作前 trajectory index 的六项记录状态／电机字段均逐位相同；说明首个动作响应来自观测干预，但该组 full rollout 没有改善回合指标且多 1 个 safety failure。
- 判定：没有新策略或训练假设达到 P2 候选条件。下一步须先提出可证伪的 phase-specific 机制与预注册的配对安全／能力拒绝标准；继续保持 P2=false、C350 正式、P3–P6 gated。
- 详细证据：[合成说明](aerowall-phase-specific-failure-synthesis-v1.md)、[JSON](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-phase-specific-failure-mode-synthesis-v1-s9524.json)；脚本 SHA256 `15665229b3f24b3774148be996b6ca50c8233e093d128568e6832f7d00052241`，JSON SHA256 `d591ae0cdcba0e27e7423dafd40178f85dce0cfeb1a35353e6ff16d02aa9b57f`。

### 2026-09-25：AeroWallRecoverHandoffBankV1 与同状态续跑

- 扩展评估器的角色化 handoff schema：既有 Hit v1 继续兼容；Recover 使用显式 `handoff_actor_role=recover`、`handoff_state_version=2`、`skill_id=2`。状态包括无人机／球位姿与速度、电机 throttle、动作历史、wall target、FSM 计数及经 PhysX 审计的 cap→wall 前缀。
- 在原始 `heldout-128-v4` 开发 bank 上，以 V6 Launch、Tanh Hit、C350 Recover 和 seed 9524 跑自然 RALLY 来源评估。128 案中 114 案实际到达 Recover；114/114 状态通过完整性和事件前缀校验，拒收 0。来源 164/164 legal-cap、158/158 wall events 均经 PhysX 核验。
- 新建 bank SHA256 `539d8999aa91f232b86ee58d852e8d0cc8ce5c5e4e159d3691cc1307381a0f4c`。对 114 案执行 Isaac reset 读回审计：20/20 字段全部通过 `1e-4` 容差；最大误差 `1.9073486328125e-6`（仅 drone／ball position），姿态、速度、电机、动作历史、FSM／计数和 wall target 误差为 0。
- 从同一 bank 做默认 Recover 与 `AeroWallRecoverTanhMappingAblationV1` 条件续跑。两组后续首次合法 cap 均为 14/114；默认／Tanh 平均 handoff 后 rallies 为 0.4123／0.2807，3 连为 5/114／4/114，5 连为 3/114／2/114。逐案 107 持平、7 下降、0 上升。按项目安全失败定义（无人机触地、非法接触、无人机撞墙），默认 18/114、Tanh 21/114；6 案新发生、3 案消失，净增 3。来源前缀每条件继承 228 个已审计事件；新增接触 control 为 47/47 cap、41/41 wall，Tanh 为 32/32 cap、29/29 wall，callback errors 均为空。
- 这是“已由默认 Recover 到达的状态”上的条件诊断，源自开发用 v4 bank；不代表独立测试集或自然整链总体率。结果不支持晋升 Recover-Tanh，也不改变正式 C350。P2 仍为 false，不启动 P3–P6。
- 产物：[来源报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-source-v1-s9524.json) SHA256 `5982e67f6a7773a91236ed4b4e6ad07b2f7e1cecaf01101c50579f53a9645bab`；[Recover bank](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-bank-v1-s9524.json) SHA256 如上；[reset audit](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-reset-audit-v1-s9524.json) SHA256 `a3137d1d28e4d8d1a83b2a8406b5e2a24190e9224e8dbb9a4e1562f1c3755da5`；[control continuation](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-continuation-control-v1-s9524.json) SHA256 `412c150905e3ff742111341c319283ebb7cb043c91f7eefd4d136e6ca86f78f3`；[Tanh continuation](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-continuation-tanh-v1-s9524.json) SHA256 `89aae97c6811499e881d22f32a1c6460ece7d1212cec72c5bd3fcc09f9824ad8`；[配对汇总](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-recover-handoff-continuation-comparison-v1-s9524.json) SHA256 `ceacfa37791e142712a47d389c8377c6661b0757cbe1686c4005e0ba76dcd49f`；分析脚本 SHA256 `5c134b93ee88e56b58b50cada5932c836e4623e7833e4e676cd522b56b79fef4`。
- 下一步：保留该 bank 供 Recover 条件分析；在提出单一 phase-specific、可证伪机制并预注册能力／安全拒绝门槛前，不进行训练或候选晋升。

### 2026-09-25：AeroWall Rotor Command Response Counterfactual V1

- 按预注册方案固定 seed 9524、12 个 Launch handoff、V6 Launch／Tanh Hit／C350 Recover；运行 1 个 fresh control 和 8 个单通道 ±0.10 rotor-command 处理，共 108 个 case-continuation。每个处理臂 12 案均完成 10 个决定、120 条 override trace；每臂 12/12 首次 body contact 均通过 PhysX sidecar 核验，callback errors 为 0。
- 九臂 post-reset physical/FSM state SHA256 均为 f5edcc051b0b154f66a83d53c368916bff51b60b3f5368679c46918987234280；Torch、NumPy、CUDA RNG 指纹与首个 actor action 一致，fresh control 第一步动作与冻结来源最大差 2.4e-7。Python stdlib RNG 没有被 evaluator seed，故九进程的该项指纹不同；记录为复现限制。
- rotor-1/-0.10、rotor-2/-0.10、rotor-3/-0.10 分别在 12/12 案把 step-9 world-y 速度推向匹配合法参考，绝对处理效应中位数为 0.03609/0.02651/0.02527 m/s。rotor-1/-2 分别新增 3/2 案非法接触，按预注册规则拒绝候选使用。rotor-3/-0.10 没有新增安全或球 world-y 边界失败，是唯一未被本 screen 安全规则排除的局部信号；但 control 与它的合法首触都是 3/12，首触径向误差只改善约 0.0012 m，平均 rallies 都是 0。
- 其他处理未通过方向／幅度门槛，或引入了非法接触／球越界；越界新增发生在 rotor-0/-0.10 的 heldout-0113、rotor-1/+0.10 的 heldout-0113、rotor-2/+0.10 的 heldout-0033/0107/0113。所有处理臂和 adverse outcome 均保留。
- 结论：获得了局部 action sensitivity 证据，但没有 task-level 提升；仍不通过 P2，不训练、不晋升、不改变 C350，P3–P6 gated。独立后续须另行预注册、使用未触碰的 bank 并 seed 所有 RNG。
- [详细结果](aerowall-rotor-command-response-counterfactual-v1.md)、[预注册协议](plans/2026-09-25-aerowall-rotor-command-response-counterfactual-v1.md)、[分析脚本](../scripts/analyze_aerowall_rotor_command_response_counterfactual.py)、[机器分析报告](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-rotor-command-response-counterfactual-v1-analysis-s9524.json)。分析报告 SHA256 52de9078969a888751493c58a2123904073a6cab50581d4a847767c88a5540cc；分析脚本 SHA256 650884dea00f62a5930f417dd470321387762390422d0e3841f6616695523703。


### 2026-09-25：AeroWall Launch Tanh Independent Replication V1

- 复核原 v4 Tanh Launch 局部正向信号时，发现 evaluator 已导入并记录 Python random 状态，但未调用 random.seed(args.seed)；已补 seed，评估器源码 SHA256 为 ba24ca57befddf42b768121020c01e2f11a856e27d989f3e070e3ebd062a8502。
- 运行前冻结无训练协议，使用新 development bank configs/wall_cases/devrep-aerowall-tanh-launch-v1-seed-260925.json（128 案，SHA256 b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da），固定 seed 260925、V6 Launch checkpoint SHA256 165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b 与 C350 Hit／Recover SHA256 4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659；唯一变量是冻结 Launch actor 使用默认分布或 HCSP TanhNormal。
- Control 与 Tanh 的 reset 物理/FSM state hash 完全相同，Python／NumPy／Torch／CUDA RNG fingerprints 完全相同，初始 actor observation 相同。两侧 128/128 结局均完成 contact audit；control 95/95 legal-contact 与 91/91 wall events、Tanh 102/102 与 92/92 均由 PhysX corroborate，callback errors 为 0。
- 总 raw action 限幅由 30.18% 降至 15.77%，Launch 限幅 29.59%→0%。合法首触 91→99/128，但合法第二击 3→2；三连／五连仍 0/128；安全失败 44→47/128（非法接触 44→47），球越界 18→14。逐案回合数 127 案持平、1 案下降，没有上升。
- 判定：限幅改善和首触增加未转化为续拍／安全 gate 改善，独立任务级复核失败；不启动新的训练、不晋升任何候选、不改变 C350，P2=false，P3–P6 gated。此 bank 明确为开发复核，不是最终冻结测试集。
- 协议 plans/2026-09-25-aerowall-tanh-launch-independent-replication-v1.md SHA256 c61d1fdddee11c5c2557d35ef921d70b5a1920311e01aec955967cbaf3f8f892；分析脚本 scripts/analyze_aerowall_tanh_launch_independent_replication.py SHA256 3977bcd3b5c9a1ccd3126cd4504aca1d0ae4767f0a399632f329db0567f64cd6；机器分析 artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-tanh-launch-independent-replication-v1-analysis-s260925.json SHA256 0738f4ba54141f2c4769fd149f535bb8572f011588344faa8e351586cb9fdc01。Control／Tanh 报告 SHA256 分别为 5ad16aef081ff2db5c469a78eaa1519218ac9e217f63cfdbb62e58afd8834c61／e027e86fb3446a0da05863058007c074f2280eb1a85f6f7d1e9db549cd003935。
- 同 bank 的逐案 phase 复算：15 案仅 Tanh 首触成功，7 案仅 control 首触成功；Tanh-only safety failure 20 案中，7 案同时丢失 control 的首触，13 案首触状态不变，没有 Tanh-only 首触成功却新增 safety failure 的案例。非法接触最后事件阶段由 control 的 phase-0 Launch 37、phase-2 Hit 6、phase-1 Recover 1，变为 Tanh 的 phase-0 Launch 29、phase-2 Hit 9、phase-2 Recover 2、phase-1 Recover 7；这是终止事件位置描述，不作因果归因。
- heldout-0063 是唯一回合数下降案：control 在 policy step 31/substep 2 有 radial 0.1806 m 的合法 Launch cap，完成 1 rally；Tanh 在同 policy step/substep 5 的 radial 0.2021 m 发生非合法 Launch contact，0 rally。该单案仅作逐案说明。
- 对 13 个首触状态相同、仅 Tanh arm 后续新增 safety failure 的 case 做了 event-level 对齐：6 个最后非合法接触位于 phase-2 Hit，5 个位于 phase-1 Recover，2 个位于 phase-2 Recover；radial error 范围 0.2416–0.4197 m，均超过 0.20 m racket radius。Tanh 后续执行的 Hit／Recover 在异于 control 的 handoff 状态上运行；这只是描述性接触位置证据，不能单独区分上游状态变化和下游控制失配。
- Phase 对齐分析脚本 ../scripts/analyze_aerowall_tanh_launch_phase_differences.py SHA256 25e66f899f28bfecdaa0965feba7ffde712a87820fb8e75856625c0222db03c0；机器报告 SHA256 9907c17713cc7221deecb170dd839a3780d7ed31169548d046c7209d4696bd76。
- 下一步：对这 13 案逐对对齐合法首触／墙接触后的状态、active actor observation 与下一次接触子步，查明 handoff 前已有差异还是下游动作进一步扩大径向误差；仅当形成可证伪机制时再写 inference-only 方案，不训练或晋升。



### 2026-09-25：AeroWall Launch Tanh 首触后状态对齐

- 新建 AeroWallLaunchTanhPostContactAlignmentV1 只读分析，使用 seed-260925 同一 128 案 development replication 中 13 个 Tanh-only 安全失败、且两臂均保持合法首触的配对案例；未运行新策略、未训练、未改 C350。
- 13 案中 12 案两臂在同一 policy step 首触；Tanh 首触 physics substep 相对 control 的中位差为 +4。首触径向误差的配对中位变化为 -0.00327 m，接触前球速差中位 L2 为 0.098 m/s，而接触后球速差中位 L2 为 0.631 m/s。相同相对 policy-step 的 post-step trajectory 里，首触所在步末球位置差中位 0.122 m、无人机位置差中位 0.092 m。
- 首触后的第一个 policy action 在 13/13 案均由 evaluator 路由为 Recover；Tanh 与 control 的动作差中位 L2 为 0.555。轨迹包未保存该时刻的 actor observation，因此不能从此动作差反推输入特征或策略失配。
- 13/13 Tanh 案在首个合法 cap 后出现非合法 body contact，control 对应 13 案均无后续非合法 body contact。Tanh 失败事件 active actor/phase 为 phase-1 Recover 5 案、phase-2 Hit 6 案、phase-2 Recover 2 案；径向误差 0.2416–0.4197 m，全部超过 0.20 m 拍面半径。
- 结论：差异在 Launch 合法首触时已体现在接触后球速与后续状态；随后 Recover 动作不同。当前配对日志无法区分 Launch 交接状态变化与下游控制响应的因果贡献。这 13 案为 post-hoc development 子集，不代表总体率或最终测试集。
- 分析脚本：scripts/analyze_aerowall_tanh_launch_postcontact_alignment.py。机器报告：artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-tanh-launch-independent-replication-v1-postcontact-alignment-s260925.json。
- 已完成后续单变量 inference-only ball-velocity restore screen，详见下节；该 screen 未通过安全拒绝门槛，因此关闭这条机制线。P2=false，不训练、不晋升，P3–P6 继续 gated。


### 2026-09-25：AeroWall Launch Tanh Post-contact Ball Velocity Restore V1

- 先按冻结协议重跑默认 Launch control 与 Launch-Tanh no-intervention。两臂均 `status=passed`；reset physical/FSM state SHA256 均为 `b202ca0861e6470f5a2dae16f028f69457c0be480cd18e3898d442572a2b85b9`，初始 active actor observation SHA256 均为 `aa9c86564971213266ff8c734fe650f2de2d38bcc9767539f9688f52e0f48a2e`。两臂 trajectory、events、contacts 均与之前存档逐字节一致；Control trajectory SHA256 `51a233a5d5b788942e82bf50fc88480e5b02034b4678567eb51b1c6dec33cbd1`，Tanh trajectory SHA256 `fb4ed6426f0d4a6d71cbefc5ded8c3e63399405f61efc43e0e88f333f5fd336f`。Control 95/95、Tanh 102/102 legal events 由 PhysX corroborate。
- 初次重跑尝试在 Isaac 初始化之前因新 guard 提前读取未加载的 restore payload 而停止，没有生成模拟结果；将 guard 移到 payload 加载后。该 opt-in-only 修复后的 evaluator SHA256 `1f4c5ca912a51d30d10adece9a23069a7e013db413c3f3a8b72b2148edc746c5`，默认/Tanh trajectory、events、contacts 精确复现。修订时间和旧／新哈希已写入冻结协议；target bank 记录最终协议 SHA256。
- 仅对预注册的 13 个 Tanh-only illegal-contact development cases 做状态干预：首个 PhysX-audited legal Launch cap 后，单独将 ball linear velocity xyz 替换为同案 Control 在首 cap policy step 的 post-step velocity；保持位置、姿态、角速度、无人机、FSM 与 transition reward/done，并同步 `prev_ball_vel` 和刷新下一状态观察。13/13 次 live readback 最大误差 0 m/s（容差 `1e-5`）；速度改变量 L2 中位数 0.596 m/s；13/13 下一 active actor 均为 Recovery。下一 Recovery observation/action 差中位数 L2 分别为 0.863／0.317。恢复组 102/102 legal events、96/96 wall events 经 PhysX corroborate，callback errors 0，128/128 outcomes audit 完整；所选新增的 12/12 ball-ground 终止也都在 terminal policy step 由球—地面正 PhysX contact_count 逐案核实。
- 预注册 gate 判定失败：所选 13 案 Tanh no-intervention 为 13 次 illegal-contact；恢复后为 1 次 illegal-contact、12 次 ball-ground。全 128 案 illegal-contact 47→35 的同时，ball-ground 67→79；out-of-bounds 为 14→14，drone-ground 与 drone-wall 均为 0→0。所选案合法第二击保持 1/13，整 bank 为 2/128；两臂 rally histogram 均为 126 案零回合、2 案一回合。速度恢复改变了后续 Recovery observation/action，却只把失败类型从 illegal contact 转为 ball-ground，没有带来回合收益；“单独恢复球线速度即可修复后续非法接触”的假设在本 screen 被拒绝。
- 该后接触干预验证的是 13 案上的局部状态反事实，不是自然策略性能、总体机制、P2 测试或候选晋升证据。C350 不变，不训练、不晋升，P2=false，P3–P6 gated。此分支关闭。按 Part 13，任何开发 pilot 未同时通过能力与安全门槛前，不创建 final heldout bank；当前也不启动训练。后续须先另行定义并预注册单机制开发 pilot 及拒绝门槛，只有其结果达到 P2 前置条件后才冻结正式测试集。
- 协议 `docs/plans/2026-09-25-aerowall-launch-postcontact-ball-velocity-restore-v1.md` SHA256 `4e46680f83342ef8666caee476db7698e1cede4a9488dde266117cd8687d9af5`；目标 bank SHA256 `1e47d4f8d7578c4ed18a2a27ace14b4e79717b2f93b07cd0509157d0439df5e3`；分析脚本 `scripts/analyze_aerowall_tanh_ball_velocity_restore.py` SHA256 `5343e2b686b99eb4477f68869634c5ad8468665c8b9a281a60d4280cebcbe34a`；机器分析报告 SHA256 `d1aa684ed106679baa92967d7351c2c703fd22eb19bdbda2ab87634a0636fa2c`。三臂结果报告 SHA256 分别为 Control `03d78ab24cdde78eb49d2fcb539d202b71e13a2e1c29af336f72b5cf35e47ec0`、Tanh `2033333bd9db793b301c903e9ff993a5987f9b09ec7ea1de78f5a3d2bc910f60`、restore `1ba65be39abcfad1a51dc5ed4343fab533e65e271974c039fa7bc832144ab823`。


### 2026-09-25：AeroWall V6 同 bank Launch Tanh 可复现配对复核

- 复核目的：使用同一 V6 Launch actor、同一 C350 Hit/Recover 与原始 heldout-128-v4 development bank，当前 evaluator 显式绑定 Launch/Hit/Recover 的 relative_v3 观测，并记录 reset、RNG 与 source hashes；唯一处理变量为 Launch action distribution 从 HCSP 默认 IndependentNormal 切到 HCSP TanhNormalWithEntropy(tanh_loc=True)。未训练、未改 checkpoint、未晋升、未改变 C350。
- 冻结输入：seed 9524、128 cases；bank SHA256 666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80；V6 Launch SHA256 165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b；C350 Hit/Recover SHA256 4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659。完整冻结协议位于 docs/plans/2026-09-25-aerowall-v6-samebank-launch-tanh-replication-v1.md。
- 配对有效性：128 个相同 case ID；两臂 reset-state SHA256 都为 c81363f3d8860bb682b1a6a8296b23f218d8883e8a4ef1b12a5b5b4f2d8ec9e9；Python/NumPy/Torch CPU/Torch CUDA RNG fingerprint、逐环境 reset fields、初始 actor observations、角色观测、environment/reward/source SHA 均相同。所有 outcome 接触均完成审计，无 callback errors。
- 指标：合法首触 110→119/128；合法第二拍 5→5/128；三连与五连均 0→0/128；最大连拍 1→1。总安全失败 39→33/128（非法接触 38→31；无人机撞墙 1→2；无人机触地 0→0）；越界 19→19，球落地 70→76。总 raw-action 限幅率 29.96%→16.19%；Launch 30.09%→0%，Hit 26.54%→0.84%，Recover 30.58%→27.03%。Launch Tanh 明显减少限幅并提高首触、降低总安全失败，但没有增加续拍。
- PhysX 审计：Control 117/117 legal events 与 108/108 wall events 被 corroborate；Tanh 127/127 legal events 与 106/106 wall events 被 corroborate；callback errors 均为 0。
- 预注册任务级信号要求合法第二击与三连都高于 control，且安全失败不增加。本次虽有首触、限幅与总安全失败改善，但第二击持平、三连持平为 0，因此严格门槛未通过；不启动新的训练，不晋升。此前已有 AeroWallBoundedInterceptV1-CausalV6 独立命名候选及自然整链评估，本次结果不足以另起一轮重复训练。新的 heldout bank 独立复核仍需一并考虑；P2=false，P3–P6 gated，正式 C350 保持。
- 运行中两次配置错误的 control 尝试被移入 artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/invalid-comparisons/ 并保留：首轮使用 legacy；第二轮只把全局 observation 设为 relative_v3、技能角色仍默认为 legacy。它们均未进入配对结论。正式 control/tanh 评估各自 status=passed，且 evaluator logs/artifacts 完整保留。
- 分析脚本 scripts/analyze_aerowall_v6_samebank_tanh_replication.py SHA256 88c9af714150c064944fb8800c7712594f4547b964515f343d24e4fc9438e6db；机器分析 artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-samebank-launch-tanh-replication-v1-analysis-s9524.json SHA256 f7733dbeb1cce650057d2284f7886b6176d6e4952759be06428506401de7f71f。Control/Tanh report SHA256 分别为 06be00b171ab29e2cd22290c690cd83210a345cbbdb1d2801ece115407d15c3a / 25801acc329dfcf2ac0726c77e1aa7946ace3d4d8079529338b39d991a2c73c6。
