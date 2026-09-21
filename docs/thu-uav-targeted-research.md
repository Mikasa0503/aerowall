# thu-uav 三个项目针对性研究

本次研究对象按最近讨论确定为 **HCSP、SimpleFlight、NeuralIMC**。原实施计划中的 JuggleRL_train、VolleyBots 已在项目 third_party 内；它们仍是 AeroWall 的主干与任务参考，本次不替换主干。

目标是解释当前问题并确定可验证的改进：拟合后的单 actor 为 3/100 接回，完整 PPO 第 25/50 次更新降为 0/100；相同留出状态上出球与恢复动作均发生偏移。以下将源码事实、迁移判断、待做实验分开。没有把下载或源码阅读算作算法复现成功。

## 下载与复现边界

服务器研究目录：`/home/public/Workspace/shy/Code/aerowall/research/thu-uav/`。Mac 镜像保存在本任务 `work/thu-uav/`。完整 Git 仓库及锁定子模块均下载，不采用仅下载 README 或浅克隆的方式；外链数据、仿真运行时、未公开权重不属于 Git 下载范围。

| 项目 | 固定 commit | 依赖与权重 |
| --- | --- | --- |
| HCSP | `009961b8f5702dd0c1c943cef0e01e09dfcd138d` | Orbit、TorchRL、TensorDict 三个子模块；含低层、高层、自博弈权重 |
| SimpleFlight | `f5ae8fc3689edc7c982e07cbfd587eeeab72f279` | TorchRL、TensorDict 两个子模块；本次未发现 tracked `.pt/.pth/.ckpt` 文件 |
| NeuralIMC | `28b205d511089fbb402f9b0ee78c8931692195b5` | 无注册子模块；本次未发现 tracked `.pt/.pth/.ckpt` 文件 |

三仓库根许可证均为 MIT，依赖各自保留许可证。完整版本记录见同目录《thu-uav源码版本清单.json》。未安装它们的依赖到当前训练环境，没有修改现有 third_party/HCSP。

下载先尝试 138，Git 代理连接阻塞后中止了本次发起的三个 clone，改为 Mac 下载并同步。SimpleFlight 旧 SSH 子模块地址不能直接获取：TorchRL 从已下载 HCSP 的 Git 对象库取得同一锁定 commit `e39e701…`；TensorDict 从 pytorch/tensordict 获取同一 `5e6205c…`。没有用最新版替代锁定版本。

## HCSP：最优先研究技能衔接与微调约束

**源码事实。** `mappo_Receive_hover.py` 创建各角色技能，`train_op` 只更新 FirstPass_hover；环境根据击球与 turn 选择阶段。`Receive_hover.py` 在实际击球之后启用 pose/up/spin 恢复奖励，并保留接地、碰网等终止。这支持“在真实击球后的状态训练恢复”，不支持用独立平稳悬停数据代替碰后状态。

- [角色与训练入口](https://github.com/thu-uav/HCSP/blob/009961b8f5702dd0c1c943cef0e01e09dfcd138d/hcsp/learning/volleyball/mappo_Receive_hover.py#L35)
- [击后恢复奖励](https://github.com/thu-uav/HCSP/blob/009961b8f5702dd0c1c943cef0e01e09dfcd138d/hcsp/envs/low_level_skill/Receive_hover.py#L704)

**与当前退化最相关的新发现。** 联合阶段不仅有策略分解，还保留冻结参考 actor。`mappo_mask_kl.py` 计算 **KL(current || reference)**，参考分布 detach；环境 `coselfplay_phase_one.py` 从策略结果取得 KL，并在 SecPass 奖励中减去系数乘 KL。shell 使用系数 `1e-4`。不能只读 `update_actor` 就判断没有 KL，也不能误说它直接加在 PPO actor loss 上。

- [参考分布与 KL 方向](https://github.com/thu-uav/HCSP/blob/009961b8f5702dd0c1c943cef0e01e09dfcd138d/hcsp/learning/mappo_mask_kl.py#L512)
- [环境中的 KL 惩罚](https://github.com/thu-uav/HCSP/blob/009961b8f5702dd0c1c943cef0e01e09dfcd138d/hcsp/envs/co_self_play/coselfplay_phase_one.py#L2337)
- [实际启动配置](https://github.com/thu-uav/HCSP/blob/009961b8f5702dd0c1c943cef0e01e09dfcd138d/scripts/shell/coselfplay_phase_one.sh)

**迁移判断。** AeroWall 已有同状态行为漂移证据，优先试“保留拟合单 actor 作为参考，对全阶段 PPO 添加可追踪的偏离约束”，保持单 actor 执行、CTBR 与真实碰撞。AeroWall 现有拟合损失是 KL(teacher || student)，方向与 HCSP 上述项不同，不能混称同一算法。参考约束只能帮助保留已有 3% 行为，不能保证达到 70%；仍须学习更好的完整回合。

**待做实验。** 从相同拟合 checkpoint、seed、场景和新交互预算出发，对照无约束 PPO 与一种预先固定的参考约束；记录两阶段 KL、实际 actor 梯度、回合/目标命中、终止原因、每步奖励分解。若添加直接可微 KL loss，应明确为本项目变体，不冒充原 HCSP 环境奖励实现。必须确认参考参数与优化器分离，checkpoint 重载后参考不变，评价阶段不更新权重。

## SimpleFlight：控制与平滑奖励的条件，不是直接换策略

**源码事实。** 默认 Track 为 Crazyflie、`PIDrate`、10 个未来参考点；train.py 的 `PIDrate` 与 `PIDrate_FM` 是不同控制器分支。当前 AeroWall 使用 Flightmare PID/CTBR，不能因为两者都称 CTBR 就交换动作尺度、惯性和控制增益。

`track.py` 用轨迹相对位置、线速度、机体状态组成观测。未来点是已知目标轨迹，不是仿真器未来真实运动；壁球只能使用已发布目标或基于当前观测的预测参考。

- [观测与参考点](https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/omni_drones/envs/single/track.py#L369)
- [两个 PID 入口](https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/scripts/train.py#L170)

**默认配置与可用机制必须分开。** 代码支持随 count 增加平滑、加速度等权重，但 Track.yaml 的平滑初值与上限均为 2，因此默认实际固定；加速度、jerk、snap 的上限为 0，因此默认不开启这些项。不能照搬论文术语声称默认运行具备递增平滑课程。随机化示例在该 YAML 中是注释，也不能据此宣称默认已启用。

- [奖励计算](https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/omni_drones/envs/single/track.py#L473)
- [真实默认参数](https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/cfg/task/Track.yaml)

**迁移判断。** 适合参考预测回位目标、动作历史与控制响应诊断。避免全阶段提高直立/低 jerk 奖励：击球需要转向，增加此类奖励可能进一步偏好垫球。若做恢复期平滑课程，应记录实际随环境步变化的系数并按时间归一化。历史缓冲必须沿用本项目已验证的逐环境 reset，不能直接复制全 batch deque 的更新逻辑。

**待做实验。** 在现有真实轨迹上先测出球/恢复期的 CTBR 饱和占比、角速度跟踪误差与恢复耗时，确认控制限制后才改增益或奖励。此时不增加额外大规模轨迹跟踪训练，也不换 Crazyflie 模型。

## NeuralIMC：一步预测误差反馈及其信息权限

**源码事实。** 默认环境创建 `RigidBody(dt=control_dt)`；独立学习预测模型默认 `enable:false`。刚体预测输入上一状态和实际处理后的 CTBR，积分推力、重力、姿态，预测下一状态；`get_dyn_err` 形成位置、速度、旋转残差。`ours.yaml` 默认短历史长度 1，内容包含 state/action/ref_err/dyn_err，长历史长度 0。不能将默认机制描述为必须先训练 Transformer 或长历史 GRU。

- [一步刚体预测](https://github.com/thu-uav/NeuralIMC/blob/28b205d511089fbb402f9b0ee78c8931692195b5/torch_control/predictive_models/models/prior/rigid_body.py#L10)
- [实际误差构造](https://github.com/thu-uav/NeuralIMC/blob/28b205d511089fbb402f9b0ee78c8931692195b5/torch_control/tasks/base.py#L427)
- [方法观测配置](https://github.com/thu-uav/NeuralIMC/blob/28b205d511089fbb402f9b0ee78c8931692195b5/torch_control/configs/controller/ours.yaml)

**信息权限发现。** `ours.yaml` 开启 extrinsics；`get_obs_extrinsics` 在风关闭时返回零，在 `use_l1ac:true` 时返回估计，否则返回真实 wind_vec。主实验脚本开启 wind，而基础配置 `use_l1ac:false`。因此必须审查解析配置，不能笼统声称所有默认风实验都只使用可观测历史。迁移至 AeroWall 时关闭真值外参通路，碰撞冲量、真实弹性和随机化标签只能用于诊断或单列 oracle。

- [外参三种返回路径](https://github.com/thu-uav/NeuralIMC/blob/28b205d511089fbb402f9b0ee78c8931692195b5/torch_control/dynamics/quadrotor/quadrotor.py#L115)
- [主实验组合](https://github.com/thu-uav/NeuralIMC/blob/28b205d511089fbb402f9b0ee78c8931692195b5/scripts/train/shell_scripts/main.sh)

**迁移判断。** 先用于诊断碰撞前后无人机运动预测残差。该残差同时混合碰撞反作用、执行器动态、时间延迟和模型失配，不能直接解释为墙面弹性估计。墙面弹性仍由球的碰前后法向速度估计器承担。

**待做实验。** 使用已记录的观测与动作重放一步模型；先在无碰撞片段校准，再比较碰撞窗口。逐环境重置历史、对齐动作延迟/单位/控制频率，明确观测时间戳。通过后才考虑作为额外消融输入；不能悄悄改变原计划 FF/GRU/Estimator 的公平信息条件。

## 决策与下一步顺序

1. **优先 HCSP 参考策略约束**：直接对应已量化的行为漂移，保持主任务/物理不变。先验证学习信号与冻结参考，再做等新预算开发对照。
2. **同时做 SimpleFlight 式控制诊断**：确认恢复失败是否存在持续 CTBR 饱和和姿态跟踪不足；结果决定是否需要独立控制接口实验。
3. **最后引入 NeuralIMC 残差**：先离线验证时间对齐及信息权限，证明有解释价值再训练，不直接扩展主方法清单。

三个项目均已完成有针对性的入口、配置和关键数据流阅读；不声称逐行审计整个仓库，也不声称 SimpleFlight/NeuralIMC 已运行复现。HCSP 原任务实测已单列在《AeroWall-HCSP预训练策略核验》，其命中率不计入壁球接回指标。
