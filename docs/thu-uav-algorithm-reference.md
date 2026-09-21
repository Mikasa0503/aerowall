# AeroWall：thu-uav 算法参考与实施方向

用户要求算法方面参考 thu-uav。现有 HCSP/VolleyBots 已用于上游环境与策略诊断；接下来加强技能训练与恢复机制的源码对照。以下迁移方案是 AeroWall 的设计判断，不是原论文已验证的壁球结论。

## 优先：HCSP 的击球后恢复技能

HCSP 把低层技能、高层策略和联合训练分阶段处理。论文：https://arxiv.org/abs/2505.04317 。本地固定源码版本为 009961b8f5702dd0c1c943cef0e01e09dfcd138d。

实际检查 `hcsp/envs/low_level_skill/Receive_hover.py`：704 行起计算恢复奖励；727 行起包含位置、机体朝上和旋转项；788 行通过 `FirstPass_hit` 选择击球后的奖励。配置 `cfg/task/Receive_hover.yaml` 显式定义击球后悬停位置；`cfg/train_receive_hover.yaml` 的训练预算为 200000000 帧。这是专门训练恢复技能的具体实现，不能把我们的短课程预算当作等量复现。

AeroWall 的下一项候选实验：在真实合法击球后启用恢复/接回课程，参考位置应由当前可观测球状态、已知墙几何和估计反弹参数确定；若用固定回位点，只作为单独恢复诊断。先训练技能再接入持续回合，所有预训练步计入公平预算。恢复中的姿态奖励仅在明确阶段启用，不把全程小倾角或固定翻滚限制作为成功条件。优先比较同一固定开发集的真实接回、失高、碰墙与目标命中。原有 FF/GRU/Estimator 三方法比较范围保留。

源码链接：https://github.com/thu-uav/HCSP/blob/009961b8f5702dd0c1c943cef0e01e09dfcd138d/hcsp/envs/low_level_skill/Receive_hover.py

## SimpleFlight：跟踪输入、动作与训练设置

官方仓库 https://github.com/thu-uav/SimpleFlight 和论文 https://arxiv.org/abs/2412.11764 使用 PPO、CTBR、系统辨识和选择性随机化，并讨论输入、奖励与训练技术。后续检查其轨迹跟踪输入和动作平滑设计与现有 CTBR 实现的差异，用于学习接回所需的快速回位。当前只核验了仓库说明与论文摘要，尚未完成这些机制的源码复现。其 Crazyflie 参数与旧 Isaac Sim 依赖不直接覆盖本项目 Iris+球拍动力学。

## NeuralIMC：预测误差反馈

官方仓库 https://github.com/thu-uav/NeuralIMC ，论文 https://arxiv.org/abs/2411.13079 。其预测误差反馈可作为碰后动力学扰动诊断的参考。候选 AeroWall 实验将仅用历史动作和观测构造预测残差，检查碰撞后控制失配；不会把真实材质或未来真实轨迹提供给 actor。此项尚未实现，也不替代现有显式有效弹性估计器。

当前已启动的奖励平衡训练继续完成，作为可比较的前馈基线。后续按技能恢复、跟踪输入、预测误差反馈的顺序评估，每项需单独验证；这些参考不代表当前 0/100 接回问题已经解决。


## SimpleFlight 源码核验

已只读检出官方仓库，固定提交 `f5ae8fc3689edc7c982e07cbfd587eeeab72f279`，只获取配置、单机环境与学习源码，没有安装其依赖或加载其飞控权重。

核验 `omni_drones/envs/single/track.py` 的 383–401 行：输入包含多点参考轨迹相对于机体的位置，以及速度和姿态表示。默认 `cfg/task/Track.yaml` 设置 `future_traj_steps: 10`，轨迹采样步距为 5，默认 dt=0.01 秒，因此包含当前点至约 0.45 秒后的参考点。这些是任务给定的参考轨迹，不是测得的未来机器人状态。迁移到 AeroWall 时，可用当前观测和估计反弹参数构造预测参考，但不可读取仿真器未来真实球轨迹或尚未公布的目标序列。

473–515 行实现距离、姿态、旋转、动作平滑等奖励。默认 Track 配置动作平滑权重初值和上限同为 2，故默认不会逐步增大；动作幅值、加速度、jerk、snap 奖励上限均为 0。动作历史虽有代码入口，默认 `use_action_history: false`，不能宣称默认策略使用了动作历史。后续若引入平滑课程，应明确阶段和权重，验证快速击球是否因此受损。

默认配置为 8192 并行环境、每批 64 步，即 524288 条交互；当前壁球开发为 128×64=8192 条。相同 PPO 更新次数不代表相同数据预算，后续仍统一报告真实交互步，不能用更新次数比较效果。SimpleFlight 的 Crazyflie、100 Hz 和本项目 Iris+拍面、50 Hz 不同，权重和奖励数值不直接复用。

固定源码链接：https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/omni_drones/envs/single/track.py
配置链接：https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/cfg/task/Track.yaml

当前优先完成 HCSP 启发的阶段恢复对照，再决定是否增加预测参考输入。上述是已完成的源码检查和迁移边界，尚不是新增控制效果证据。
