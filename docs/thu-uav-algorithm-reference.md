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
