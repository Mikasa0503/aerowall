# 上游源码与配置核查（运行前）

日期：2026-09-20。以下为固定源码的静态证据，不代表策略已重放或物理已验证。

## 来源与用途

精确版本及子模块记录在 `upstream-lock.json`。三个仓库均保留完整 Git 元数据、LICENSE、资产及源码内版权声明。

- JuggleRL_train：MIT；主干实现，完整初始化其 Orbit/TensorDict/TorchRL 子模块。
- VolleyBots：保留上游 LICENSE；只作任务、规则与评测参考，不安装第二套同名 omni_drones。
- HCSP：保留上游 LICENSE；只作技能衔接和动作分析参考，不混用 PRT 与 CTBR checkpoint。
- 所有第三方修改当前为空；未复制或导入 uav-visual-gap 的代码、运行时或环境。
- 参考仓库的子模块 commit 由 Git tree 固定，未重复安装；共享的是版本选择，不是旧项目可变目录。

## JuggleRL 默认 YAML 与实际启动入口

证据：`scripts/shell/singlejuggle_sim2real.sh`、`cfg/task/SingleJuggle.yaml`、`cfg/base/sim_base.yaml`。

| 字段 | 默认 YAML | 上游启动脚本 |
|---|---|---|
| drone_model | Air | air |
| action_transform | PIDrate_FM | PIDrate_FM |
| sim.dt / substeps | 0.02 / 1 | 0.02 / 继承 1 |
| ball_mass / ball_radius | 0.0472 kg / 0.04 m | 相同 |
| num_envs | 4096 | 4096（部署验证需覆盖为 16） |
| drone_vel_latent_step / ball_linear_vel_latent_step | 2 / 2 | 0 / 0 |
| true_hit_step_gap | 35 | 30 |
| use_distance_clip | true | false |
| use_spin_reward | true | false |
| use_velocity_penalty | true | false |
| use_yaw_penalty | false | true |
| restitution randomization | false, [0.7, 0.95] | true, [0.7, 1.0] |
| wandb.mode | 见 train.yaml | online（项目验证改为 offline） |

不能把上表当作已经导出的实际仿真配置。成功启动后必须保存 Hydra 解析配置、运行时质量/惯量和碰撞形状，并核对命令覆盖项。

`omni_drones/robots/assets/usd/air.yaml` 写有 `update_sim: True`、质量 0.9505 kg、惯量对角约 (0.00529025, 0.00562380, 0.00780462) kg·m²。`multirotor.py` 中初始化代码会设置质量和惯量。USD 是二进制资产；拍面真实碰撞几何及最终物理属性仍需用运行时读取，不从奖励检测半径反推几何尺寸。

## 实现风险及后续核查

1. `scripts/train.py` 的 `PIDrate_FM` 分支使用 Flightmare PID 控制器。保持这个实际接口及限幅，不能因为注释都称 CTBR 就切换到其他控制器。
2. `single_juggle.py:check_hit` 用邻近区域、球速变化和时间间隔判断击球。它可以作为上游基线，不能直接充当 WallRally 接触对象核验及进入/离开去重规则。
3. `learning/mappo.py` 已有 GRU、is_init 与序列 minibatch 逻辑。后续必须实测 episode 重置、截断边界和序列状态传递，再决定最小改动。
4. `init_simulation_app` 在 headless 模式仍选择 `omni.isaac.sim.python.kit`。无窗口不等于无图形依赖，必须在 A100 上实测。
5. `setup.py` 没有锁定所有依赖；`train.py` 还直接导入 setproctitle、matplotlib、tqdm 等。运行时到位后对照其内置 PyTorch 固定兼容依赖，不能直接升级到最新 TorchRL。
6. 四阶段动作分析尚未完成：源码、论文说明与实测轨迹必须分开标注；拍面接触点速度采用同坐标系的 `v_com + omega × (p_contact - p_com)`。禁止将图示或推断标注为已重放行为。

## 下一步门槛

先获取并核实 2023.1.0-hotfix.1 Linux x86_64 包，运行最小 GPU 物理初始化；随后才执行 16 环境 × 10,000 策略步、短 PPO 更新/保存/重载和物理标定。当前没有训练 checkpoint，也没有任何壁球成功率证据。
