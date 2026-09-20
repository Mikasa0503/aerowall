# AeroWall 执行状态

## 当前状态：2026-09-21 00:30（优先于下方历史记录）

- 原生 A100 渲染验证通过：960×720 RGB，前后画面随真实物理状态变化，单独 render 不推进物理；两张 PNG 已实际查看。证据 runs/singlejuggle-render-02.json、docs/render-review.json，服务器图片 artifacts/singlejuggle-render-02/。
- 已从固定 HCSP 源码复制缺失的三张地面贴图，蓝色网格恢复，缺失贴图错误消失。复制脚本 scripts/prepare_assets.py 校验固定 commit 和 SHA-256；来源清单 docs/asset-restoration.json。没有修改已有 USD、物理或控制源码。
- 默认镜头仍较远，当前图片仅证明渲染能力，不是任务演示或持续垫球成果。
- 128 环境短 PPO 已通过：3 轮更新、24,576 次转换，采样吞吐约 4,184–5,009 次/秒，单轮更新约 0.58–0.66 秒，保存/重载与 128 步评测通过。证据 runs/singlejuggle-ppo-128-01.json。
- 512 环境比较正在运行：runs/singlejuggle-ppo-512-01.json / .log，PID 2165818 已核实存活，任务句柄 57472。恢复时先核实进程，不要重复启动。
- 下一步按吞吐决定训练规模，并开展 1,000 次碰撞/时间步收敛、完整隔离覆盖与上游垫球基线训练。四阶段动作分析、WallRally、正式方法比较及 Demo 尚未完成。

## 当前状态：2026-09-21 00:21（优先于下方历史记录）

- 修复后的 16 环境 10,000 步已完成：3,881 次重置，数值有限，选择性重置后控制器与全新实例输出差异为 0。证据 runs/singlejuggle-10000-reset-safe-01.json。
- 受控接触验证通过：16 个环境均报告球与真实 bat 碰撞体接触并发生相对速度反向；环境 0/1 的球有意交叉时无碰撞事件、无水平速度变化。证据 runs/singlejuggle-contacts-03.json。尚非全部 120 个环境对的穷举测试。
- 必须在初次物理场景解析前挂接 PhysX 接触报告 API；前两次事后挂接没有报告，因此保留为失败证据。没有通过改写球轨迹构造任务成功。
- 短 PPO 已通过：3 次更新、3,072 次环境转换，actor 参数实际变化；checkpoint 重载动作误差 0，随后完成 128 步评测。证据 runs/singlejuggle-ppo-02.json，真实模型 checkpoints/singlejuggle-ppo-02.pt（约 3.3 MB）。没有持续垫球成功率结论。
- 首次 PPO 失败源于诊断预热步保留梯度图；改为 no_grad 环境步进后通过，并检查采样数据不带梯度。没有改动上游 PPO 算法。
- 当前所有上述测试进程均已结束。下一步分别验证渲染与吞吐、补全接触/隔离覆盖、开始上游垫球基线与四阶段动作分析，以及至少 1,000 次碰撞和时间步收敛验证。
- WallRally、三方法三种子公平比较、消融和视频交付仍未完成。详细门槛范围见 docs/RUNTIME_GATES.md。

## 当前状态：2026-09-21 00:06（以下旧时间段为阶段记录）

- 原始上游 16 环境已完成 10,000 策略步、3,888 次 episode 重置，有限数值检查通过。证据 runs/singlejuggle-10000-01.json 及 .exit.json。
- 选择性重置实测发现 PID 历史泄漏：重置后 done=false、is_init=true，而上游根据 done 清空历史；与全新控制器的电机输出相差 0.01747644。物理位置、速度、计数器等选择性重置检查通过。失败证据 runs/singlejuggle-reset-01.json。
- 项目适配 scripts/runtime_adapters.py 在实际 reset 回调仅清空选中环境的积分和滤波历史；没有改动上游源码、控制参数或动作接口。设计见 docs/plans/2026-09-21-controller-reset-design.md。
- 修复实测通过：runs/singlejuggle-reset-02.json 中控制器输出差异为 0；其余环境的物理状态、积分和滤波历史均未改变。
- 修复后 10,000 步复测已启动：runs/singlejuggle-10000-reset-safe-01.json / .log，PID 2135733 已核实存活，任务句柄 44514，启动时间 00:06:04。恢复时先查该进程及 JSON verdict，避免重复启动。
- 接触对象与跨环境碰撞隔离仍待受控试验，完整 16 环境门槛尚未通过；短 PPO、碰撞标定、WallRally、三方法三种子、公平评测和 Demo 全部仍在原目标范围内。
- 附带的 scene_inventory 仅是 USD authored 数据，不能当作 GPU 物理实时位置或质量；首版不可见碰撞体的空 bounds 已在后续检查脚本中修正并标明范围。

## 最新进展：2026-09-20 23:58

- 固定训练依赖已安装：TensorDict 0.4.0+a8c5397、TorchRL 0.4.0+aacf134、Orbit 0.15.9、omni_drones；Torch 保持 2.0.1+cu118。上游工作树干净，项目提交 49afbf6。
- 安装入口为 scripts/install_training_dependencies.sh；依赖选择与冻结表位于 docs/training-python-requirements.txt、docs/training-env-freeze.txt。已补齐 tomli，采用 AV 12.3.0 二进制包解决旧版源码构建失败。
- 启动器仅为项目进程预加载独立环境的 libstdc++.so.6，修复 Torch 先导入时系统旧库造成的 SQLite/ICU CXXABI_1.3.15 缺失。未改动系统库。
- runs/singlejuggle-smoke-02.json 已通过：16 环境、100 策略步、27 次重置；观测与控制器 NaN 清理前输出均有限。地面 USD 仍缺三项贴图，渲染未验收。
- 10,000 步检查正在执行：runs/singlejuggle-10000-01.json / .log，PID 2123898，任务句柄 95739；最近已完成 500 步、189 次 episode 重置，进程确认存活。恢复时先核实进程与报告，不要重复启动。
- scripts/run_probe.sh 同时核对进程退出与 JSON verdict，防止 Kit fast shutdown 将失败返回码改成 0 而误判。
- 下一步：完成当前步进测试，再做受控球拍碰撞、选择性重置和跨环境隔离验证。随机步进不能替代完整 16 环境门槛；PPO、物理标定、WallRally、正式比较和 Demo 仍未完成。

以下保留上一部署阶段记录；其中“训练依赖尚未安装”已被上述新证据更新。

更新时间：2026-09-20 23:48（Asia/Shanghai）。阶段：第 1 周隔离部署与运行时门槛。完整项目目标尚未完成。

## 已验证

- 项目：`/home/public/Workspace/shy/Code/aerowall`；独立 Python 3.10.21 环境：`/home/public/Workspace/boweiy/mambaforge/envs/aerowall`。
- 三项上游及主干子模块固定版本，见 `docs/upstream-lock.json`；未导入或复用旧穿缝项目环境、代码或运行时。
- JuggleRL 主干保持原样。动作分析已完成文献/源码准备，尚未完成实际 checkpoint 四阶段重放，见 `docs/ACTION_ANALYSIS.md`。
- 已直接下载并隔离解压作者 README 链接的 Isaac Sim 包；没有修改系统驱动、默认编译器或全局 shell。
- 138 上 8 项路径与解压保护测试通过。归档 115,011 个条目完整审核，无危险路径/链接/特殊文件，gzip 完整性通过；安全解压 115,009 个项目内条目。
- 最小 GPU 物理测试 `runs/gate1-native-02.json` 通过：A100、Torch 2.0.1+cu118、GPU dynamics 开启、200 步、4 次反弹，CUDA 状态有限且在界内。此结果仅证明一个球和地面的最小物理路径，不能替代上游无人机、16 环境隔离、PPO 或渲染验收。
- 运行时使用原版 `omni.isaac.sim.python.kit`。Kit 实际日志位于项目 `.cache/kit/logs/Kit/Isaac-Sim/2023.1/`。HOME 未改动；启动器移除了作者脚本加入的空搜索路径与项目外父目录，保持运行时内部库顺序。

## 下载与安装证据

- 来源：HCSP / VolleyBots 作者固定 README 同一链接，Google Drive 文件 ID `1Rt4B3U3nGtnvqXrzTAEa6JcH5OaMxqfY`。
- 文件：`.cache/downloads/isaac-sim-author-2023.1.0-hotfix.1.tar.gz`，10,246,767,862 字节。
- SHA-256：`dac8addc9407ad4ca2b39ab01c1655b1d88ff3a880cb4e7266fef235fcdc5804`。
- 下载、审核与解压记录：`runs/runtime-download.json`、`runs/runtime-archive-audit.json`、`runs/runtime-extraction.json`。
- 隔离位置：`third_party/isaac-sim-2023.1.0-hotfix.1`，约 20 GB。没有恢复归档中的作者 home 目录。
- 实际包版本：`2023.1.0-rc.42+2023.1.667.d641947f.tc`，构建时间 2023-11-01。此内部版本亦见 [IsaacLab issue 225](https://github.com/isaac-sim/IsaacLab/issues/225) 的 hotfix.1 路径记录；这不是供应链真实性证明。
- 原始许可证文件完整保留在 `PACKAGE-LICENSES/`。Isaac Sim 运行时许可证不等同于上游训练源码的 MIT 许可证。
- HTTPS 校验始终开启。作者镜像没有独立官方哈希，当前检查能证明下载/布局完整性，不能证明官方签名认证。

## 依赖与当前验证

- 第一次测试因缺少 NumPy、typing_extensions、Pydantic、attrs，在初始化阶段停止；只终止本任务进程，记录 `runs/gate1-native.json` / `.log`。
- 已在独立环境安装固定版本，见 `docs/runtime-python-requirements.txt`。botocore 版本按运行时内置 boto3 1.26.63 的依赖选择；详细冻结表 `docs/runtime-env-freeze.txt`。
- 第二次测试通过，但初始化日志包含补齐 botocore 前的扩展错误，第三次干净进程复核也已通过，日志 Error 为 0。
- 第三次测试：`runs/gate1-native-03.json` / `.log`，退出码 0，200 步耗时 0.527 秒，4 次反弹；进程已正常退出。
- 训练依赖 TensorDict/TorchRL/Orbit 及 omni_drones 还未安装完成；目前没有训练 checkpoint。

## 磁盘与网络

- 旧项目约 13 GB，旧 uav_gap 环境约 9.6 GB；未删除或迁移。
- 工作盘 `/dev/sdb` 总计 3.6T，下载后约 355G 空闲（解压还会占约 20G）；`/data` `/dev/sda1` 仍约 3.4T 空闲。用户记忆中的 3T 对应另一个挂载点。
- 本任务反向代理使用项目 Unix socket（600 权限）和 `127.0.0.1:17891` 桥接 Mac 代理。下载已结束；未来需要时先验证通道。

## 下一步与未完成范围

1. 安装上游固定依赖而不升级 Torch；最小物理测试已复核通过。
2. 按上游实际 SingleJuggle 启动配置执行 16 环境至少 10,000 策略步，检查重置、接触、有限数值和跨环境隔离。
3. 短 PPO 收集/更新/保存/重载；分别测试训练与渲染；通过后按吞吐扩大规模。
4. 四阶段实测、至少 1,000 次碰撞和时间步收敛验证、WallRally、三方法三种子、公平评测、消融、连续视频与 Demo 均待完成。

最小物理通过不等于计划完成，不等于上游训练复现成功，也没有壁球成功率结果。
