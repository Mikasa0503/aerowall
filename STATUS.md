# AeroWall 执行状态

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
