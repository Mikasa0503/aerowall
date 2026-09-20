# AeroWall 执行状态

更新时间：2026-09-20 23:17（Asia/Shanghai）。总体阶段：第 1 周部署准备，项目目标尚未完成。

## 已完成且有证据

- 138 SSH 连接、Ubuntu 20.04、A100 40 GB、驱动及 GPU 占用核查；启动本轮下载前 GPU 显存约 13 MiB。
- 创建项目 `/home/public/Workspace/shy/Code/aerowall` 和独立 Python 3.10.21 环境 `/home/public/Workspace/boweiy/mambaforge/envs/aerowall`。
- 修复本项目 Conda 继承失效代理的问题：使用项目 `.cache/condarc`，未修改全局配置。
- 固定并同步 JuggleRL_train、VolleyBots、HCSP 与主干三项子模块；预检 `source_locks_match: true`。
- 验证 Python 前缀正确、用户 site-packages 禁用，路径未混入旧穿缝项目或 uav_gap 环境。
- 添加项目启动器、预检、下载记录和压缩包路径检查；5 个路径检查测试通过，shell 语法与 Python 编译检查通过。
- 旧穿缝项目约 13 GB，uav_gap 环境约 9.6 GB。工作盘剩余约 365 GB；另一个挂载点 `/data` 剩余约 3.4 TB。没有删除或迁移旧项目；存储布局仍按原计划。

- 已完成动作分析的文献/源码准备，记录 CTBR 实际单位、饱和链、四阶段测量定义和必须补齐的轨迹证据，见 `docs/ACTION_ANALYSIS.md`。未将其计为 4.1a 实测通过。

## 正在执行

下载由 HCSP 和 VolleyBots 作者 README 链接的 `isaac-sim.tar.gz`，HTTP 元信息大小 10,246,767,862 字节。初步只读目录检查可见 `home/chenyinuo/isaac-sim/isaac-sim-2023.1.0-hotfix.1/`。

- 下载目标：`.cache/downloads/isaac-sim-author-2023.1.0-hotfix.1.tar.gz.part`
- 下载状态：`runs/runtime-download.json`
- 日志：`runs/runtime-download.log`
- 当前下载父进程 PID：2074626，curl PID：2074630；23:17 核实仍存活，已下载 3,132,272,640 字节（约 30.6%）。
- SSH 反向通道：`.cache/mac-proxy.sock`，属主 boweiy，权限 600。
- 项目代理桥：`127.0.0.1:17891`，连接 Mac 的 `127.0.0.1:7890`；已经验证服务器经该通道获得 NGC 的正常 HTTP 401 认证挑战响应。
- 下载只写缓存，不解压、不执行安装脚本。HTTPS 证书验证保持开启。

恢复工作时先核实上述 PID/下载脚本及文件增长；PID 不可仅凭记录视为仍运行。Mac 隧道与代理桥也必须检查。不要在原下载仍运行时重复启动。

## 来源与剩余验证

- NVIDIA NGC 的 2023.1.0-hotfix.1 标签请求经认证后为 404，当前公开列表不含 2023 系列；网络连通与版本可获得性是两项不同检查。
- 作者入口：<https://github.com/thu-uav/HCSP> README 第 32 行、VolleyBots README 第 58 行（固定版本见 lock）。二者链接同一文件：<https://drive.google.com/file/d/1Rt4B3U3nGtnvqXrzTAEa6JcH5OaMxqfY/view>。
- 用户提供的 CSDN 帖子也可定位夸克上的 Linux 2023.1.0-hotfix.1 ZIP，但需要登录；当前未使用该包。
- 作者镜像无独立可核对的官方哈希。下载 SHA-256、大小、gzip 完整性和目录检查只证明本次产物与传输/布局情况，不证明供应链真实性。

## 下一步

1. 核实下载完成状态及 SHA-256；执行 `scripts/audit_runtime_archive.py`，审查包布局、链接和必要文件。不得直接恢复压缩包内的作者 home 路径。
2. 核实许可证、运行时版本、启动脚本、依赖与资产路径；仅在项目内隔离部署运行时。
3. 首先实测 A100 的最小 GPU 物理初始化，再执行 16 环境至少 10,000 策略步，短 PPO 更新/保存/重载，以及训练/渲染能力分离测试。
4. 上游动作四阶段实测、至少 1,000 次碰撞及时间步收敛验证随后进行。完整 WallRally、三方法三种子、公平评测、消融与 Demo 均仍待实现。

当前没有训练 checkpoint、已验证碰撞结果、回合成功率或 Demo；不能把部署预检当作训练验收。
