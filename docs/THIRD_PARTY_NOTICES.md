# 第三方组件与许可证

AeroWall 自有代码使用根目录 LICENSE 中的 MIT 许可证。该许可不替代上游代码、仿真运行时、模型或素材的各自条款。

| 项目 | 锁定版本 | 声明的许可证 | 角色 |
|---|---|---|---|
| VolleyBots | 10e3701e480b518041c8be6a40efee7bdcb69283 | MIT，以上游仓库 LICENSE 为准 | 通用训练与仿真能力来源；当前 upstream-lock 标记为非运行时依赖 |
| JuggleRL_train | 7ff6efb499e53b16dbaa024f2397f1101c7614ad | MIT，以上游仓库 LICENSE 为准 | 当前固定训练运行链 |
| HCSP | 009961b8f5702dd0c1c943cef0e01e09dfcd138d | MIT，以上游仓库 LICENSE 为准 | 策略和任务兼容源；版本用于检查点兼容和复现实验 |
| Orbit、rl、tensordict 子模块 | 版本见 docs/upstream-lock.json | 保留各自 LICENSE | JuggleRL 相关依赖 |

根目录 MIT 许可仅用于 AeroWall 自有源代码。随项目保留的 `launch-wall.pt`、`c-u350.pt` 检查点和演示视频有独立来源；本地审阅版本没有为这些二进制素材另行指定许可，MIT 文件不应被视为它们的许可声明。

本源码边界不包含 third_party/ 下的上游 checkout，也不包含 Isaac Sim 二进制运行时。Isaac Sim、Omniverse Kit 及其附带组件可能适用独立的 NVIDIA 条款；获取和使用前请查阅所用版本的许可：

- [NVIDIA Isaac Sim Additional Software and Materials License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/)
- [NVIDIA Isaac Sim licensing documentation](https://docs.isaacsim.omniverse.nvidia.com/2023.1.1/common/NVIDIA_Omniverse_License_Agreement.html)

获取上游源码时，请保留其许可证、版权声明和文件头。精确仓库 URL、commit、submodule 与依赖职责以 docs/upstream-lock.json 为准。
