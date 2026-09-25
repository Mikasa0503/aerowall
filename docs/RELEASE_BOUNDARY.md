# 项目文件与发布边界

本文件说明仓库中哪些内容是 AeroWall 自有交付，哪些内容需要单独准备。当前文件树为本地审阅准备版本，未形成公开发行包。

## 审阅与复现面

- AeroWall 自有 Python 源码、脚本、固定配置、测试和研究文档。
- 当前最佳基线的两个小检查点，保留原文件名：checkpoints/launch-wall.pt、checkpoints/c-u350.pt。
- 当前验收演示：media/AeroWall-wall-rally-demo.mp4。
- C350 结果机器可读摘要：docs/results/c350_baseline.json。
- 上游提交锁：docs/upstream-lock.json。

## 本机或服务器工作区

以下目录和产物不属于源码复现包，并由 .gitignore 排除：

- .cache/：Conda、Isaac Sim、CUDA 和渲染缓存。
- third_party/：上游 checkout 与仿真运行时；需按锁定版本单独获取。
- work/：临时 worktree、迁移备份、中间实验工作区。
- runs/：原始训练检查点、日志、rollout、RGB 帧和仿真报告。
- artifacts/：临时分析图、诊断材料和中间视频。
- research/thu-uav/：本地参考源码镜像。
- checkpoints/ 中除 launch-wall.pt 和 c-u350.pt 外的训练中间模型。

完整原始报告及逐事件数据保留在本地运行工作区。结果摘要保存必要指标和原始报告哈希；通过复现指南可使用发布面检查点重新运行正式评测。
