# AeroWall

AeroWall 是一个用于研究单无人机对墙连续击球的仿真项目。项目基于 VolleyBots 的通用训练与仿真能力，围绕壁球式击球任务构建 AeroWall 自有环境、策略路由、奖励、扰动课程和物理接触审计。运行依赖的确切上游代码版本见 [docs/upstream-lock.json](docs/upstream-lock.json)。

当前经验证效果最好的方案是 **冻结 Launch 发球策略 → C350 持续回接策略**。该方案沿用原策略名和检查点，未将尚未晋升的实验候选算作最终能力。

## 当前最佳方案与结果

| 协议 | 结果 | 解释 |
|---|---|---|
| 正式批量评测 | 128/128 场均完成 17 轮；均运行到 1,200 个策略步的评测上限；无安全失败 | 证明 C350 在该正式协议内可重复完成至少 17 轮。时间上限结束不等于策略自然失败 |
| 展示回放 | 演示种子 5001 完成 25 轮，在第 1,840 步球落地；26 次拍面和 26 次墙面接触均通过物理审计 | 单次展示结果，不代表跨种子可靠性 |

C350 使用的模型为 [c-u350.pt](checkpoints/c-u350.pt)，冻结发球模型为 [launch-wall.pt](checkpoints/launch-wall.pt)。检查点哈希、评测报告来源及适用范围见 [模型卡](docs/MODEL_CARD.md) 与 [结果记录](docs/RESULTS.md)。 文件完整性清单为 [SHA256SUMS](SHA256SUMS)。验收演示视频为 [AeroWall-wall-rally-demo.mp4](media/AeroWall-wall-rally-demo.mp4)。

## 项目贡献与研究边界

AeroWall 在上游通用仿真与训练能力上自行构建单无人机对墙击球任务，包含冻结发球与持续回接技能链、壁球场景观测、连续回合和回中奖励、分阶段训练与扰动课程，以及实验和演示证据。更完整的贡献归属与拓展状态见 [研究状态](docs/RESEARCH_STATUS.md)。

多样初始来球上的首击已有改善，但持续回合能力仍未达标。Launch／Hit／Recover 拆分方案已经实现并训练，尚未证明优于 C350 双策略。新观测和因果奖励版本有工程与诊断价值，整体性能收益尚未成立。Tanh／有界动作实验减少了执行器限幅，尚未证明稳定提高续拍。reset、策略交接、执行器和模型身份审计提升了实验可信度，属于工程成果。高层策略和共同适应尚未完成，不计作已有贡献。

## 快速开始

目标运行环境为 Linux x86_64、NVIDIA GPU、Python 3.10 和 Isaac Sim 2023.1.0-hotfix.1。Isaac Sim 运行时和上游源码不随项目源码分发；按各自许可证单独获取。详细步骤见 [复现指南](docs/REPRODUCIBILITY.md)。

    export AEROWALL_CONDA_BIN=/path/to/conda
    export AEROWALL_ENV=/path/to/aerowall-python-3.10
    export AEROWALL_ISAACSIM_PATH=/path/to/isaac-sim-2023.1.0-hotfix.1

    bash scripts/create_env.sh
    bash scripts/install_training_dependencies.sh
    bash scripts/python.sh --plain scripts/preflight.py --output runs/preflight.json

## 复现 C350

在完成锁定版本运行环境配置后，正式批量评测命令如下。报告写入被忽略的 runs 目录，不会覆盖已提交的结果记录。

    bash scripts/python.sh scripts/evaluate_aerowall_wall_rl.py --output runs/c350-formal-reproduction.json --checkpoint checkpoints/c-u350.pt --launch-checkpoint checkpoints/launch-wall.pt --num-envs 128 --seed 3001 --stage RALLY --experiment-config configs/recenter_recovery.json --protocol natural

展示协议的单场自然落地回放使用 configs/c350_presentation_4800.json、种子 5001 和 --num-envs 1。预期对照范围及其限制记录于 docs/REPRODUCIBILITY.md。

## 仓库结构

- aerowall/：AeroWall 自有的环境、物理接触处理、回合逻辑、策略和奖励模块。
- scripts/：训练、评测、环境检查和诊断入口；以复现指南列出的入口为主。
- configs/：锁定的任务配置、案例库和实验设置。
- checkpoints/：仅保留可复现 C350 基线所需的两个原始命名检查点。
- media/：验收演示视频。
- docs/：贡献边界、结果、模型信息、上游版本和实验记录。

本项目是依赖 Isaac Sim 与锁定上游源码的研究代码仓库，不是可通过单独 pip install 安装并运行的独立仿真发行包。

## 文档

- [文档索引](docs/INDEX.md)
- [研究贡献与进展](docs/RESEARCH_STATUS.md)
- [正式结果与限制](docs/RESULTS.md)
- [C350 模型卡](docs/MODEL_CARD.md)
- [运行复现指南](docs/REPRODUCIBILITY.md)
- [发布边界](docs/RELEASE_BOUNDARY.md)
- [上游版本锁](docs/upstream-lock.json)
- [第三方许可说明](docs/THIRD_PARTY_NOTICES.md)
- [实验状态与历史进度](STATUS.md)

## 许可证

AeroWall 自有代码采用 MIT 许可证，见 [LICENSE](LICENSE)。VolleyBots、JuggleRL、HCSP、Isaac Sim 及其他依赖保留各自的许可证与条款，详见 [第三方许可说明](docs/THIRD_PARTY_NOTICES.md)。
