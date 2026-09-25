# 复现指南

## 运行依赖

- Linux x86_64 与 NVIDIA GPU。
- Python 3.10 隔离环境。
- Isaac Sim 2023.1.0-hotfix.1；需从 NVIDIA 单独获取并按其许可使用。
- `docs/upstream-lock.json` 中记录的固定源码提交与子模块。
- C350 基线文件 checkpoints/launch-wall.pt 与 checkpoints/c-u350.pt。

上游代码、Isaac Sim 运行时和本机缓存不放在仓库源码里。按以下步骤获取锁定代码；不要用未经记录的 HEAD 替代。安装依赖脚本需要 GCC/G++ 9。环境参考快照见 [runtime-env-freeze.txt](runtime-env-freeze.txt)、[training-env-freeze.txt](training-env-freeze.txt) 和 [conda-linux-64-explicit.txt](conda-linux-64-explicit.txt)。

## 获取固定上游源码

在仓库根目录执行。JuggleRL_train 是运行时训练依赖；HCSP checkout 用于锁定检查点兼容来源及恢复校验过的缺失纹理。VolleyBots 在 upstream-lock 中用于来源追溯，当前不作为运行时依赖。

    mkdir -p third_party
    git clone https://github.com/thu-uav/JuggleRL_train third_party/JuggleRL_train
    git -C third_party/JuggleRL_train checkout --detach 7ff6efb499e53b16dbaa024f2397f1101c7614ad
    git -C third_party/JuggleRL_train submodule update --init --recursive

    git clone https://github.com/thu-uav/HCSP third_party/HCSP
    git -C third_party/HCSP checkout --detach 009961b8f5702dd0c1c943cef0e01e09dfcd138d
    git -C third_party/HCSP submodule update --init --recursive

可选地获取 VolleyBots 来源快照：

    git clone https://github.com/thu-uav/VolleyBots third_party/VolleyBots
    git -C third_party/VolleyBots checkout --detach 10e3701e480b518041c8be6a40efee7bdcb69283
    git -C third_party/VolleyBots submodule update --init --recursive

用 `git -C <checkout> rev-parse HEAD` 和 `git -C <checkout> submodule status --recursive` 对照 `docs/upstream-lock.json` 核验来源。获取代码前应阅读上游各自的许可证。

## 文件完整性

在仓库根目录用 `sha256sum --check SHA256SUMS` 核验纳入复现包的检查点、视频和正式配置。结果 JSON 还记录来源运行报告与配置的 SHA-256。

## 环境准备

先设置本机路径并取得与项目版本匹配、受单独许可的 Isaac Sim：

    export AEROWALL_CONDA_BIN=/path/to/conda
    export AEROWALL_ENV=/path/to/aerowall-python-3.10
    export AEROWALL_ISAACSIM_PATH=/path/to/isaac-sim-2023.1.0-hotfix.1

随后运行：

    bash scripts/create_env.sh
    bash scripts/install_training_dependencies.sh
    bash scripts/python.sh --plain scripts/preflight.py --output runs/preflight.json

create_env.sh 和 install_training_dependencies.sh 只写入 runs/environment/ 下的本地环境快照，不修改 docs 中的固定环境记录。

## 正式 C350 批量评测

configs/recenter_recovery.json 是正式配置，SHA-256 为 b08e362b4394cde8e5baf2b7c6ec4bb384580bd65e955f1f4764648147457b52。它设置 1,200 策略步上限和 seed 3001 正式协议。

    bash scripts/python.sh scripts/evaluate_aerowall_wall_rl.py --output runs/c350-formal-reproduction.json --checkpoint checkpoints/c-u350.pt --launch-checkpoint checkpoints/launch-wall.pt --num-envs 128 --seed 3001 --stage RALLY --experiment-config configs/recenter_recovery.json --protocol natural

原正式报告为 128/128 场完成 17 轮并到达评测上限。由于 GPU、驱动和仿真构建可能带来差异，复现结果应以本地新报告为准；需同时检查 checkpoint SHA、环境信息、终止原因和 PhysX 接触审计。

## 单场自然落地回放

展示回放使用 configs/c350_presentation_4800.json：

    bash scripts/python.sh scripts/evaluate_aerowall_wall_rl.py --output runs/c350-presentation-reproduction.json --checkpoint checkpoints/c-u350.pt --launch-checkpoint checkpoints/launch-wall.pt --num-envs 1 --seed 5001 --stage RALLY --experiment-config configs/c350_presentation_4800.json --protocol natural --record-rgb

历史回放完成 25 轮，第 1,840 步因球落地结束。命令会在 runs/ 下生成本次报告和 RGB 帧。仓库中的验收视频还包含早期和中期检查点片段；这些展示快照不是当前最佳 C350 策略，也不随基线模型一起提供，因此重新组合完全相同的三阶段视频需额外取得相应历史快照。
