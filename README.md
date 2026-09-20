# AeroWall

基于 JuggleRL 的状态输入无人机连续壁球与未知墙面弹性适应项目。

当前阶段：部署准备。完整目标、物理门槛与实验协议见 `docs/EXECUTION_PLAN.md`，实测进度见 `STATUS.md`。

## 138 部署

项目：`/home/public/Workspace/shy/Code/aerowall`

独立环境：`/home/public/Workspace/boweiy/mambaforge/envs/aerowall`

```bash
cd /home/public/Workspace/shy/Code/aerowall
bash scripts/create_env.sh
bash scripts/python.sh --plain scripts/preflight.py --output runs/preflight.json
```

预检退出码 2 表示部署条件不全，不能据此进入训练。预检只验证环境和源码，不替代物理或 PPO 验收。

指定版本运行时需放入 `third_party/isaac-sim-2023.1.0-hotfix.1`；其中必须包含 `setup_conda_env.sh`。只有运行时准备完成才使用 `bash scripts/python.sh <script.py>`。其他路径可通过 `AEROWALL_ISAACSIM_PATH` 指定，必须记录来源与版本。该变量不是版本升级授权。

本项目源码和依赖版本锁见 `docs/upstream-lock.json`、`docs/conda-linux-64-explicit.txt`。第三方代码的许可及版权以各仓库保留的 LICENSE 和文件头为准。项目新增内容当前仅为隔离启动、预检和执行记录；尚无完成的 WallRally 环境、训练结果或 Demo。


## Wall contact integration (development fixtures)

The project scene adds cloned walls and explicitly enables original body colliders. The as-authored central cap is occluded by the body box; the default fixture preserves and reports that failure. An optional stage-only overlay aligns the cap to the visual bat top without changing the pinned asset. It changes the auto COM and requires new dynamics/policy validation.

```bash
./scripts/run_probe.sh scripts/probe_wall_scene.py runs/wall-scene-recheck.json --align-cap-to-visual-top
./scripts/python.sh --plain -m unittest discover -s tests -p test_rally_events.py -v
```

Use a new output name for each run. Evidence: `docs/wall-scene-integration-evidence.json`, `docs/air-contact-layout.json`, and `docs/plans/2026-09-21-wall-scene-integration.md`. These are controlled collision fixtures, not a completed WallRally policy/environment or formal comparison. The adapter combines current CPU lifecycle/point reports with GPU readback where needed, verifies signed point/aggregate consistency, and never credits unmatched stale GPU buffers.
