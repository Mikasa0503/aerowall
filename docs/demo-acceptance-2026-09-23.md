# Demo 验收记录 — 2026-09-23

## 验收视频

验收视频位于 [media/AeroWall-wall-rally-demo.mp4](../media/AeroWall-wall-rally-demo.mp4)，SHA-256 为 d9149f13bebe7858fcf880d20b778c6a042674ae432528faa4b2c7250dcc3ab7。

视频规格为 1280×720、25 fps、1,107 帧、44.28 秒。内容由早期、中期、最终三个策略快照按时间顺序组成，分别展示 1、2、25 轮。画面直接取自仿真 RGB，不包含标题卡、字幕、片头片尾、变速或重复片段。

## 最终回放

- 演示种子：5001；选择组：C。
- 结果：25 个完成回合，26 次拍面接触和 26 次墙面接触。
- 终止：策略第 1840 步球落地；未达到 4,800 步保护上限。
- 接触审计：通过。
- 该结果是单个展示种子的结果，不代表跨种子可靠性。

## 可复现范围

C350 的单场自然落地回放可用 configs/c350_presentation_4800.json、checkpoints/launch-wall.pt 和 checkpoints/c-u350.pt 复现；完整命令见 REPRODUCIBILITY.md。展示视频中的早期和中期快照是历史视频素材，不是当前最佳模型，不包含在 C350 基线模型清单里。重新构建完全相同的三阶段视频需额外取得相应历史检查点与报告。

最终原始 rollout 报告名为 c-natural-ball-drop-final-5001-h4800.json，SHA-256 为 c9dd6ff64759662cded9a79bf65751058f3d435666878f6ddac19e5a62312374。该大型原始报告及逐帧 RGB 保留在原始运行目录，不纳入轻量复现面；关键结果和报告哈希记录于 docs/results/c350_baseline.json。
