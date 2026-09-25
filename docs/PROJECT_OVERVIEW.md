# AeroWall 项目概览

## 问题与目标

AeroWall 研究一架多旋翼无人机如何把球合法击向墙面，再移动到回球轨迹附近继续击球。任务包含拍面接触、墙面接触、无人机与墙／地的安全边界，以及长回合计分。

项目以 VolleyBots 的通用训练与仿真能力为基础。具体运行环境、依赖职责和固定源码版本列于 upstream-lock.json；可复现实验应使用该文件中的锁定版本，而不是随时间变化的主分支。

## 系统结构

    Isaac Sim / GPU physics
             |
       pinned upstream
    VolleyBots / JuggleRL / HCSP
             |
       AeroWall task layer
    wall scene, contact ledger, rally state,
    observation and reward variants, curriculum
             |
       frozen skill route
    Launch -> C350 rally controller

物理仿真、上游 PPO/MAPPO 和原始检查点保留各自上游名称。AeroWall 自有代码负责单无人机对墙任务集成、接触生命周期、技能链配置、壁球观测与奖励变体、扰动课程和实验审计。

## 当前选择

当前最优且有正式评测证据的方案是冻结 Launch 加 C350 回接策略。正式评测的 128 场都在 1,200 个策略步上限内达到 17 轮；单次展示回放达到 25 轮并自然结束于球落地。两个证据适用范围不同，详见 RESULTS.md。

## 研究边界

已完成的工程实现不自动代表性能提升。多样首球上的首击改善尚未带来达标的连续回合；拆分 Launch／Hit／Recover 尚未证明胜过 C350；新观测、因果奖励和有界动作主要形成诊断或工程成果；高层策略与共同适应未完成。完整状态见 RESEARCH_STATUS.md。
