"""Write an evidence-first A/B/C report from untouched formal evaluations."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from run_hcsp_wall_recenter_experiment import RUNS


def format_rate(count, total):
    return f'{count}/{total} ({count / total:.1%})'


def challenged_return_positions(report):
    """Ballistic lateral return at the normal 2.18 m contact height after wall 3."""
    walls = {}
    for line in Path(report['first_episode_events']).open():
        event = json.loads(line)
        if event.get('wall'):
            walls.setdefault(event['env'], []).append(event)
    assert len(walls) == report['num_envs'] == 40
    result = []
    for events in walls.values():
        assert len(events) >= 3
        wall = events[2]
        y, z = wall['ball_position'][1], wall['ball_position'][2]
        vy, vz = wall['ball_velocity_after'][1], wall['ball_velocity_after'][2]
        discriminant = vz * vz + 2 * 9.81 * (z - 2.18)
        assert discriminant >= 0
        t = (vz + math.sqrt(discriminant)) / 9.81
        result.append(y + vy * t)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    formal = json.loads((RUNS / 'formal-summary.json').read_text())
    assert formal['status'] == 'passed'
    training = json.loads((RUNS / 'execution-state.json').read_text())
    result = {}
    for group in ('A', 'B', 'C'):
        natural = json.loads((RUNS / f'{group.lower()}-formal-natural-128.json').read_text())
        recovery = json.loads((RUNS / f'{group.lower()}-formal-recovery-100.json').read_text())
        assert natural['status'] == recovery['status'] == 'passed'
        assert all(row['contact_audit_passed'] for row in natural['outcomes'] + recovery['outcomes'])
        side = recovery['summary']['recovery_by_direction']
        cells = {}
        for direction in ('left', 'right'):
            for difficulty in ('light', 'medium'):
                rows = [row for row in recovery['outcomes']
                        if row['direction'] == direction and row['difficulty'] == difficulty]
                cells[f'{direction}_{difficulty}'] = {'success': sum(row['recovery_success'] for row in rows),
                                                       'total': len(rows)}
        result[group] = {
            'checkpoint_sha256': natural['checkpoint_sha256'],
            'updates': formal['selection']['groups'][group].get('update', 0),
            'safe10': natural['summary']['safe10_count'],
            'safe15': natural['summary']['safe15_count'],
            'mean_rallies': round(natural['summary']['mean_rallies'], 2),
            'mean_centered_prefix': round(natural['summary']['mean_centered_prefix_rallies'], 2),
            'failure_reasons': dict(Counter(row['failure_reason'] for row in natural['outcomes'])),
            'wall_y_mean_by_ordinal': [
                round(statistics.mean(row['wall_y_sequence'][ordinal] for row in natural['outcomes']
                                      if len(row['wall_y_sequence']) > ordinal), 3)
                if any(len(row['wall_y_sequence']) > ordinal for row in natural['outcomes']) else None
                for ordinal in range(17)
            ],
            'recovery': recovery['summary']['recovery_success_count'],
            'recovery_left': side['left']['success'],
            'recovery_right': side['right']['success'],
            'recovery_off_center_subset': recovery['summary']['off_center_recovery_subset'],
            'recovery_cells': cells,
            'natural_gate': natural['gate']['passed'],
            'recovery_gate': recovery['gate']['passed'],
        }
    selected = formal['selection']['selected_demo_group']
    rows = ['# AeroWall 单机对墙回中与恢复：正式结果', '',
            f'开发集预先选定演示模型：{selected}。训练停止原因：{training["stop_reason"]}。', '',
            '| 组别 | 更新 | safe10 /128 | safe15 /128 | 平均轮数 | 平均连续居中轮数 | 恢复 /100 | 左 /50 | 右 /50 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group, data in result.items():
        rows.append(f'| {group} | {data["updates"]} | {data["safe10"]} | {data["safe15"]} | '
                    f'{data["mean_rallies"]} | {data["mean_centered_prefix"]} | {data["recovery"]} | '
                    f'{data["recovery_left"]} | {data["recovery_right"]} |')
    rows += ['', f'演示模型自然 safe10：{format_rate(result[selected]["safe10"], 128)}。',
             f'演示模型恢复：{format_rate(result[selected]["recovery"], 100)}；'
             f'左 {format_rate(result[selected]["recovery_left"], 50)}；'
             f'右 {format_rate(result[selected]["recovery_right"], 50)}。',
             f'20 个独立进程复测 safe10：{format_rate(formal["repeat_safe10_count"], 20)}。', '',
             '注意：正式轻/中扰动中，B/C 在首次回中前实际出带的案例均为 0/100；'
             '100/100 只表示通过已冻结的轻扰动协议，不能称为真正的出带救援。', '',
             '墙面横向位置（前十七次，按有该次接触的回合求均值，单位 m）：', '']
    for group, data in result.items():
        rows.append(f'- {group}: {data["wall_y_mean_by_ordinal"]}')
    rows += ['', '恢复测试每格 25 个案例：', '']
    for group, data in result.items():
        rows.append(f'- {group}: {data["recovery_cells"]}；'
                    f'首次回中前曾撞出中心带的子集 {data["recovery_off_center_subset"]}')
    challenge_path = RUNS / 'challenge-summary.json'
    if challenge_path.exists():
        challenge = json.loads(challenge_path.read_text())
        assert challenge['status'] == 'passed'
        rows += ['', '冻结模型后的强冲量挑战（Δvy=3.0/4.0 m/s；不参与选模或原定验收）：', '',
                 '| 组别 | 恢复 /40 | 首次回中前实际出带 | 出带后救回 | 接触审计 |',
                 '|---|---:|---:|---:|---|']
        for group in ('A', 'B', 'C'):
            item = challenge['results'][group]
            subset = item['actually_off_center_before_recovery']
            assert item['contact_audit_passed']
            rows.append(f'| {group} | {item["all_recovery_success"]}/40 | '
                        f'{subset["count"]} | {subset["recovery_success"]}/{subset["count"]} | 通过 |')
        c_challenge = json.loads((RUNS / 'c-challenge-40.json').read_text())
        predicted_y = challenged_return_positions(c_challenge)
        rows += ['', 'C 组 40 例在第 3 次墙面接触后，按无额外碰撞的弹道外推至常规接球高度 '
                 '`z=2.18 m`：预计横向位置范围 '
                 f'{min(predicted_y):+.2f} 至 {max(predicted_y):+.2f} m，'
                 f'落在合法 `|y|≤3 m` 内的仅 {sum(abs(y) <= 3 for y in predicted_y)}/40。'
                 '高位提前拦截仍可能改变结果，因此强挑战的 0/40 不能单独证明策略不能救物理可救的偏球；'
                 '下一阶段应先校准真正出带且可接的中等扰动。']
    zero_path = RUNS / 'diagnostics/a-zero-force-comparison.json'
    if zero_path.exists():
        zero = json.loads(zero_path.read_text())
        rows += ['', f'A 策略的零力 API 对照：相同检查点和 seed 的 {zero["num_envs"]} 对自然回合中，'
                 f'轮数差异 {zero["rally_count_differences"]}，失败原因差异 {zero["failure_reason_differences"]}，'
                 f'配对墙面横向位置最大差 {zero["max_paired_wall_y_difference_m"]:.6f} m。'
                 '该诊断不等于多训练种子因果证明。']
    matched = training['completed_pairs'][-1]
    rows += ['', f'B/C 相同训练量对照：第 {matched} 次更新的开发集指标 '
             f'{training["pair_metrics"][str(matched)]}。',
             f'截至正式评测累计单卡时间：{formal["total_gpu_seconds_through_formal"] / 3600:.2f} 小时。',
             '固定投球的并行环境轨迹高度相似，不能把 128 个结果当作独立来球样本；'
             '20 次重新启动进程只检验该固定条件的进程重复性，因此不计算二项置信区间。'
             '扰动成绩仅覆盖冻结的轻、中幅度与时机。'
             '本实验仅有一个训练种子，不做跨训练种子稳健性或实机能力结论。',
             'A 是旧模型，B/C 的相同更新量对比用于判断扰动课程是否提供增益。', '']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(rows))
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'markdown': str(a.output), 'selected_group': selected,
                      'complete_target_passed': formal['complete_target_passed']}))


if __name__ == '__main__':
    main()
