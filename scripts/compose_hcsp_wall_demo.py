"""Compose a 30-second documentary from audited, uninterrupted RGB rollouts."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def run(*args):
    subprocess.run(args, check=True)


def stamp(seconds):
    cs = round(seconds * 100)
    return f'{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ('early', 'middle', 'final', 'evaluation', 'recheck', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(exist_ok=False, parents=True)
    reports = [json.loads(path.read_text()) for path in (a.early, a.middle, a.final)]
    evaluation, recheck = [json.loads(path.read_text()) for path in (a.evaluation, a.recheck)]
    for r in (evaluation, recheck):
        assert r['gate']['five_passed'], 'Main goal has not passed independent evaluation'
    assert evaluation['num_envs'] >= 128 and recheck['num_envs'] >= 20
    assert evaluation['checkpoint_sha256'] == recheck['checkpoint_sha256'] == reports[2]['checkpoint_sha256']
    clips, durations, manifest = [], [], []
    ass_header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Noto Sans CJK SC,32,&H00FFFFFF,&H00FFFFFF,&H00202020,&H80202020,0,0,0,0,100,100,0,0,3,2,0,7,28,28,24,1
Style: Footer,Noto Sans CJK SC,23,&H00FFFFFF,&H00FFFFFF,&H00202020,&H80202020,0,0,0,0,100,100,0,0,3,2,0,1,28,28,22,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    labels = ['早期策略｜一次回接后失败', '中期策略｜连续两轮', '最终策略｜连续回合']
    for index, r in enumerate(reports):
        assert r['status'] == 'passed' and r['num_envs'] == 1
        audit = r['contact_audit']
        assert audit['legal_events'] == audit['corroborated_events']
        assert audit['wall_events'] == audit['corroborated_wall_events'] and not audit['callback_errors']
        rgb = r['rgb']
        assert rgb['fps'] == 25 and rgb['initial_frame_time'] == 0
        frame_dir = Path(rgb['directory'])
        assert len(list(frame_dir.glob('*.png'))) == rgb['frames']
        duration = rgb['frames'] / 25
        durations.append(duration)
        raw = a.output / f'{index}-raw.mp4'
        common = ('-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart')
        run('ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-framerate', '25', '-i', str(frame_dir / '%05d.png'), *common, str(raw))
        events = [json.loads(line) for line in Path(r['first_episode_events']).read_text().splitlines()]
        phase, count, changes = 0, 0, [(0., 0)]
        for event in events:
            if event['env'] != 0:
                continue
            if event['legal_cap']:
                if phase == 2:
                    count += 1
                    time = (event['policy_step'] * 8 + event['substep'] + 1) * .0025
                    changes.append((time, count))
                phase = 1
            elif event['wall']:
                assert phase == 1
                phase = 2
        assert count == r['outcomes'][0]['rallies']
        if index == 2:
            assert count >= 5
        lines = [ass_header]
        for j, (start, n) in enumerate(changes):
            end = changes[j + 1][0] if j + 1 < len(changes) else duration
            lines.append(f'Dialogue: 0,{stamp(start)},{stamp(end)},Title,,0,0,0,,{labels[index]}  ·  已完成 {n} 轮\n')
        lines.append(f'Dialogue: 0,0:00:00.00,{stamp(duration)},Footer,,0,0,0,,固定投球 · 仿真原速 1× · 单局完整重放\n')
        subtitle = a.output / f'{index}.ass'
        subtitle.write_text(''.join(lines))
        labelled = a.output / f'{index}-labelled.mp4'
        run('ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(raw), '-vf', f'ass={subtitle}', *common, str(labelled))
        clips.append(labelled)
        manifest.append({'checkpoint_sha256': r['checkpoint_sha256'], 'rallies': count,
                         'frames': rgb['frames'], 'duration': duration, 'raw': str(raw),
                         'raw_sha256': sha(raw), 'event_sha256': sha(r['first_episode_events']),
                         'event_counter_times': changes})
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    def card(name, title, lines, seconds):
        canvas = Image.new('RGB', (1280, 720), '#101c2c')
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((80, 110, 88, 600), fill='#48d5be')
        draw.text((120, 125), 'AEROWALL  /  HCSP', font=ImageFont.truetype(font, 26), fill='#48d5be')
        draw.text((120, 220), title, font=ImageFont.truetype(font, 48), fill='white')
        for i, line in enumerate(lines):
            draw.text((120, 330 + 64 * i), line, font=ImageFont.truetype(font, 29), fill='#d6e0ec')
        still, video = a.output / f'{name}.png', a.output / f'{name}.mp4'
        canvas.save(still)
        run('ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-loop', '1', '-framerate', '25', '-i', str(still), '-t', f'{seconds:.2f}', *common, str(video))
        return video
    intro = card('intro', '单机对墙连续击球', ['同一 HCSP 无人机与小球模型', '固定投球 · PPO 课程学习 · 真实物理接触'], 3.)
    transition1 = card('middle', '学会接回，也要再次发墙', ['较早检查点 → 中期检查点', '每个运行片段内部保持连续、原速'], 2.)
    transition2 = card('final', '最终策略：完整运行', ['从固定投球开始', '保底 3 轮 / 主目标 5 轮'], 2.)
    remaining = round(30. - 7. - sum(durations), 2)
    assert remaining >= 2., 'Clips exceed the 30-second composition budget'
    final_count = reports[2]['outcomes'][0]['rallies']
    summary = card('summary', f'1 → 2 → {final_count} 轮',
                   ['128/128 局达成至少 5 轮；独立复测 20/20',
                    '真实接触记录逐事件核验；固定起点条件',
                    '第六轮后仍会失败，未声称长期稳定控制'], remaining)
    # The text above uses exact counts, so reject a merely 80%-passing result.
    assert evaluation['summary']['five_rally_rate'] == recheck['summary']['five_rally_rate'] == 1.
    sequence = [intro, clips[0], transition1, clips[1], transition2, clips[2], summary]
    listing = a.output / 'concat.txt'
    listing.write_text(''.join(f"file '{path}'\n" for path in sequence))
    final_video = a.output / 'HCSP-wall-learning-demo.mp4'
    run('ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(final_video))
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-show_entries', 'format=duration', '-of', 'json', str(final_video)]))
    assert abs(float(probe['format']['duration']) - 30.) < .1
    (a.output / 'manifest.json').write_text(json.dumps({'clips': manifest, 'probe': probe,
        'video_sha256': sha(final_video), 'evaluation_sha256': sha(a.evaluation),
        'recheck_sha256': sha(a.recheck), 'script_sha256': sha(__file__),
        'sequence': [str(path) for path in sequence], 'final_rollout_uninterrupted': True}, indent=2))


if __name__ == '__main__':
    main()
