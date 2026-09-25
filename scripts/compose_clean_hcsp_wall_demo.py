"""Join audited raw HCSP rollouts without titles, subtitles, or end cards."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', required=True, type=Path,
                        help='JSON with ordered early, middle, and final raw-rollout segments')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    segment_spec = json.loads(args.segments.read_text())['segments']
    assert len(segment_spec) == 3 and [item['role'] for item in segment_spec] == ['early', 'middle', 'final']
    clips = []
    manifest = []
    encode = ('-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
              '-pix_fmt', 'yuv420p', '-movflags', '+faststart')
    for index, segment in enumerate(segment_spec):
        report_path = Path(segment['report'])
        report = json.loads(report_path.read_text())
        assert report['status'] == 'passed' and report['num_envs'] == 1
        assert report.get('protocol', 'natural') == 'natural'
        assert report['outcomes'][0]['rallies'] >= int(segment.get('minimum_rallies', 0))
        audit = report['contact_audit']
        assert audit['legal_events'] == audit['corroborated_events']
        assert audit['wall_events'] == audit['corroborated_wall_events']
        assert not audit['callback_errors']
        rgb = report['rgb']
        frame_dir = Path(rgb['directory'])
        frames = sorted(frame_dir.glob('*.png'))
        assert rgb['fps'] == 25 and rgb['initial_frame_time'] == 0.0
        assert len(frames) == rgb['frames']
        start_frame = int(segment.get('start_frame', 0))
        end_frame = int(segment.get('end_frame', len(frames)))
        assert 0 <= start_frame < end_frame <= len(frames)
        if segment['role'] == 'final':
            assert start_frame == 0 and end_frame == len(frames), 'Final run must be uninterrupted from fixed drop'
        clip = args.output / f'{index}-raw.mp4'
        run('ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-framerate', '25',
            '-start_number', str(start_frame), '-i', str(frame_dir / '%05d.png'),
            '-frames:v', str(end_frame - start_frame), *encode, str(clip))
        clips.append(clip)
        manifest.append({
            'role': segment['role'],
            'report': str(report_path.resolve()),
            'checkpoint_sha256': report['checkpoint_sha256'],
            'rallies': report['outcomes'][0]['rallies'],
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames': end_frame - start_frame,
            'duration_seconds': (end_frame - start_frame) / 25,
            'clip_sha256': sha256(clip),
            'contact_events_sha256': sha256(report['first_episode_events']),
        })

    listing = args.output / 'concat.txt'
    listing.write_text(''.join(f"file '{clip.resolve()}'\n" for clip in clips))
    demo = args.output / 'HCSP-wall-learning-demo-clean.mp4'
    run('ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
        '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(demo))
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'stream=width,height,r_frame_rate,nb_frames:format=duration', '-of', 'json', str(demo)
    ]))
    expected_frames = sum(item['frames'] for item in manifest)
    assert probe['streams'][0]['width'] == 1280 and probe['streams'][0]['height'] == 720
    assert probe['streams'][0]['r_frame_rate'] == '25/1'
    assert int(probe['streams'][0]['nb_frames']) == expected_frames
    assert abs(float(probe['format']['duration']) - expected_frames / 25) < 0.05
    (args.output / 'manifest.json').write_text(json.dumps({
        'clips': manifest,
        'demo_sha256': sha256(demo),
        'probe': probe,
        'presentation': 'raw simulator RGB only; no text, intro, outro, speed change, or repeated footage',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
