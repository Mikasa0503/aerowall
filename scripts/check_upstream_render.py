"""Verify RGB readback from the original upstream scene, separately from training."""
from pathlib import Path
import time
import numpy as np
import torch


def check_render(env, base, output_dir, record):
    from PIL import Image
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'warming_renderer', 'visual_review': 'pending',
              'policy_performance_claim': False, 'resolution': list(base.cfg.viewer.resolution)}
    record(render_checks=report)
    td = env.reset()
    before = base.ball.get_world_poses()[0].clone()
    started = time.monotonic()
    for _ in range(5):
        base.sim.render()
    after = base.ball.get_world_poses()[0].clone()
    assert torch.equal(before, after), 'Rendering alone advanced physics'
    first = np.asarray(base.render(mode='rgb_array')).copy()
    assert first.ndim == 3 and first.shape[2] == 3 and first.size > 0
    assert first.dtype == np.uint8 and float(first.std()) > 1.0, 'Uniform/invalid RGB frame'
    Image.fromarray(first).save(output_dir / 'frame-000.png')
    samples = []
    base.enable_render(True)
    try:
        for step in range(20):
            action = torch.zeros(base.num_envs, 1, 4, device=base.device)
            action[..., 3] = 0.32
            td.set(('agents', 'action'), action)
            nxt = env.step(td)['next']
            samples.append({'step': step + 1, 'ball_pos': base.ball.get_world_poses()[0].cpu().tolist()})
            if nxt['done'].any():
                nxt.set('_reset', nxt['done'])
                td = env.reset(nxt)
            else:
                td = nxt
        last = np.asarray(base.render(mode='rgb_array')).copy()
        assert last.shape == first.shape
        Image.fromarray(last).save(output_dir / 'frame-020.png')
        mean_change = float(np.abs(last.astype(np.int16) - first.astype(np.int16)).mean())
        assert mean_change > 0.01, 'RGB readback did not reflect moving scene'
        report.update(status='passed_readback', pixel_readback_passed=True,
                      shape=list(first.shape), first_frame_std=float(first.std()),
                      mean_pixel_change=mean_change, elapsed_seconds=time.monotonic() - started,
                      rendering_only_preserved_physics=True, frame_directory=str(output_dir),
                      diagnostic_steps=20, samples=samples)
        return report
    finally:
        base.enable_render(False)
