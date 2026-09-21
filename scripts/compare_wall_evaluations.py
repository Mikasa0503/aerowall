"""Compare completed development evaluations with identical serving scenarios."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rows=[];shared=None
for path in a.evaluation:
    r=json.loads(path.read_text());assert r['status']=='passed',path
    identity={k:r[k] for k in ['scenario_sha256','physics_dt','policy_dt','episode_seconds','wall_fixture']}
    identity['task']={k:r['wall_task'][k] for k in ['targets_yz','target_radius','bounds_low','bounds_high']}
    if shared is None:shared=identity
    assert identity==shared,'Different physical scenarios or scoring geometry'
    events=[json.loads(line) for line in path.with_suffix('.events.jsonl').read_text().splitlines()]
    caps=[e for e in events if e.get('wall_cap_credit')]
    velocity=np.array([e['ball_velocity_after'][:3] for e in caps])
    outcomes=r['outcomes'];n=len(outcomes)
    assert n==100 and all(o['scenario_id']==i for i,o in enumerate(outcomes))
    reasons={}
    for o in outcomes:reasons[o['reason']]=reasons.get(o['reason'],0)+1
    rows.append({'evaluation':str(path),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],
                 'controller_mode':r.get('controller_mode','learned_policy'),
                 'trained_frames':r['trained_frames'],'scenarios':n,
                 'one_rally_scenarios':sum(o['rallies']>=1 for o in outcomes),
                 'ten_rally_streak_scenarios':sum(o['max_streak']>=10 for o in outcomes),
                 'wall_contact_scenarios':sum(o['wall_hits']>0 for o in outcomes),
                 'legal_cap_contacts':len(caps),'reasons':reasons,
                 'outgoing_velocity_quantiles_10_50_90_xyz':np.quantile(velocity,[.1,.5,.9],axis=0).tolist() if len(caps) else None,
                 'forward_over_1m_s':int((velocity[:,0]>1).sum()) if len(caps) else 0,
                 'source_hashes':r['source_hashes']})
result={'scope':'100 identical first-serving development episodes; descriptive comparison, not three-seed or formal evidence',
        'scenario_identity':shared,'results':rows}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
    print(json.dumps({k:r[k] for k in ['evaluation','one_rally_scenarios','wall_contact_scenarios','legal_cap_contacts','forward_over_1m_s','reasons']}))
