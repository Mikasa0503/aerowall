"""Reconstruct physical rally counts from saved first-episode contact events.

Independent minimal chain counter; does not import the environment/state machine.
Geometry and impulse validation remain the responsibility of separate fixtures.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();report=json.loads(a.evaluation.read_text())
assert report['status']=='passed'
ratio=round(report['policy_dt']/report['physics_dt'])
groups=defaultdict(lambda:defaultdict(list))
clock_ok=True
for line in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines():
    e=json.loads(line);micro=e['policy_step']*ratio+e['physics_substep']+1
    clock_ok &= abs(e['time']-micro*report['physics_dt'])<1e-7
    groups[e['env_id']][micro].append(e)
checks=[]
illegal={'ball_ground','ball_body','non_cap','drone_wall','drone_ground'}
for row in report['outcomes']:
    phase='wait';rallies=streak=maximum=walls=caps=0
    last=row.get('physics_steps',row['steps'])
    for micro,events in sorted(groups[row['scenario_id']].items()):
        # Terminal state increments its clock before processing the failure.
        # Contacts in later held-action substeps cannot count for the episode.
        if micro>last or (micro==last and row['terminated']):continue
        accepted=[e for e in events if e['credited']]
        if any(e['kind'] in illegal for e in accepted):
            phase='wait';streak=0;continue
        nc=sum(e['kind']=='cap' and e['wall_cap_credit'] for e in accepted)
        nw=sum(e['kind']=='wall' for e in accepted)
        caps+=nc;walls+=nw
        if nc>1 or nw>1 or (nc and nw):phase='wait';streak=0
        elif nc:
            if phase=='return':rallies+=1;streak+=1;maximum=max(maximum,streak)
            else:streak=0
            phase='outbound'
        elif nw:
            if phase=='outbound':phase='return'
            else:phase='wait';streak=0
    expected={k:row[k] for k in ['rallies','max_streak','wall_hits','legal_caps']}
    actual={'rallies':rallies,'max_streak':maximum,'wall_hits':walls,'legal_caps':caps}
    checks.append({'scenario_id':row['scenario_id'],'passed':expected==actual,'expected':expected,'reconstructed':actual})
result={'status':'passed' if clock_ok and all(x['passed'] for x in checks) else 'failed',
        'evaluation':str(a.evaluation),'event_clocks_match':clock_ok,'scenarios':len(checks),
        'rallies':sum(x['reconstructed']['rallies'] for x in checks),
        'mismatches':[x for x in checks if not x['passed']],
        'scope':'Independent event-chain/count consistency, not an independent validation of collision geometry'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
assert result['status']=='passed'
