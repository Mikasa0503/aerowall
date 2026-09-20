"""Audit policy/physics clocks and counted impacts in a substep evaluation."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
t=r['timing_audit'];dt=t['physics_dt'];ratio=round(t['policy_dt']/dt)
events=[json.loads(x) for x in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines()]
checks={'controller_equals_policy':t['controller_calls']==t['policy_steps'],
        'physics_clock':t['physics_steps']==t['policy_steps']*ratio,
        'ten_second_limit':abs(r['steps']*t['policy_dt']-10.)<1e-8,
        'event_clocks':all(abs(e['time']-(e['physics_step']+1)*dt)<1e-7 for e in events),
        'no_credit_on_illegal_physics_step':all(not e['credited_top_impact'] or not e['same_step_illegal_priority'] for e in events),
        'episode_counts_match':all(sum(e['credited_top_impact'] for e in events if e['scenario_id']==x['scenario_id'])==x['provisional_top_entries']==x['upstream_stats']['num_true_hits'] for x in r['outcomes'])}
result={'status':'passed' if all(checks.values()) else 'failed','evaluation':str(a.evaluation),'checks':checks,'timing':t,
        'event_count':len(events),'legal_credits':sum(e['credited_top_impact'] for e in events),
        'scope':'Clock and event/statistics consistency only; contact geometry and physical fidelity have separate fixtures.'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['status']=='passed'
