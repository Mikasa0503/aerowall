"""Compare matched initial states and physical delivery outcomes across dt/CCD."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
reports=[json.loads(x.read_text()) for x in a.runs];ref=reports[0]['initial_state'];rows=[]
positions=[]
wall_points=[]
for path,r in zip(a.runs,reports):
 assert r['status']=='passed' and r['matched_initial_state']
 diff={k:float(np.max(np.abs(np.asarray(ref[k])-np.asarray(v)))) for k,v in r['initial_state'].items()}
 assert all(v==0 for v in diff.values()), (path,diff)
 trajectory=json.loads(path.with_suffix('.trajectory.json').read_text())
 sample=[x for x in trajectory if abs(x['time']-.1)<1e-8]
 assert len(sample)==1
 positions.append(np.asarray(sample[0]['ball_position'])[:,0,:])
 events=json.loads(path.with_suffix('.events.json').read_text())
 points={}
 for e in events:
  if e['kind']=='wall' and e['credited'] and e['points'] and e['env_id'] not in points:
   weights=np.array([x['impulse'] for x in e['points']]);points[e['env_id']]=(np.asarray([x['point'] for x in e['points']])*weights[:,None]).sum(0)/weights.sum()
 wall_points.append(points)
 outcomes=r['outcomes'];caps=[x['first_cap'] for x in outcomes if x['first_cap'] is not None]
 rows.append({'run':str(path),'dt':r['dt'],'ccd':r.get('ccd_requested',False),'initial_state_max_abs_differences':diff,
              'first_cap_count':len(caps),'cap_then_wall':r['cap_then_wall_count'],
              'wall_before_failure':sum(x['cap_then_wall_observed'] and x['rally_state']['wall_hits']>0 for x in outcomes),
              'full_rallies':sum(x['rally_state']['rallies'] for x in outcomes),
              'cap_outgoing_velocities':[x['first_cap']['ball_velocity_after'] if x['first_cap'] else None for x in outcomes],
              'failure_reasons':[x['rally_state']['reason'] for x in outcomes]})
comparisons=[]
for i in range(1,len(rows)):
 x,y=rows[i-1],rows[i]
 wall_ids=set(wall_points[i-1]) & set(wall_points[i])
 wall_errors=[float(np.linalg.norm(wall_points[i][j]-wall_points[i-1][j])) for j in wall_ids]
 eligible=[j for j,(u,v) in enumerate(zip(x['cap_outgoing_velocities'],y['cap_outgoing_velocities'])) if u is not None and v is not None]
 differences=[float(np.linalg.norm(np.array(x['cap_outgoing_velocities'][j])-y['cap_outgoing_velocities'][j])) for j in eligible]
 comparisons.append({'from':x['run'],'to':y['run'],'paired_wall_cases':len(wall_ids),'max_wall_contact_point_difference_m':max(wall_errors) if wall_errors else None,'paired_cap_cases':len(eligible),'max_outgoing_vector_difference_mps':max(differences) if differences else None,
                     'max_ball_position_difference_at_0_1s_m':float(np.linalg.norm(positions[i]-positions[i-1],axis=-1).max()),
                     'note':'Post-impact discrete samples differ in sampling time; this is not an instantaneous rebound or complete convergence proof.'})
result={'status':'passed','scope':'Matched fixture initial states and observed outcomes; not trained return success','runs':rows,'adjacent_comparisons':comparisons}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'}));print([{k:r[k] for k in ['run','dt','ccd','first_cap_count','cap_then_wall','wall_before_failure','full_rallies']} for r in rows])
