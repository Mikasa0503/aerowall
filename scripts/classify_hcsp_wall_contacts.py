"""Classify native IrisTest collision-box top-face candidates, not AeroWall cap legality."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());assert r['status']=='passed'
z=np.load(r['trajectory']);assert hashlib.sha256(Path(r['trajectory']).read_bytes()).hexdigest()==r['trajectory_sha256']
origins=np.array(r['env_origins']);rows=[];counts=[]
for env in range(len(r['outcomes'])):
 phase=0;completed=0;body_after_wall=[]
 points=[p for p in r['contact_points'] if p['env']==env and p['episode']==1]
 for step in np.where(z['contact_entry'][:,env].any(-1)&z['active'][:,env])[0]:
  body,wall=z['contact_entry'][step,env]
  samples=[p for p in points if p['global_step']==int(step)]
  assert all(p['step']==step for p in samples)
  kind0=[p for p in samples if p['kind']==0];kind1=[p for p in samples if p['kind']==1]
  if body:assert kind0
  if wall:assert kind1
  # Native physical box: half x/y=.235, top z=.055. End-step body motion
  # prevents exact face-distance tests; require upward-region and normal.
  top=bool(kind0) and all(p['body_local_point'][2]>0 and abs(p['body_local_normal'][2])>=.9 and max(abs(v) for v in p['body_local_point'][:2])<=.255 for p in kind0)
  wallvalid=bool(kind1) and all(abs((np.array(p['position'])-origins[env])[0]-.1)<.06 and abs(p['normal'][0])>=.9 for p in kind1)
  if wall:assert wallvalid
  rows.append({'env':env,'step':int(step),'body':bool(body),'wall':bool(wall),'top_face_candidate':top,
    'body_points':[p['body_local_point'] for p in kind0],'body_normals':[p['body_local_normal'] for p in kind0]})
  if body and wall:phase=0;continue
  if body:
   if phase==2 and top:completed+=1
   phase=1 if top else 0
  elif wall:phase=2 if phase==1 else 0
 counts.append(completed)
skill_names=['FirstPass_goto','FirstPass_pass','FirstPass_hover','FirstPass_serve','FirstPass_serve_hover','SecPass_goto','SecPass_hit','SecPass_hover','Att_goto','Att_hit','Att_hover']
result={'status':'passed','scope':'Native IrisTest top-face contact candidates; not certified AeroWall bat legality',
 'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),
 'criterion':{'normal_abs_z_min':.9,'point_z_min':0,'box_xy_with_motion_tolerance':.255,'wall_point_x':.1,'wall_point_tolerance':.06},
 'top_face_return_candidates':counts,'episodes_with_top_face_return_candidate':sum(v>0 for v in counts),
 'executed_skill_steps':{name:int(((z['executed_skill']==i)&z['active']).sum()) for i,name in enumerate(skill_names)},
 'events':rows,'limitations':['Contact point transformed with end-step body pose; not exact impact pose.',
 'Native box top is not the AeroWall legal bat cap.','Rotor contacts are not included in this base-link audit.']}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
