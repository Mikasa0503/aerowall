"""Aggregate frozen one-drone runs without mixing wall distances into seed rates."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('reports',nargs='+',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];rows=[];weights=None;initial_seed0=None
for path in a.reports:
 r=json.loads(path.read_text());assert r['status']=='passed' and r['hit_memory']
 assert json.loads(path.with_suffix('.exit.json').read_text())['passed']
 w=[(c['role'],c['sha256']) for c in r['checkpoints']]
 if weights is None:weights=w
 assert w==weights
 docs={suffix:json.loads((root/'docs'/f'{path.stem}-{suffix}.json').read_text()) for suffix in ['audit','classification','state-audit']}
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 assert all(d['status']=='passed' and d['report_sha256']==digest for d in docs.values())
 if r['seed']==0:
  initial=(r['initial_ball'],r['initial_drone'])
  if initial_seed0 is None:initial_seed0=initial
  assert initial==initial_seed0
 rows.append({'run':path.stem,'seed':r['seed'],'wall_center_x':r['wall']['center'][0],
 'episodes':docs['audit']['episodes'],'wall_contact_episodes':docs['audit']['episodes_wall_contact'],
 'body_return_episodes':docs['audit']['episodes_wall_return'],
 'top_face_candidate_episodes':docs['classification']['episodes_with_top_face_return_candidate'],
 'maximum_returns':docs['audit']['maximum_returns'],'report_sha256':digest})
base=[x for x in rows if x['wall_center_x']==0]
result={'status':'passed','identical_weights':True,'seed0_wall_test_initial_state_identical':True,'rows':rows,
 'wall0_aggregate':{k:sum(x[k] for x in base) for k in ['episodes','wall_contact_episodes','body_return_episodes','top_face_candidate_episodes']},
 'scope':'Three-seed single-drone prototype and separate wall-distance sensitivity; top face is not formal bat legality'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
