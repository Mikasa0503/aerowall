"""Compare frozen single-drone arbitration on exactly shared initial states."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('reports',nargs='+',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];baseline=None;rows=[]
for path in a.reports:
 r=json.loads(path.read_text());assert r['status']=='passed'
 exit_report=json.loads(path.with_suffix('.exit.json').read_text());assert exit_report['passed']
 key=(r['initial_ball'],r['initial_drone'],[(c['role'],c['sha256']) for c in r['checkpoints']])
 if baseline is None:baseline=key
 assert key==baseline,'Initial state or checkpoint mismatch'
 audit=json.loads((root/'docs'/f'{path.stem}-audit.json').read_text())
 cls=json.loads((root/'docs'/f'{path.stem}-classification.json').read_text())
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 assert audit['report_sha256']==cls['report_sha256']==digest
 rows.append({'run':path.stem,'priority':r.get('priority','firstpass'),'episodes':audit['episodes'],
 'wall_contacts':audit['episodes_wall_contact'],'body_wall_body':audit['episodes_wall_return'],
 'top_face_return_candidates':cls['episodes_with_top_face_return_candidate'],
 'maximum_body_wall_body_returns':audit['maximum_returns'],'executed_skill_steps':cls['executed_skill_steps'],
 'report_sha256':digest})
result={'status':'passed','same_initial_states_and_weights':True,'rows':rows,
 'scope':'Single-drone arbitration sensitivity; native-box top-face candidates, not formal bat legality'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
