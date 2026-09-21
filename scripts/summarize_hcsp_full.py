"""Describe a completed full HCSP baseline without treating proxy hits as contacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());assert r['status']=='passed'
t=Path(r['trajectory']);assert hashlib.sha256(t.read_bytes()).hexdigest()==r['trajectory_sha256']
z=np.load(t);active=z['active'];hits=z['hits']*active[...,None]
ball=z['ball'][:,:,0,:];cross=(ball[1:,:,0]*ball[:-1,:,0]<0)&active[1:]&active[:-1]
counts=hits.sum((0,2));maxhit=int(counts.max());reasons={}
for o in r['outcomes']:
 for k,v in o['stats'].items():
  if k.startswith('done_') and np.asarray(v).any():reasons[k]=reasons.get(k,0)+1
out={'scope':'Original six-drone HCSP baseline; upstream hit proxies, not independent GPU contacts',
     'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),'episodes':len(r['outcomes']),
     'proxy_hit_counts':counts.tolist(),'proxy_hits_by_drone':hits.sum((0,1)).tolist(),
     'episodes_with_proxy_hit':int((counts>0).sum()),'max_proxy_hits_per_episode':maxhit,
     'episodes_with_net_plane_crossing':int((cross.sum(0)>0).sum()),
     'net_plane_crossings_per_episode':cross.sum(0).tolist(),'terminal_reasons':reasons,
     'high_level_unique_actions':[np.unique(z['high'][:,:,i,:][active],axis=0).tolist() for i in range(2)],
     'limitations':['Net-plane crossing does not imply legal over-net flight.','No wall is present.','No independent contact force confirmation yet.']}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
