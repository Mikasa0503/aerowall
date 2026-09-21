"""Measure requested CTBR saturation and end-of-hold body-rate tracking, not reachability."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
z=np.load(a.evaluation.with_suffix('.trajectory.npz'))
raw=z['action'];assert raw.shape[-1]==4
requested=np.tanh(raw);rate=requested[...,:3]*np.pi
q=z['drone_quaternion_wxyz'];assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-4
# Rotate world angular velocity into the poststep body frame using q inverse.
v=z['drone_velocity'][...,3:];u=-q[...,1:];t=2*np.cross(u,v)
body=v+q[...,:1]*t+np.cross(u,t)
error=np.linalg.norm(body-rate,axis=-1)
times=(np.arange(len(raw))+1)*float(z['dt'])
active=z['active_before'].astype(bool)
for outcome in r['outcomes']:
 i=outcome['scenario_id'];terminal=outcome['physics_steps']*r['physics_dt']
 active[:,i]&=times<terminal-1e-8
phase=z['actor_observation_before'][...,40:43].argmax(-1)
rows={}
for name,mask in [('all',active)]+[(n,active&(phase==k)) for k,n in enumerate(['wait','to_wall','to_bat'])]:
 if not mask.any():rows[name]={'samples':0};continue
 # End-of-policy-interval tracking: not an instantaneous servo error.
 rows[name]={'samples':int(mask.sum()),'episodes':int(mask.any(axis=0).sum()),
 'requested_rate_near_limit_fraction':float((np.abs(requested[...,:3][mask])>=.95).any(axis=-1).mean()),
 'requested_thrust_near_upper_limit_fraction':float((requested[...,3][mask]>=.95).mean()),
 'requested_collective_quantiles_m_s2':np.quantile(7.5*(1+requested[...,3][mask]),[.1,.5,.9]).tolist(),
 'body_rate_tracking_norm_quantiles_rad_s':np.quantile(error[mask],[.1,.5,.9]).tolist(),
 'poststep_body_rate_norm_quantiles_rad_s':np.quantile(np.linalg.norm(body[mask],axis=-1),[.1,.5,.9]).tolist()}
result={'status':'passed','evaluation':str(a.evaluation),'report_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
 'trajectory_sha256':hashlib.sha256(a.evaluation.with_suffix('.trajectory.npz').read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'scope':'All active first episodes, excluding terminal held-action samples. Phase from actual preaction input. Rates compared to end-of-hold body-frame measured rate. Saturation is requested CTBR, not rotor saturation; this does not prove task infeasibility or controller causality.',
 'mapping':'Pinned Flightmare transform: target body rate=pi*tanh(raw[:3]), collective=7.5*(1+tanh(raw[3]))',
 'phase_statistics':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))
