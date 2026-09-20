"""Frozen-scenario aligned juggling failure strata; no success-only filtering."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text())
initial=json.loads(a.evaluation.with_suffix('.initial-scenarios.json').read_text())['initial_state']
d=np.asarray(initial['drone_position']);b=np.asarray(initial['ball_position'])
offset=np.linalg.norm((b-d)[:,:2],axis=-1)
v=np.linalg.norm(np.asarray(initial['ball_velocity'])[:,:2],axis=-1)
outcomes=r['outcomes']
def summarize(ids):
 rows=[outcomes[int(i)] for i in ids]
 return {'count':len(rows),'five_legal_caps':sum(x['provisional_top_entries']>=5 for x in rows),
         'reasons':dict(Counter(x['reason'] for x in rows)),
         'zero_cap_count':sum(x['provisional_top_entries']==0 for x in rows),
         'scenario_ids':[int(i) for i in ids]}
# Rank strata are descriptive, not fixed thresholds or separately drawn tests.
rank=np.argsort(offset,kind='stable');groups=[]
for ids in np.array_split(rank,4):
 groups.append({**summarize(ids),'initial_horizontal_offset_range_m':[float(offset[ids].min()),float(offset[ids].max())]})
result={'evaluation':str(a.evaluation),'checkpoint_sha256':r['checkpoint_sha256'],
        'all':summarize(range(len(outcomes))),'initial_offset_rank_quartiles':groups,
        'initial_ball_horizontal_speed_range_mps':[float(v.min()),float(v.max())],
        'note':'All fixed scenarios retained; rank quartiles describe initial difficulty, not causal effects or controlled disturbance recovery. No new success definition.'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
