"""Summarize true completed training episodes, not aliased collector statistics."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--metrics',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--window',type=int,default=25)
a=p.parse_args();assert a.window>0
rows=[];sources=[]
for path in a.metrics:
 raw=path.read_bytes();lines=raw.splitlines(keepends=True)
 if lines and not lines[-1].endswith(b'\n'):lines.pop()
 prefix=b''.join(lines);part=[json.loads(l) for l in lines];rows.extend(part)
 sources.append({'path':str(path),'complete_bytes_read':len(prefix),'prefix_sha256':hashlib.sha256(prefix).hexdigest(),'updates_read':len(part)})
assert rows
for prev,cur in zip(rows,rows[1:]):
 assert cur['total_updates']==prev['total_updates']+1,'Missing or repeated update'
 assert cur['environment_frames']>prev['environment_frames'],'Non-increasing frame count'
windows=[]
for start in range(0,len(rows),a.window):
 block=rows[start:start+a.window];episodes=[e for r in block for e in r['terminal_episodes']]
 assert all(len(r['terminal_episodes'])==r['completed_episodes'] for r in block)
 windows.append({'first_update':block[0]['total_updates'],'last_update':block[-1]['total_updates'],
  'end_environment_frames':block[-1]['environment_frames'],'complete_window':len(block)==a.window,
  'completed_episodes':len(episodes),'wall_episodes':sum(e['wall_hits']>0 for e in episodes),
  'return_episodes':sum(e['rallies']>0 for e in episodes),'rallies':sum(e['rallies'] for e in episodes),
  'joint_rallies':sum(e['joint_rallies'] for e in episodes),'reasons':dict(Counter(e['reason'] for e in episodes)),
  'return_episode_fraction':sum(e['rallies']>0 for e in episodes)/len(episodes) if episodes else None})
result={'scope':'Training terminal episodes grouped by optimizer updates; not fixed-scene evaluation or independent-seed evidence; last window may be partial',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':sources,'windows':windows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(windows))
