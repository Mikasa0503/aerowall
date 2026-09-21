"""One-shot bounded resource dependency wait; only resume the identified owned run."""
import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True);p.add_argument('--max-wait',type=int,default=1200)
a=p.parse_args();assert 0<a.max_wait<=1200
pause_path=a.run.with_suffix('.resource-pause.json');pause=json.loads(pause_path.read_text())
assert pause['status']=='suspended_for_resource_scheduling'
report=json.loads(a.run.read_text());pid=report['pid'];assert pid==pause['pid']
def identity(pid):
 try:
  base=Path('/proc')/str(pid);stat=(base/'stat').read_text().rsplit(')',1)[1].split()
  if stat[0]=='Z':return None
  return (stat[19],(base/'cmdline').read_bytes())
 except FileNotFoundError:return None
owned=identity(pid);assert owned and b'scripts/train_wall_rally.py' in owned[1] and str(a.run).encode() in owned[1]
deps=[]
for name in pause['resume_after']:
 d=json.loads((a.run.parent/(name+'.json')).read_text());deps.append((name,d['pid'],identity(d['pid'])))
output=a.run.with_suffix('.resource-resume.json');assert not output.exists()
result={'pid':os.getpid(),'target_pid':pid,'status':'waiting','started_unix':time.time(),'max_wait_seconds':a.max_wait,'dependencies':[d[0] for d in deps]}
def save():
 temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output)
save();start=time.monotonic()
try:
 while True:
  current=identity(pid)
  if current!=owned:raise RuntimeError('Original suspended target ended or identity changed; no signal sent')
  remaining=[name for name,dep_id,original in deps if original is not None and identity(dep_id)==original]
  if not remaining or time.monotonic()-start>=a.max_wait:
   assert identity(pid)==owned
   os.kill(pid,signal.SIGCONT)
   result.update(status='resumed',resumed_unix=time.time(),reason='dependencies_ended' if not remaining else 'bounded_wait_expired',dependencies_still_live=remaining)
   pause.update(status='resumed',resumed_unix=result['resumed_unix']);pause_path.write_text(json.dumps(pause,indent=2)+'\n');save();break
  time.sleep(10)
except Exception as exc:
 result.update(status='failed',error=repr(exc));save();raise
print(json.dumps(result))
