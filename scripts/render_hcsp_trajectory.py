"""Render measured one-drone trajectories as a labelled schematic, not camera footage."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--env',type=int,default=3);a=p.parse_args()
r=json.loads(a.report.read_text());assert r['status']=='passed'
z=np.load(r['trajectory']);assert hashlib.sha256(Path(r['trajectory']).read_bytes()).hexdigest()==r['trajectory_sha256']
i=a.env;count=int(z['active'][:,i].sum());b=z['ball'][:count,i,0];d=z['physical_drone'][:count,i,0]
fig,axes=plt.subplots(1,2,figsize=(10,4),dpi=110)
for ax,vertical,label,limits in zip(axes,[2,1],['height z (m)','lateral y (m)'],[(0,6),(-4,4)]):
 ax.axvspan(-.1,.1,color='#687785',alpha=.6,label='wall');ax.set(xlim=(-.3,7),ylim=limits,xlabel='x (m)',ylabel=label);ax.grid(alpha=.2)
 ax.plot(b[:,0],b[:,vertical],color='#ec9c35',alpha=.25);ax.plot(d[:,0],d[:,vertical],color='#3479bc',alpha=.25)
axes[0].set_title('Side view');axes[1].set_title('Top view')
lines=[]
for ax in axes:
 ball,=ax.plot([],[],'o',color='#ec9c35',label='ball');drone,=ax.plot([],[],'s',color='#3479bc',label='single drone');lines.append((ball,drone))
axes[0].legend(loc='upper right',fontsize=8)
fig.suptitle(f'Measured trajectory schematic | scene {i} | not camera footage',fontsize=11)
text=fig.text(.5,.01,'',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94])
a.output.parent.mkdir(parents=True,exist_ok=True)
writer=FFMpegWriter(fps=25,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
with writer.saving(fig,str(a.output),110):
 for t in range(0,count,max(1,round(.04/r['dt']))):
  for (ball,drone),v in zip(lines,[2,1]):ball.set_data([b[t,0]],[b[t,v]]);drone.set_data([d[t,0]],[d[t,v]])
  text.set_text(f't={(t+1)*r["dt"]:.2f}s | role={int(z["executed_role"][t,i])} | body-wall-body={int(z["wall_returns"][t,i])}')
  writer.grab_frame()
fig.savefig(a.output.with_suffix('.png'))
meta={'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),'env':i,'steps':count,'representation':'Recorded position schematic; no camera rendering','video_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
