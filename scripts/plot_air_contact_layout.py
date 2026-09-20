"""Dimension-based schematic of original and aligned Air contact surfaces."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser();p.add_argument('--layout',type=Path,required=True);p.add_argument('--aligned-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
l=json.loads(a.layout.read_text());r=json.loads(a.aligned_report.read_text());offset=r['bat_overlay']['collider_z_offset'];visual_top=r['bat_overlay']['visual_top_z']
fig,axes=plt.subplots(1,2,figsize=(11,6),sharey=True)
for ax,dz,title in zip(axes,[0,offset],['Original colliders with body enabled','Project-stage aligned collider']):
 lo,hi=l['body_collision_box_min'],l['body_collision_box_max'];rad=l['bat_cylinder_radius'];height=l['bat_cylinder_height']
 ax.add_patch(Rectangle((lo[0]*100,lo[2]*100),(hi[0]-lo[0])*100,(hi[2]-lo[2])*100,facecolor='#708391',alpha=.45,label='Body collision box'))
 ax.add_patch(Rectangle((-rad*100,(-height/2+dz)*100),rad*200,height*100,facecolor='#df9140',edgecolor='#aa5511',alpha=.35,label='Bat cylinder (cross-section)'))
 ax.axhline(visual_top*100,color='#2677ac',ls=':',label='Authored visual bat top')
 ax.plot([-rad*100,rad*100],[(height/2+dz)*100]*2,color='#a84716',lw=3,label='Physical cap surface')
 ax.annotate(f'cap {(height/2+dz)*100:.1f} cm',xy=(0,(height/2+dz)*100),xytext=(-8,11),arrowprops={'arrowstyle':'->'})
 ax.annotate(f'body top {hi[2]*100:.1f} cm',xy=(hi[0]*100,hi[2]*100),xytext=(6,8),arrowprops={'arrowstyle':'->'})
 ax.set_xlim(-11,14);ax.set_ylim(-8,14);ax.set_aspect('equal');ax.set_xlabel('Local x (cm)');ax.set_title(title);ax.grid(alpha=.2)
axes[0].set_ylabel('Height relative to bat/body frame (cm)');axes[0].legend(loc='lower left',fontsize=8)
fig.suptitle('Air collider audit: original body top occludes the central bat cap\nAuthored-dimension schematic, not a rendered simulation or learned trajectory',fontsize=12)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);plt.close(fig)
