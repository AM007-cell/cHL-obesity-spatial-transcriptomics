import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42})
OUT='/mnt/user-data/outputs/panels/'
BLUE='#2980B9'; RED='#C0392B'
XL='/mnt/user-data/uploads/markers_obese_vs_control.xlsx'
df=pd.concat([pd.read_excel(XL,str(i)) for i in range(1,9)],ignore_index=True)

UP=['IGKC','IGHG1','IGHA1','IGLC1','IGHG4','IGHM','MZB1','JCHAIN','XBP1','DERL3','CXCL13','CCL19','CCL21']
DOWN=['CCL22','COL1A2','COL3A1']
FLAG={'COL1A2','COL3A1'}
GENES=UP+DOWN
cmap=LinearSegmentedColormap.from_list('d',['#0B3C5D','#4A90C2','#D6E6F2','#F7F7F7',
                                            '#F5C6BE','#D4695C','#8C2F22'])
fig,ax=plt.subplots(figsize=(7.087,2.80))
# shaded band behind the two flagged columns
for g in FLAG:
    x=GENES.index(g)
    ax.add_patch(Rectangle((x-0.5,-0.6),1,8.2,facecolor='#EFEFEF',edgecolor='none',zorder=0))
for j,g in enumerate(GENES):
    s=df[df.gene==g]
    for _,r in s.iterrows():
        ax.scatter(j,int(r['cluster'])-1,s=18+r['pct.1']*95,c=[r['avg_log2FC']],cmap=cmap,
                   norm=TwoSlopeNorm(vcenter=0,vmin=-6,vmax=6),linewidths=0.25,
                   edgecolors='white',zorder=3)
sep=len(UP)-0.5
ax.axvline(sep,color='0.55',lw=0.8,ls='--',zorder=2)
ax.text((len(UP)-1)/2,8.05,'↑ Upregulated in obese',ha='center',fontsize=6,color=RED,fontweight='bold')
ax.text(sep+(len(DOWN))/2,8.05,'↓ Downregulated in obese',ha='center',fontsize=6,color=BLUE,fontweight='bold')
ax.set_xticks(range(len(GENES)))
ax.set_xticklabels([g+' †' if g in FLAG else g for g in GENES],fontsize=6,style='italic',
                   rotation=45,ha='right')
ax.set_yticks(range(8)); ax.set_yticklabels([f'C{i}' for i in range(1,9)],fontsize=6.5)
ax.set_xlim(-0.7,len(GENES)-0.3); ax.set_ylim(-0.7,8.4)
ax.tick_params(length=1.8)
for s_ in ['top','right']: ax.spines[s_].set_visible(False)
for s_ in ['left','bottom']: ax.spines[s_].set_linewidth(0.6)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=TwoSlopeNorm(vcenter=0,vmin=-6,vmax=6))
cb=plt.colorbar(sm,ax=ax,fraction=0.020,pad=0.02,shrink=0.62,anchor=(0.0,1.0))
cb.set_label('log$_2$FC',fontsize=6.2); cb.ax.tick_params(labelsize=5.4,length=1.5)
cb.outline.set_linewidth(0.3)
for p in [0.25,0.50,0.75,1.00]:
    ax.scatter([],[],s=18+p*95,c='0.55',linewidths=0.25,edgecolors='white',label=f'{int(p*100)}%')
ax.legend(title='% spots expressing\n(obese)',fontsize=5.2,title_fontsize=5.4,frameon=False,
          loc='upper left',bbox_to_anchor=(1.055,0.40),labelspacing=0.85,handletextpad=0.7,
          borderpad=0.2)
ax.text(0,-0.52,'† significance not retained after correction for sequencing depth (Figure 4E)',
        transform=ax.transAxes,fontsize=5.2,color='0.4',style='italic')
for e in ['pdf','png']:
    fig.savefig(f'{OUT}Fig2A_dotplot_flagged.{e}',dpi=600,bbox_inches='tight',facecolor='white')
print('done')
