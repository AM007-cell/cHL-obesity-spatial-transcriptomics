import numpy as np, json, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.sparse import csc_matrix
from PIL import Image
import spatial_relationship as SR

OUT='/mnt/user-data/outputs/panels/'; DPI=300
BLUE='#2980B9'; BLUE2='#5DADE2'; RED='#C0392B'
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':600,'savefig.dpi':600})
cmap=LinearSegmentedColormap.from_list('e',['#FFF9C4','#FFD54F','#FF6F00','#B71C1C'])

def load(sid):
    d=f'data/{sid}-analysis/'
    spots,raw,_,_=SR.load_sample(d+'filtered_feature_bc_matrix.h5',
        d+'tissue_positions_list.csv',d+'scalefactors_json.json')
    lo=json.load(open(d+'scalefactors_json.json'))['tissue_lowres_scalef']
    return dict(img=Image.open(d+'tissue_lowres_image.png'),raw=raw,
                px=spots['pxl_col_fullres'].values*lo,py=spots['pxl_row_fullres'].values*lo)
S={c:load(s) for c,s in [('C1','C1'),('C2','C2'),('O2','O2')]}
ROWS=[('C1','C1\nNon-obese',BLUE),('C2','C2\nNon-obese',BLUE2),('O2','O2\nObese',RED)]
GENES=['IGKC','IGHG1','IGHA1','XBP1','MZB1','CXCL13']

fig=plt.figure(figsize=(7.087,4.15))
gs=fig.add_gridspec(4,6,height_ratios=[1,1,1,0.07],hspace=0.16,wspace=0.035,
                    left=0.055,right=0.995,top=0.905,bottom=0.10)
for j,g in enumerate(GENES):
    vmax=max(max(np.percentile(S[c]['raw'](g),99.5) for c,_,_ in ROWS),2.0)
    for r,(code,lab,col) in enumerate(ROWS):
        ax=fig.add_subplot(gs[r,j]); d=S[code]; v=d['raw'](g)
        ax.imshow(d['img'],alpha=0.38,aspect='equal')
        z=v==0
        ax.scatter(d['px'][z],d['py'][z],c='#E8E8E8',s=2.0,linewidths=0,rasterized=True)
        nz=~z; o=np.argsort(v[nz])
        sc=ax.scatter(d['px'][nz][o],d['py'][nz][o],c=np.clip(v[nz][o],1,vmax),cmap=cmap,
                      norm=LogNorm(vmin=1,vmax=vmax),s=2.0,linewidths=0,rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(False)
        pct=f'{100*(v>0).mean():.0f}%'
        if r==0:
            ax.set_title(g,fontsize=7,style='italic',pad=10)
            ax.text(0.5,1.008,pct,transform=ax.transAxes,ha='center',va='bottom',
                    fontsize=5.6,color='0.25')
        else:
            ax.set_title(pct,fontsize=5.6,color='0.25',pad=2)
        if j==0:
            ax.text(-0.10,0.5,lab,transform=ax.transAxes,rotation=90,va='center',ha='center',
                    fontsize=6,fontweight='bold',color=col,linespacing=1.25)
    cax=fig.add_subplot(gs[3,j]); pos=cax.get_position()
    cax.set_position([pos.x0+pos.width*0.10,pos.y0,pos.width*0.80,pos.height])
    cb=fig.colorbar(sc,cax=cax,orientation='horizontal')
    from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
    cb.ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4))
    cb.ax.xaxis.set_minor_formatter(NullFormatter())
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f'{v:.0f}'))
    cb.ax.tick_params(labelsize=4.4,length=1.1,pad=0.8,which='major')
    cb.ax.tick_params(length=0,which='minor')
    cb.outline.set_linewidth(0.3)
fig.text(0.5,0.028,'counts (log scale, shared across samples within each gene)',
         ha='center',fontsize=5.2,color='0.45',style='italic')
for e in ['pdf','png']:
    fig.savefig(f'{OUT}Fig2B_compact_3samples.{e}',dpi=600,bbox_inches='tight',facecolor='white')
print('done')
