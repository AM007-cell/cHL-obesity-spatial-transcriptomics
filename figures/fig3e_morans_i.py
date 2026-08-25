#!/usr/bin/env python3
"""Regenerate Fig 2B (three samples) and Fig 3E (Moran's I vs higher control)."""
import numpy as np, json, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from PIL import Image
import spatial_relationship as SR

OUT='/mnt/user-data/outputs/panels/'
DPI=300; BLUE='#2980B9'; BLUE2='#5DADE2'; RED='#C0392B'
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42})
cmap_expr=LinearSegmentedColormap.from_list('e',['#EFEFEF','#FFF176','#FF6F00','#B71C1C'])
rng=np.random.default_rng(0)

def save(fig,n):
    for e in ['pdf','png']:
        fig.savefig(f'{OUT}{n}.{e}',dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(' ->',n)

def load(sid):
    d=f'data/{sid}-analysis/'
    spots,raw,_,_=SR.load_sample(d+'filtered_feature_bc_matrix.h5',
        d+'tissue_positions_list.csv',d+'scalefactors_json.json')
    lo=json.load(open(d+'scalefactors_json.json'))['tissue_lowres_scalef']
    with h5py.File(d+'filtered_feature_bc_matrix.h5','r') as f:
        g=f['matrix/features/name'][:].astype(str)
        m=csc_matrix((f['matrix/data'][:],f['matrix/indices'][:],f['matrix/indptr'][:]),
                     shape=f['matrix/shape'][:])
    m=m[:,spots['mi'].values].tocsc(); u=np.asarray(m.sum(0)).ravel()
    p=np.minimum(1.0,410/np.maximum(u,1)); co=m.tocoo()
    ds=csc_matrix((rng.binomial(co.data.astype(int),p[co.col]),(co.row,co.col)),shape=m.shape)
    return dict(img=Image.open(d+'tissue_lowres_image.png'),raw=raw,idx={k:i for i,k in enumerate(g)},
                ds=ds,px=spots['pxl_col_fullres'].values*lo,py=spots['pxl_row_fullres'].values*lo,
                xy=spots[['x_um','y_um']].values)

S={c:load(s) for c,s in [('C1','C1'),('C2','C2'),('O2','O2')]}
ROWS=[('C1','C1 (Non-obese)',BLUE),('C2','C2 (Non-obese)',BLUE2),('O2','O2 (Obese)',RED)]

# ---------------- Fig 2B : three samples --------------------------------------
def featgrid(genes,name,figsize):
    fig,axes=plt.subplots(3,len(genes),figsize=figsize,squeeze=False)
    for j,gene in enumerate(genes):
        vmax=max(np.percentile(S[c]['raw'](gene),99.5) for c,_,_ in ROWS)
        vmax=max(vmax,1)
        for r,(code,lab,col) in enumerate(ROWS):
            ax=axes[r][j]; d=S[code]; v=d['raw'](gene); o=np.argsort(v)
            ax.imshow(d['img'],alpha=0.40,aspect='equal')
            sc=ax.scatter(d['px'][o],d['py'][o],c=v[o],cmap=cmap_expr,s=1.5,vmin=0,vmax=vmax,
                          linewidths=0,rasterized=True)
            cb=plt.colorbar(sc,ax=ax,fraction=0.040,pad=0.012,shrink=0.58)
            cb.ax.tick_params(labelsize=4.4,length=1.2); cb.outline.set_linewidth(0.3)
            cb.set_label('counts',fontsize=4.9)
            ax.set_title(f'{gene}  ({100*(v>0).mean():.0f}% spots)',fontsize=6.6,pad=2,style='italic')
            ax.axis('off')
            if j==0:
                ax.text(-0.07,0.5,lab,transform=ax.transAxes,rotation=90,va='center',
                        ha='center',fontsize=6.6,fontweight='bold',color=col)
    fig.subplots_adjust(hspace=0.12,wspace=0.04)
    save(fig,name)




# ---------------- Fig 3E : Moran's I vs higher control ------------------------
FAM=[('Plasma cell / Ig','#6A4C93',['IGKC','IGHG1','IGLC1','MZB1','CXCL13']),
     ('Antigen presentation','#1B998B',['HLA-C','B2M','CD74']),
     ('Metabolic / UPR','#E07A5F',['TXNIP','XBP1','SFRP4']),
     ('AP-1','#B58900',['FOS','FOSB']),
     ('ECM / stromal','#6C757D',['COL1A2','COL3A1']),
     ('CCL22','#C0392B',['CCL22'])]

OFFS={'IGKC':(-24,-9),'TXNIP':(-4,-9),'HLA-C':(4,-1),'B2M':(-16,-8),'CD74':(3,4),
      'CXCL13':(-30,-3),'MZB1':(-6,6)}

def moran(xy,v):
    _,nn=cKDTree(xy).query(xy,k=7); nn=nn[:,1:]
    z=v-v.mean()
    return np.nan if z.std()==0 else (z*z[nn].mean(1)).sum()/(z**2).sum()

def mi(code,g):
    d=S[code]; t=np.asarray(d['ds'].sum(0)).ravel(); t[t==0]=1
    return moran(d['xy'],np.log1p(np.asarray(d['ds'][d['idx'][g]].todense()).ravel()/t*1e4))

fig,ax=plt.subplots(figsize=(4.3,4.0))
ax.plot([-0.05,0.80],[-0.05,0.80],color='0.6',lw=0.7,ls='--',zorder=1)
ax.fill_between([-0.05,0.80],[-0.05,0.80],[0.80,0.80],color=RED,alpha=0.04,zorder=0)
ax.fill_between([-0.05,0.80],[-0.05,-0.05],[-0.05,0.80],color=BLUE,alpha=0.04,zorder=0)
for fam,col,genes in FAM:
    xs,ys=[],[]
    for g in genes:
        x=max(mi('C1',g),mi('C2',g)); y=mi('O2',g)
        xs.append(x); ys.append(y)
        off=OFFS.get(g,(3.2,2.6))
        ax.annotate(g,(x,y),xytext=off,textcoords='offset points',fontsize=5.2,
                    color=col,style='italic')
    ax.scatter(xs,ys,s=17,color=col,label=fam,zorder=3,linewidths=0)
ax.set_xlabel("Moran's I — higher of C1 and C2 (non-obese)",fontsize=7.5)
ax.set_ylabel("Moran's I — O2 (obese)",fontsize=7.5)
ax.set_xlim(-0.05,0.80); ax.set_ylim(-0.05,0.80)
ax.tick_params(labelsize=6.5,length=2.2)
ax.text(0.03,0.955,'higher in obese',transform=ax.transAxes,fontsize=5.8,color=RED,style='italic')
ax.text(0.985,0.05,'higher in non-obese',transform=ax.transAxes,fontsize=5.8,color=BLUE,
        style='italic',ha='right')
ax.legend(fontsize=5.6,loc='upper right',handletextpad=0.35,borderaxespad=0.5,
          labelspacing=0.32,frameon=True,framealpha=0.92,edgecolor='none',facecolor='white')
ax.set_title('Spatial autocorrelation at matched depth',fontsize=8.5,fontweight='bold',pad=4)

for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_linewidth(0.6)
save(fig,'Fig3E_morans_I_three_samples')
