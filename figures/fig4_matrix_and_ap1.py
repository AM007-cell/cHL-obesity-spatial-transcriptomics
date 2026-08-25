#!/usr/bin/env python3
"""Figure 4 - Wnt antagonism, immediate-early activation and matrix compositional shift."""
import numpy as np, pandas as pd, json, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csc_matrix
from PIL import Image
import spatial_relationship as SR

OUT='/mnt/user-data/outputs/panels/'
DPI=300; BLUE='#2980B9'; BLUE2='#7FB3D5'; RED='#C0392B'
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42})
cmap_expr=LinearSegmentedColormap.from_list('e',['#EFEFEF','#FFF176','#FF6F00','#B71C1C'])
cmap_seq=LinearSegmentedColormap.from_list('d',['#EDE4F5','#B39DDB','#7E57C2','#4527A0'])
rng=np.random.default_rng(0)
XL='/mnt/user-data/uploads/markers_obese_vs_control.xlsx'
df=pd.concat([pd.read_excel(XL,str(i)) for i in range(1,9)],ignore_index=True)

def save(fig,n):
    for e in ['pdf','png']:
        fig.savefig(f'{OUT}{n}.{e}',dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(' ->',n)

def despine(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_linewidth(0.6)

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
                ds=ds,px=spots['pxl_col_fullres'].values*lo,py=spots['pxl_row_fullres'].values*lo)

S={c:load(s) for c,s in [('C1','C1'),('C2','C2'),('O2','O2')]}

# ---- A : SFRP4 per cluster ---------------------------------------------------
c=df[df.gene=='SFRP4'].sort_values('cluster')
fig,ax=plt.subplots(figsize=(3.35,2.75))
xs=np.arange(len(c))
ax.bar(xs,c['avg_log2FC'],0.62,color=RED,edgecolor='none')
for i,(_,r) in enumerate(c.iterrows()):
    ax.text(i,r['avg_log2FC']-0.055,f"{r['avg_log2FC']:.2f}",ha='center',va='top',
            fontsize=6.4,fontweight='bold',color='white')
    ax.text(i,r['avg_log2FC']+0.05,f"{r['pct.1']*100:.0f}%\nvs {r['pct.2']*100:.0f}%",
            ha='center',va='bottom',fontsize=5.6,color='0.35',linespacing=1.15)
ax.axhline(0,color='0.3',lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels([f'C{int(v)}' for v in c['cluster']],fontsize=7)
ax.set_ylabel('log$_2$FC (obese vs. non-obese)',fontsize=7.5)
ax.set_ylim(0,max(c['avg_log2FC'])*1.52)
ax.tick_params(labelsize=6.5,length=2.2); ax.tick_params(axis='x',length=0)
ax.set_title('SFRP4 upregulation',fontsize=8.5,fontweight='bold',pad=4,style='italic')
ax.text(0.5,0.995,'obese vs. non-obese, % spots expressing',transform=ax.transAxes,ha='center',va='top',fontsize=5.6,color='0.45',style='italic')
despine(ax); save(fig,'Fig4A_SFRP4_clusters')

# ---- B : SFRP4 spatial -------------------------------------------------------
fig,axes=plt.subplots(2,1,figsize=(2.7,5.2))
vmax=max(np.percentile(S[c2]['raw']('SFRP4'),99.5) for c2 in ['C1','O2'])
for r,(code,lab,col) in enumerate([('C1','C1 (Non-obese)',BLUE),('O2','O2 (Obese)',RED)]):
    ax=axes[r]; d=S[code]; v=d['raw']('SFRP4'); o=np.argsort(v)
    ax.imshow(d['img'],alpha=0.40,aspect='equal')
    sc=ax.scatter(d['px'][o],d['py'][o],c=v[o],cmap=cmap_expr,s=1.9,vmin=0,vmax=max(vmax,1),
                  linewidths=0,rasterized=True)
    cb=plt.colorbar(sc,ax=ax,fraction=0.040,pad=0.012,shrink=0.60)
    cb.ax.tick_params(labelsize=4.8,length=1.3); cb.outline.set_linewidth(0.3)
    cb.set_label('counts',fontsize=5.2)
    ax.set_title(f'SFRP4  ({100*(v>0).mean():.0f}% spots)',fontsize=7,pad=2,style='italic')
    ax.axis('off')
    ax.text(-0.06,0.5,lab,transform=ax.transAxes,rotation=90,va='center',ha='center',
            fontsize=7,fontweight='bold',color=col)
fig.subplots_adjust(hspace=0.10)
save(fig,'Fig4B_SFRP4_spatial')

# ---- C : immediate-early genes heatmap --------------------------------------
rows=['FOS','JUNB','EGR1','FOSB','JUN']
M=np.full((len(rows),8),np.nan)
for i,g in enumerate(rows):
    for _,r in df[df.gene==g].iterrows(): M[i,int(r['cluster'])-1]=r['avg_log2FC']
fig,ax=plt.subplots(figsize=(3.9,2.4))
im=ax.imshow(M,cmap=cmap_seq,vmin=1.0,vmax=2.2,aspect='auto')
ax.set_xticks(range(8)); ax.set_xticklabels([f'C{i}' for i in range(1,9)],fontsize=6.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=7,style='italic')
for i in range(len(rows)):
    for j in range(8):
        if np.isnan(M[i,j]):
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor='#D5D8DC',edgecolor='white',lw=0.5))
        else:
            ax.text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=5.4,
                    color='white' if M[i,j]>1.8 else '0.15')
ax.axhline(2.5,color='white',lw=1.8)
ax.set_xticks(np.arange(-.5,8,1),minor=True); ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
ax.grid(which='minor',color='white',lw=0.5); ax.tick_params(which='minor',length=0)
ax.tick_params(length=1.8)
ax.text(8.62,1.0,'depth-robust',fontsize=6,rotation=270,va='center',ha='center',color='0.35')
ax.text(8.62,3.5,'C2-driven',fontsize=6,rotation=270,va='center',ha='center',color='0.35')
cb=plt.colorbar(im,ax=ax,fraction=0.030,pad=0.20,shrink=0.9)
cb.set_label('log$_2$FC',fontsize=6.5); cb.ax.tick_params(labelsize=5.6,length=1.6)
cb.outline.set_linewidth(0.3)
ax.set_title('Immediate-early / AP-1 response',fontsize=8.5,fontweight='bold',pad=4)
ax.text(0.5,-0.24,'grey = not differentially expressed in that cluster',transform=ax.transAxes,
        ha='center',fontsize=5.8,color='0.45',style='italic')
save(fig,'Fig4C_AP1_clusters')

# ---- D : matrix composition at matched depth --------------------------------
GEN=[('SFRP4','Wnt'),('LUM','Matrix'),('FBLN1','Matrix'),('C7','Matrix'),
     ('COL1A2','Collagen'),('COL3A1','Collagen'),('PCOLCE','Processing')]
fig,ax=plt.subplots(figsize=(4.6,3.0))
w=0.26; xs=np.arange(len(GEN))
for k,(code,col,lab) in enumerate([('C1',BLUE,'C1 (non-obese)'),('C2',BLUE2,'C2 (non-obese)'),
                                   ('O2',RED,'O2 (obese)')]):
    vals=[100*(np.asarray(S[code]['ds'][S[code]['idx'][g]].todense()).ravel()>0).mean()
          for g,_ in GEN]
    ax.bar(xs+(k-1)*w,vals,w,color=col,edgecolor='none',label=lab)
ax.set_xticks(xs)
ax.set_xticklabels([g for g,_ in GEN],fontsize=6.8,style='italic',rotation=30,ha='right')
ax.set_ylabel('% spots with detectable expression',fontsize=7.5)
ax.tick_params(labelsize=6.5,length=2.2); ax.tick_params(axis='x',length=0)
ax.legend(fontsize=6.2,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.30),
          ncol=3,handlelength=1.2,handleheight=0.9,columnspacing=1.8)
ax.set_title('Matrix composition at matched sequencing depth',fontsize=8.5,fontweight='bold',pad=4)
ax.text(0.5,-0.44,'all samples downsampled to 410 UMI per spot',transform=ax.transAxes,
        ha='center',fontsize=5.8,color='0.45',style='italic')
despine(ax); save(fig,'Fig4D_matrix_matched_depth')
