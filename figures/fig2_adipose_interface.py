#!/usr/bin/env python3
"""Figure 2C - spatial relationship between adipose tissue and plasma cell infiltrate."""
import numpy as np, h5py, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from PIL import Image
import spatial_relationship as SR

FONT='Liberation Sans'; DPI=300
BLUE='#2980B9'; RED='#C0392B'; ADIP='#01579B'
plt.rcParams.update({'font.family':FONT,'pdf.fonttype':42,'ps.fonttype':42})
cmap_plasma=LinearSegmentedColormap.from_list('p',['#EBEBEB','#FFF176','#FF6F00','#B71C1C'])
CTRL=['ACTB','GAPDH','RPL13A','TMSB4X','EEF1A1']

def prep(sid):
    d=f'data/{sid}-analysis/'
    spots,raw,lognorm,_=SR.load_sample(d+'filtered_feature_bc_matrix.h5',
        d+'tissue_positions_list.csv',d+'scalefactors_json.json')
    sf=json.load(open(d+'scalefactors_json.json')); lo=sf['tissue_lowres_scalef']
    img=Image.open(d+'tissue_lowres_image.png')
    adip=SR.module_score(lognorm,SR.ADIPO); plas=SR.module_score(lognorm,SR.PLASMA)
    ctrl=SR.module_score(lognorm,CTRL)
    with h5py.File(d+'filtered_feature_bc_matrix.h5','r') as f:
        mat=csc_matrix((f['matrix/data'][:],f['matrix/indices'][:],f['matrix/indptr'][:]),
                       shape=f['matrix/shape'][:])
    ld=np.log10(np.asarray(mat[:,spots['mi'].values].sum(0)).ravel()+1)
    X=np.c_[np.ones_like(ld),ld]
    res=lambda v: v-X@np.linalg.lstsq(X,v,rcond=None)[0]
    mask=SR.adipocyte_mask(raw,adip)
    xy=spots[['x_um','y_um']].values
    prof=SR.distance_profile(xy,mask,res(plas),np.random.default_rng(0))
    L,p=SR.lees_l(xy,res(adip),res(plas),np.random.default_rng(0))
    Lc,pc=SR.lees_l(xy,adip,ctrl,np.random.default_rng(0))
    return dict(img=img,px=spots['pxl_col_fullres'].values*lo,py=spots['pxl_row_fullres'].values*lo,
                xy=xy,mask=mask,plas_r=res(plas),prof=prof,L=L,p=p,Lc=Lc,pc=pc,n=len(spots))

S={'C1':prep('C1'),'O2':prep('O2')}

fig=plt.figure(figsize=(7.087,6.30),facecolor='white')
gs=GridSpec(2,2,figure=fig,hspace=0.26,wspace=0.22,height_ratios=[1.05,0.85])

def border(ax,c):
    for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(c); s.set_linewidth(1.3)

def tag(ax,letter,col):
    ax.text(0.035,0.955,letter,transform=ax.transAxes,fontsize=11,fontweight='bold',
            color='white',va='top',bbox=dict(boxstyle='round,pad=0.22',facecolor=col,alpha=0.85,lw=0))

# --- A / B : maps -------------------------------------------------------------
for k,(code,letter,col,lab) in enumerate([('C1','C',BLUE,'Non-obese'),('O2','D',RED,'Obese')]):
    d=S[code]; ax=fig.add_subplot(gs[0,k])
    ax.imshow(d['img'],alpha=0.42,aspect='equal')
    m=~d['mask']
    v=d['plas_r'][m]; vmax=np.percentile(v,97); vmin=np.percentile(v,3)
    sc=ax.scatter(d['px'][m],d['py'][m],c=v,cmap=cmap_plasma,s=1.6,alpha=0.9,
                  linewidths=0,vmin=vmin,vmax=vmax,rasterized=True)
    ax.scatter(d['px'][d['mask']],d['py'][d['mask']],s=3.4,c=ADIP,alpha=0.95,
               linewidths=0,rasterized=True)
    # 250 um exclusion contour
    gx=np.linspace(d['xy'][:,0].min(),d['xy'][:,0].max(),260)
    gy=np.linspace(d['xy'][:,1].min(),d['xy'][:,1].max(),260)
    GX,GY=np.meshgrid(gx,gy)
    dist,_=cKDTree(d['xy'][d['mask']]).query(np.c_[GX.ravel(),GY.ravel()],k=1)
    hull,_=cKDTree(d['xy']).query(np.c_[GX.ravel(),GY.ravel()],k=1)
    Z=np.where(hull<120,dist,np.nan).reshape(GX.shape)
    sx=(d['px'].max()-d['px'].min())/(d['xy'][:,0].max()-d['xy'][:,0].min())
    ax.contour(d['px'].min()+(GX-d['xy'][:,0].min())*sx,
               d['py'].min()+(GY-d['xy'][:,1].min())*sx,Z,levels=[250],
               colors=[ADIP],linewidths=0.7,linestyles='--')
    cb=plt.colorbar(sc,ax=ax,fraction=0.036,pad=0.012,shrink=0.62)
    cb.set_label('Plasma cell module\n(depth-adjusted)',fontsize=5.6,linespacing=1.1)
    cb.ax.tick_params(labelsize=4.8,length=1.6); cb.outline.set_linewidth(0.3)
    ax.set_title(f'{code} — {lab}\n{d["mask"].sum()} adipocyte-positive spots / {d["n"]:,}',
                 fontsize=8,fontweight='bold',pad=3,linespacing=1.3)
    ax.axis('off'); border(ax,col)

fig.legend(handles=[mpatches.Patch(color=ADIP,label='Adipocyte-positive spot'),
                    plt.Line2D([],[],color=ADIP,ls='--',lw=0.9,label='250 µm from adipose')],
           loc='upper center',bbox_to_anchor=(0.5,0.512),ncol=2,frameon=False,
           fontsize=6.2,handlelength=1.5,columnspacing=1.8)

# --- C : distance-decay profiles ---------------------------------------------
ax=fig.add_subplot(gs[1,0])
for code,col in [('C1',BLUE),('O2',RED)]:
    p=S[code]['prof']; ok=np.isfinite(p['obs'])
    ax.fill_between(p['centre'][ok],p['null_lo'][ok],p['null_hi'][ok],
                    color=col,alpha=0.10,lw=0,zorder=1)
    ax.fill_between(p['centre'][ok],p['ci_lo'][ok],p['ci_hi'][ok],color=col,alpha=0.28,lw=0,zorder=2)
    ax.plot(p['centre'][ok],p['obs'][ok],'-o',color=col,ms=2.6,lw=1.3,zorder=3,label=code)
ax.axhline(0,color='0.45',lw=0.6,ls=':')
ax.set_xlabel('Distance to nearest adipocyte-positive spot (µm)',fontsize=7)
ax.set_ylabel('Plasma cell module\n(depth-adjusted)',fontsize=7,linespacing=1.2)
ax.tick_params(labelsize=6,length=2.2)
ax.legend(fontsize=6.2,frameon=False,loc='lower right',handlelength=1.3)
ax.set_title('Distance-decay profile',fontsize=8,fontweight='bold',pad=3)
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_linewidth(0.6)

# --- D : Lee's L --------------------------------------------------------------
ax=fig.add_subplot(gs[1,1])
xs=np.arange(2); w=0.34
main=[S['C1']['L'],S['O2']['L']]; ctl=[S['C1']['Lc'],S['O2']['Lc']]
ax.bar(xs-w/2,main,w,color=[BLUE,RED],edgecolor='none',label="Adipocyte vs plasma cell")
ax.bar(xs+w/2,ctl,w,color='none',edgecolor='0.45',lw=0.8,hatch='////',
       label='Adipocyte vs housekeeping (control)')
ax.axhline(0,color='0.3',lw=0.7)
for i,code in enumerate(['C1','O2']):
    d=S[code]
    off=-0.012 if d['L']<0 else 0.012
    ax.text(i-w/2,d['L']+off,f"{d['L']:+.3f}",ha='center',
            va='top' if d['L']<0 else 'bottom',fontsize=5.8,fontweight='bold')
    off=-0.012 if d['Lc']<0 else 0.012
    ax.text(i+w/2,d['Lc']+off,f"{d['Lc']:+.3f}",ha='center',
            va='top' if d['Lc']<0 else 'bottom',fontsize=5.8,color='0.35')
ax.set_xticks(xs); ax.set_xticklabels(['C1\nnon-obese','O2\nobese'],fontsize=6.5,linespacing=1.4)
ax.set_ylabel("Lee's L (depth-adjusted)",fontsize=7)
ax.tick_params(axis='y',labelsize=6,length=2.2); ax.tick_params(axis='x',length=0)
ax.set_ylim(-0.30,0.34)
ax.legend(fontsize=5.6,frameon=False,loc='upper right',handlelength=1.2,
          handleheight=0.9,borderaxespad=0.2)
ax.set_title('Bivariate spatial association',fontsize=8,fontweight='bold',pad=3)
ax.text(0.5,-0.20,'negative = spatial segregation   ·   positive = co-localisation',
        transform=ax.transAxes,fontsize=5.4,color='0.45',ha='center',style='italic')
for s_ in ['top','right']: ax.spines[s_].set_visible(False)
for s_ in ['left','bottom']: ax.spines[s_].set_linewidth(0.6)

for ext in ['pdf','png']:
    fig.savefig(f'/mnt/user-data/outputs/Fig2C_SpatialRelationship_FINAL.{ext}',
                dpi=DPI,bbox_inches='tight',facecolor='white')
print('C1 L=%.3f p=%.4f | ctrl %.3f' % (S['C1']['L'],S['C1']['p'],S['C1']['Lc']))
print('O2 L=%.3f p=%.4f | ctrl %.3f' % (S['O2']['L'],S['O2']['p'],S['O2']['Lc']))
