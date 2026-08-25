#!/usr/bin/env python3
"""Supplementary Figure 1 — per-section quality control."""
import numpy as np, json, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from scipy.sparse import csc_matrix
from PIL import Image
import spatial_relationship as SR

OUT='/mnt/user-data/outputs/panels/'
BLUE='#2980B9'; BLUE2='#5DADE2'; RED='#C0392B'; RED2='#E59866'
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42,
                     'figure.dpi':600,'savefig.dpi':600})
cmap=LinearSegmentedColormap.from_list('q',['#F7F7F7','#9ECAE1','#3182BD','#08306B'])

S=[('C1','C1','C1  Non-obese',BLUE),
   ('C2','C2','C2  Non-obese',BLUE2),
   ('O1','O1','O1  Obese  (excluded)',RED2),
   ('O2','O2','O2  Obese',RED)]

D={}
for code,sid,lab,col in S:
    d=f'data/{sid}-analysis/'
    spots,_,_,_=SR.load_sample(d+'filtered_feature_bc_matrix.h5',
        d+'tissue_positions_list.csv',d+'scalefactors_json.json')
    lo=json.load(open(d+'scalefactors_json.json'))['tissue_lowres_scalef']
    with h5py.File(d+'filtered_feature_bc_matrix.h5','r') as f:
        m=csc_matrix((f['matrix/data'][:],f['matrix/indices'][:],f['matrix/indptr'][:]),
                     shape=f['matrix/shape'][:])
    sub=m[:,spots['mi'].values]
    D[code]=dict(umi=np.asarray(sub.sum(0)).ravel(),
                 genes=np.asarray((sub>0).sum(0)).ravel(),
                 px=spots['pxl_col_fullres'].values*lo, py=spots['pxl_row_fullres'].values*lo,
                 img=Image.open(d+'tissue_lowres_image.png'), lab=lab, col=col)

fig=plt.figure(figsize=(7.087,7.4))
gs=fig.add_gridspec(4,4,height_ratios=[0.95,0.95,1.25,0.07],hspace=0.50,wspace=0.16,
                    left=0.085,right=0.985,top=0.945,bottom=0.075)

# ---- A/B : violin distributions ---------------------------------------------
for row,(key,label,logy) in enumerate([('umi','Panel UMI per spot',True),
                                       ('genes','Panel genes per spot',True)]):
    ax=fig.add_subplot(gs[row,:])
    data=[D[c]['umi' if key=='umi' else 'genes'] for c,_,_,_ in S]
    parts=ax.violinplot([np.log10(v+1) for v in data],positions=range(4),widths=0.75,
                        showextrema=False,showmedians=False)
    for pc,(c,_,_,col) in zip(parts['bodies'],S):
        pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor('none')
    for i,v in enumerate(data):
        med=np.median(v)
        ax.hlines(np.log10(med+1),i-0.28,i+0.28,color='0.15',lw=1.1,zorder=4)
        ax.text(i,np.log10(v.max()+1)+0.30,f'{med:.0f}',ha='center',
                fontsize=7.5,fontweight='bold',color='0.15')
    ax.set_xticks(range(4))
    ax.set_xticklabels([D[c]['lab'] for c,_,_,_ in S],fontsize=8)
    ax.set_ylabel(f'{label}\n(log$_{{10}}$ scale)',fontsize=8.5,linespacing=1.2)
    ax.set_ylim(-0.15,4.55)
    ticks=[1,10,100,1000,10000]
    ax.set_yticks([np.log10(t+1) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks],fontsize=7.5)
    ax.tick_params(length=2.0); ax.tick_params(axis='x',length=0)
    ax.set_title('Panel UMI per spot' if row==0 else 'Panel genes per spot',
                 fontsize=9.5,fontweight='bold',pad=5)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
    for s_ in ['left','bottom']: ax.spines[s_].set_linewidth(0.6)

# ---- C : spatial maps of UMI depth ------------------------------------------
vmax=max(np.percentile(D[c]['umi'],99) for c,_,_,_ in S)
for j,(code,sid,lab,col) in enumerate(S):
    ax=fig.add_subplot(gs[2,j]); d=D[code]
    ax.imshow(d['img'],alpha=0.35,aspect='equal')
    o=np.argsort(d['umi'])
    sc=ax.scatter(d['px'][o],d['py'][o],c=np.clip(d['umi'][o],1,vmax),cmap=cmap,
                  norm=LogNorm(vmin=10,vmax=vmax),s=1.6,linewidths=0,rasterized=True)
    padx=(d['px'].max()-d['px'].min())*0.04; pady=(d['py'].max()-d['py'].min())*0.04
    ax.set_xlim(d['px'].min()-padx,d['px'].max()+padx)
    ax.set_ylim(d['py'].max()+pady,d['py'].min()-pady)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(lab,fontsize=8,fontweight='bold',color=col,pad=4)
cax=fig.add_subplot(gs[3,1:3]); pos=cax.get_position()
cax.set_position([pos.x0+pos.width*0.15,pos.y0,pos.width*0.70,pos.height])
cb=fig.colorbar(sc,cax=cax,orientation='horizontal')
cb.ax.xaxis.set_major_locator(LogLocator(base=10,numticks=5))
cb.ax.xaxis.set_minor_formatter(NullFormatter())
cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f'{v:.0f}'))
cb.ax.tick_params(labelsize=6.5,length=1.4,pad=0.8,which='major')
cb.ax.tick_params(length=0,which='minor'); cb.outline.set_linewidth(0.3)
cb.set_label('Panel UMI per spot (log scale, shared)',fontsize=7,labelpad=2.5)

for e in ['pdf','png']:
    fig.savefig(f'{OUT}SupplFig1_QC_NEW.{e}',dpi=600,bbox_inches='tight',facecolor='white')
print('done')
for c,_,_,_ in S:
    print(f"  {c}: median UMI {np.median(D[c]['umi']):.0f} | median genes {np.median(D[c]['genes']):.0f}")
