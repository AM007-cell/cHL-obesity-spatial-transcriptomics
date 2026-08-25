#!/usr/bin/env python3
"""Figure 4A - gene selection rationale: enrichment vs prevalence across all DE genes."""
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import spatial_relationship as SR

OUT='/mnt/user-data/outputs/panels/'
DPI=300; RED='#C0392B'
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42})
rng=np.random.default_rng(0)
XL='/mnt/user-data/uploads/markers_obese_vs_control.xlsx'
df=pd.concat([pd.read_excel(XL,str(i)) for i in range(1,9)],ignore_index=True)

FAM=[('Matrix / Wnt','#8E44AD',['SFRP4','SFRP2','LUM','FBLN1','C7','C3','PCOLCE','CLU','TFPI2']),
     ('Plasma cell / Ig','#6A4C93',None),
     ('Antigen presentation','#1B998B',['HLA-C','HLA-E','HLA-DRA','HLA-DPA1','HLA-DPB1','B2M','CD74']),
     ('Metabolic / UPR','#E07A5F',['TXNIP','XBP1','DERL3','SSR4','UCP2','FBP1']),
     ('Immediate-early','#B58900',['FOS','FOSB','JUN','JUNB','EGR1','ZFP36L1','ZFP36L2','TSC22D3','BTG2']),
     ('Chemokine','#2980B9',['CCL19','CCL21','CCL22','CXCL13','CXCR4'])]

D={}
for code,sid in [('C1','C1'),('C2','C2'),('O2','O2')]:
    d=f'data/{sid}-analysis/'
    spots,_,_,_=SR.load_sample(d+'filtered_feature_bc_matrix.h5',
        d+'tissue_positions_list.csv',d+'scalefactors_json.json')
    with h5py.File(d+'filtered_feature_bc_matrix.h5','r') as f:
        g=f['matrix/features/name'][:].astype(str)
        m=csc_matrix((f['matrix/data'][:],f['matrix/indices'][:],f['matrix/indptr'][:]),
                     shape=f['matrix/shape'][:])
    m=m[:,spots['mi'].values].tocsc(); u=np.asarray(m.sum(0)).ravel()
    p=np.minimum(1.0,410/np.maximum(u,1)); co=m.tocoo()
    D[code]=({k:i for i,k in enumerate(g)},
             csc_matrix((rng.binomial(co.data.astype(int),p[co.col]),(co.row,co.col)),shape=m.shape))

def fam_of(g):
    if pd.Series([g]).str.match(r'^(IG[HKL]|JCHAIN|MZB1)')[0]: return 'Plasma cell / Ig'
    for name,_,gl in FAM:
        if gl and g in gl: return name
    return 'Other'
COL={n:c for n,c,_ in FAM}; COL['Other']='#BDC3C7'

rows=[]
for gn in sorted(df.gene.unique()):
    v={}
    if any(gn not in D[c][0] for c in D): continue
    for c in D:
        i,m=D[c]; v[c]=100*(np.asarray(m[i[gn]].todense()).ravel()>0).mean()
    if v['O2']<3: continue
    hi=max(v['C1'],v['C2'],0.15)
    rows.append((gn,v['O2'],v['O2']/hi,fam_of(gn)))
r=pd.DataFrame(rows,columns=['gene','pct_O2','ratio','fam'])

fig,ax=plt.subplots(figsize=(4.5,3.6))
ax.axvspan(3,20,color='#F4F6F7',zorder=0)
ax.axvline(3,color='0.65',lw=0.7,ls='--',zorder=1)
for name in ['Other']+[n for n,_,_ in FAM]:
    s=r[r.fam==name]
    if not len(s): continue
    ax.scatter(s['ratio'],s['pct_O2'],s=16,color=COL[name],
               label=None if name=='Other' else name,zorder=2,linewidths=0,
               alpha=0.55 if name=='Other' else 0.95)
LAB=['SFRP4','TXNIP','FBLN1','LUM','C7','CCL21','EGR1','C3','SFRP2','HLA-C','IGKC','IGHG1','FOS','UCP2']
for _,q in r[r.gene.isin(LAB)].iterrows():
    ax.annotate(q['gene'],(q['ratio'],q['pct_O2']),xytext=(3.5,2.5),textcoords='offset points',
                fontsize=5.2,style='italic',color=COL[q['fam']])
sf=r[r.gene=='SFRP4'].iloc[0]
ax.scatter([sf['ratio']],[sf['pct_O2']],s=110,facecolor='none',edgecolor=RED,lw=1.1,zorder=3)
ax.set_xscale('log')
ax.set_xticks([0.5,1,2,5,10]); ax.set_xticklabels(['0.5×','1×','2×','5×','10×'],fontsize=6.5)
ax.set_xlabel('Enrichment in obese at matched depth\n(O2 ÷ higher of the two non-obese)',
              fontsize=7.5,linespacing=1.25)
ax.set_ylabel('% spots expressing in O2',fontsize=7.5)
ax.tick_params(labelsize=6.5,length=2.2)
ax.legend(fontsize=5.6,frameon=False,loc='upper left',handletextpad=0.3,
          labelspacing=0.3,borderaxespad=0.3)
ax.set_title('Selection of SFRP4: enrichment and prevalence',fontsize=8.5,fontweight='bold',pad=4)
ax.text(0.985,0.03,'grey band: enriched > 3×',transform=ax.transAxes,ha='right',
        fontsize=5.4,color='0.5',style='italic')
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_linewidth(0.6)
for e in ['pdf','png']:
    fig.savefig(f'{OUT}Fig4A_SFRP4_selection.{e}',dpi=DPI,bbox_inches='tight',facecolor='white')
print('rank of SFRP4 by ratio:',(r.ratio>sf['ratio']).sum()+1,'of',len(r))
print('genes with ratio>3 and pct_O2>30%:',r[(r.ratio>3)&(r.pct_O2>30)].gene.tolist())
