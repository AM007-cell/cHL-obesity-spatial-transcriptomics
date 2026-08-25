#!/usr/bin/env python3
"""Figure 5 - GO enrichment of the depth-robust DE gene set, with technical-artefact control."""
import numpy as np, pandas as pd, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

U='/mnt/user-data/uploads/'; OUT='/mnt/user-data/outputs/panels/'
DPI=300; RED='#C0392B'; BLUE='#2980B9'; GREY='#7F8C8D'
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42})
cmap=LinearSegmentedColormap.from_list('p',['#F5B7B1','#CD6155','#922B21'])

SHORT={
 'adaptive immune response based on somatic recombination of immune receptors built from immunoglobulin superfamily domains':
     'Adaptive immune response\n(somatic recombination)',
 'peptide antigen assembly with MHC class II protein complex':'Peptide assembly with MHC class II',
 'positive regulation of receptor-mediated endocytosis':'Receptor-mediated endocytosis (pos. reg.)',
 'positive regulation of T cell mediated cytotoxicity':'T cell mediated cytotoxicity (pos. reg.)',
 'positive regulation of cell adhesion':'Cell adhesion (pos. reg.)',
 'cellular response to epidermal growth factor stimulus':'Response to EGF stimulus',
}
def short(s):
    s=SHORT.get(s,s)
    return s[0].upper()+s[1:]

# ---------------- A : lollipop -------------------------------------------------
d=pd.read_csv(U+'enrichment_results_simplified.csv').sort_values('p.adjust').head(14).copy()
d['lab']=d.Description.map(short)
d=d.iloc[::-1]
fig,ax=plt.subplots(figsize=(5.4,4.0))
y=np.arange(len(d))
nl=-np.log10(d['p.adjust'])
ax.hlines(y,0,d.FoldEnrichment,color='#D5DBDB',lw=1.0,zorder=1)
sc=ax.scatter(d.FoldEnrichment,y,s=d.Count*7.5,c=nl,cmap=cmap,zorder=2,
              linewidths=0.3,edgecolors='white')
ax.set_yticks(y); ax.set_yticklabels(d['lab'],fontsize=6.3,linespacing=1.1)
ax.set_xscale('log')
ax.set_xticks([5,10,20,50,100]); ax.set_xticklabels(['5','10','20','50','100'],fontsize=6.5)
ax.set_xlabel('Fold enrichment',fontsize=7.5)
ax.tick_params(labelsize=6.5,length=2.2)
ax.set_xlim(4,190)
cb=plt.colorbar(sc,ax=ax,fraction=0.028,pad=0.02,shrink=0.55)
cb.set_label('$-$log$_{10}$ $p_{adj}$',fontsize=6.2); cb.ax.tick_params(labelsize=5.4,length=1.5)
cb.outline.set_linewidth(0.3)
for n in [4,8,13]:
    ax.scatter([],[],s=n*7.5,c='#CD6155',label=f'{n}',linewidths=0.3,edgecolors='white')
ax.legend(title='Genes',fontsize=5.8,title_fontsize=6,frameon=False,loc='lower right',
          labelspacing=0.75,borderpad=0.3,handletextpad=0.6)
ax.set_title('GO biological process — depth-robust gene set',fontsize=8.5,fontweight='bold',pad=7)
ax.text(0.5,-0.135,'43 genes enriched ≥2× in obese vs. both non-obese samples at matched depth',
        transform=ax.transAxes,fontsize=5.6,color='0.45',style='italic',ha='center')
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_linewidth(0.6)
for e in ['pdf','png']:
    fig.savefig(f'{OUT}Fig5A_GO_enrichment.{e}',dpi=DPI,bbox_inches='tight',facecolor='white')
plt.close(fig); print(' -> Fig5A_GO_enrichment')

# ---------------- B : artefact control ----------------------------------------
rob=pd.read_csv(U+'enrichment_results.csv'); allg=pd.read_csv(U+'enrichment_results_ALL.csv')
TECH=re.compile(r'translat|ribosom|rRNA|RNA splic|spliceosom|mRNA process|peptide biosynth|protein folding',re.I)
cats=['Immune / antigen presentation','Translation / RNA processing','Other']
def cat(s):
    if TECH.search(s): return cats[1]
    if re.search(r'immun|antigen|lymphocyte|leukocyte|B cell|T cell|MHC|cytotox|complement|chemotax',s,re.I):
        return cats[0]
    return cats[2]
tab=[]
for name,df_ in [('Depth-robust\n(43 genes)',rob),('Full DE list\n(128 genes)',allg)]:
    c=df_.Description.map(cat).value_counts()
    tab.append([c.get(k,0) for k in cats])
tab=np.array(tab,dtype=float)
frac=tab/tab.sum(1,keepdims=True)*100

fig,ax=plt.subplots(figsize=(3.6,2.9))
xs=np.arange(2); bottom=np.zeros(2)
for j,(k,col) in enumerate(zip(cats,[BLUE,RED,'#BDC3C7'])):
    ax.bar(xs,frac[:,j],0.55,bottom=bottom,color=col,edgecolor='white',lw=0.6,label=k)
    for i in range(2):
        if frac[i,j]>4:
            ax.text(i,bottom[i]+frac[i,j]/2,f'{int(tab[i,j])}',ha='center',va='center',
                    fontsize=6.2,color='white',fontweight='bold')
    bottom+=frac[:,j]
ax.set_xticks(xs); ax.set_xticklabels(['Depth-robust\n(43 genes)','Full DE list\n(128 genes)'],
                                      fontsize=6.5,linespacing=1.3)
ax.set_ylabel('% of significant GO terms',fontsize=7.5)
ax.set_ylim(0,100); ax.tick_params(labelsize=6.5,length=2.2); ax.tick_params(axis='x',length=0)
ax.legend(fontsize=5.8,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.20),ncol=1,
          handlelength=1.2,handleheight=0.9,labelspacing=0.3)
ax.set_title('Effect of filtering on enriched terms',fontsize=8.5,fontweight='bold',pad=4)
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_linewidth(0.6)
for e in ['pdf','png']:
    fig.savefig(f'{OUT}Fig5B_enrichment_QC.{e}',dpi=DPI,bbox_inches='tight',facecolor='white')
plt.close(fig); print(' -> Fig5B_enrichment_QC')
print('term counts (robust, full):'); print(pd.DataFrame(tab,columns=cats,index=['robust','full']))
