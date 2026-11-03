"""Render v3 figures from frozen records only. No training or model inference.

Run from the repository root: python report/build_figures.py
Requires numpy/matplotlib. --results can point at an archived results directory.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, default=HERE.parent / 'results')
args = parser.parse_args()
R = args.results
F = HERE / 'figures'
F.mkdir(exist_ok=True)
C = ['#245B83', '#BD6335', '#408572', '#8C649B', '#AD8A2D']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.labelsize':10, 'axes.titlesize':10, 'legend.fontsize':9,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':'#65717B', 'axes.labelcolor':'#26343F',
    'xtick.color':'#49545E', 'ytick.color':'#49545E',
    'axes.axisbelow':True, 'grid.color':'#DFE4E8', 'grid.linewidth':.6,
    'lines.linewidth':1.7, 'pdf.fonttype':42, 'savefig.pad_inches':.05})

def save(fig, name):
    fig.savefig(F / (name+'.pdf'), bbox_inches='tight')
    fig.savefig(F / (name+'.png'), bbox_inches='tight', dpi=180)
    plt.close(fig)

def h(group, name):
    with (R/group/name/'history.csv').open() as f:
        rows = list(csv.DictReader(f))
    return {k:np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in rows[0]}

def summaries(group, variant):
    return [json.loads(p.read_text()) for p in sorted((R/group).glob(variant+'__s*/summary.json'))]

def accuracy_axes(ax):
    ax.grid(axis='y'); ax.set(xlabel='Epoch', ylabel='Accuracy (%)', xlim=(1,30))

# 5.1: compact, factorized legend.
fig, ax = plt.subplots(figsize=(6.25,2.8),layout='constrained')
labels = ['Baseline','FC dropout 0.5','BatchNorm']
for i,(g,n) in enumerate([('e0_baseline','vgg-none__s0'),('e1_dropout','fc0.5__s0'),('e2_norm','bn__s0')]):
    a=h(g,n)
    for key,ls in [('train_eval_acc','-'),('val_acc','--')]:
        ax.plot(a['epoch'],a[key]*100,color=C[i],ls=ls)
accuracy_axes(ax)
ax.set_ylim(38,101)
handles=[Line2D([],[],color=C[i],label=s) for i,s in enumerate(labels)]
handles += [Line2D([],[],color='#52616B',ls=ls,label=s) for ls,s in [('-','Train subset'),('--','Validation')]]
ax.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
save(fig,'baseline_curves')

# 5.2: actual seed points with mean and sample SD, training values stay in table.
fig, ax=plt.subplots(figsize=(6.25,2.45),layout='constrained')
variants=[('e0_baseline','vgg-none','None'),('e1_dropout','fc0.3','FC 0.3'),('e1_dropout','fc0.5','FC 0.5'),('e1_dropout','fc0.7','FC 0.7'),('e1_dropout','conv0.1-fc0.5','Conv 0.1 + FC 0.5'),('e1_dropout','conv0.2-fc0.5','Conv 0.2 + FC 0.5')]
for y,(g,n,l) in enumerate(variants):
    a=np.array([x['final_val_acc']*100 for x in summaries(g,n)])
    ax.scatter(a,y+np.linspace(-.12,.12,len(a)),s=22,color='#A5B8C6',zorder=3)
    ax.errorbar(a.mean(),y,xerr=a.std(ddof=1),fmt='D',color=C[0],capsize=3,ms=4,zorder=4)
ax.set_yticks(range(6),[x[2] for x in variants]);ax.invert_yaxis()
ax.set(xlabel='Validation accuracy (%)',xlim=(82.4,86.5));ax.grid(axis='x')
ax.legend(handles=[Line2D([],[],color='#A5B8C6',marker='o',ls='',label='3 seeds'),Line2D([],[],color=C[0],marker='D',label='Mean ± sample SD')],loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
save(fig,'dropout_gap')

fig, ax=plt.subplots(figsize=(6.25,2.65),layout='constrained')
for i,(g,n,l) in enumerate([('e0_baseline','vgg-none','None')]+[('e2_norm',k,k.upper()) for k in ['bn','ln','in','gn']]):
    a=h(g,n+'__s0');ax.plot(a['epoch'],100*a['val_acc'],color=C[i],label=l,ls=['-','-','--',':','-.'][i])
accuracy_axes(ax);ax.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
save(fig,'norm_curves')

fig,ax=plt.subplots(figsize=(6.25,2.5),layout='constrained')
lrs=[.005,.02,.1,.4]
for i,n in enumerate(['none','bn','gn']):
    vals=[]
    for lr in lrs:
        g,v=('e0_baseline','vgg-none') if n=='none' else ('e2_norm',n)
        s=summaries(g,v)[0] if lr==.02 else summaries('e2_lr',f'{n}-lr{lr}')[0]
        vals.append(np.nan if s['diverged'] else s['final_val_acc']*100)
    ax.plot(lrs,vals,'o-',color=C[i],label='None' if n=='none' else n.upper(),ms=5)
ax.set_xscale('log');ax.set_xticks(lrs,[str(x) for x in lrs]);ax.minorticks_off()
ax.set(xlabel='Peak learning rate',ylabel='Validation accuracy (%)',ylim=(77,91));ax.grid(axis='y')
ax.text(.4,78.2,'None: diverged\nNaN in epoch 1',ha='right',va='bottom',color=C[0],fontsize=9,
        bbox=dict(facecolor='white',edgecolor='#B8C6D0',boxstyle='round,pad=.4'))
ax.legend(loc='upper left',frameon=False,ncol=3)
save(fig,'norm_lr')

A=json.loads((R/'analysis/analysis.json').read_text())
# 6.1: changes, rather than duplicate absolute accuracy table.
b=list(A['b1']['runs'].values()); vals=np.array([x['val_acc']*100 for x in b]);delta=np.diff(vals)
fig,ax=plt.subplots(figsize=(6.25,2.35),layout='constrained')
names=['+ Crop / flip','+ Cutout','+ Label smoothing','+ Width ×2','Replace with ResNet-56']
for i,(v,n) in enumerate(zip(delta,names)):
    ax.plot([0,v],[i,i],color=C[0] if i<4 else C[1],lw=2)
    ax.scatter(v,i,color=C[0] if i<4 else C[1],s=36,zorder=3)
    ax.text(v+(.13 if v>=0 else -.13),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=9)
ax.axvline(0,color='#65717B',lw=.8);ax.axhline(3.5,color='#BAC4CC',ls='--',lw=.8)
ax.set_yticks(range(5),names);ax.invert_yaxis();ax.set(xlabel='Change in validation accuracy (pp)',xlim=(-3.3,3.9));ax.grid(axis='x')
save(fig,'b1_ablation')

fig,axes=plt.subplots(1,2,figsize=(6.25,2.5),sharey=True,layout='constrained')
for ax,kind,title in zip(axes,['plain','res'],['(a) Plain','(b) ResNet']):
    for i,d in enumerate([20,32,56]):
        a=h('b2_depth',f'{kind}{d}__s0');ax.plot(a['epoch'],100*(1-a['train_eval_acc']),color=C[i],label=f'{d} layers',ls=['-','--','-.'][i])
    ax.set(title=title,xlabel='Epoch',xlim=(1,30));ax.grid(axis='y')
axes[0].set_ylabel('Training subset error (%)')
fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='outside upper center',frameon=False)
save(fig,'depth_train_error')

# 6.3: actual within-bin confidence; empty bins omitted, count shown below.
panels=[('b1-2-cutout__s0','bins','(a) Cutout'),('b1-2-cutout__s0','bins_ts','(b) Cutout + TS'),('b1-3-ls__s0','bins','(c) Label smoothing')]
fig,axes=plt.subplots(2,3,figsize=(6.25,3.15),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
for j,(name,key,title) in enumerate(panels):
    s=A['calibration'][name];bins=s[key]
    active=[x for x in bins if x['count']>0]
    top,bot=axes[:,j]
    top.plot([0,1],[0,1],color='#77838C',ls='--',lw=1)
    top.plot([x['conf'] for x in active],[x['acc'] for x in active],'-o',color=C[0],ms=3,lw=1.2)
    top.set(title=title,xlim=(0,1),ylim=(0,1.03));top.grid(axis='y')
    ece=s['ece_ts'] if key=='bins_ts' else s['ece']
    top.text(.05,.9,f'ECE {ece*100:.2f}%',fontsize=9,transform=top.transAxes)
    bot.bar([(x['lo']+x['hi'])/2 for x in bins],[x['count']/sum(t['count'] for t in bins)*100 for x in bins],width=1/15*.85,color='#A5B8C6')
    bot.set(xlabel='Confidence',ylim=(0,100),xticks=[0,.5,1],yticks=[0,100]);bot.grid(axis='y')
    if j: top.tick_params(labelleft=False);bot.tick_params(labelleft=False)
axes[0,0].set_ylabel('Accuracy');axes[1,0].set_ylabel('Samples (%)')
save(fig,'reliability')

fig,ax=plt.subplots(figsize=(6.25,2.65),layout='constrained')
for i,(k,s) in enumerate(A['shift'].items()):
    p,aug=k.split('-');label=('MaxPool' if p=='max' else 'MaxBlurPool')+' / '+('no aug.' if aug=='none' else 'crop + flip')
    pairs=sorted((int(k),v*100) for k,v in s['acc_by_magnitude'].items())
    ax.plot(*zip(*pairs),marker='o',color=C[i],label=label,ls='--' if p=='blur' else '-',ms=4)
ax.set(xlabel='Shift radius (pixels, Chebyshev)',ylabel='Test accuracy (%)',xticks=range(5));ax.grid(axis='y')
ax.legend(loc='lower left',frameon=False,ncol=2)
save(fig,'shift_accuracy')

# The architecture uses a separate, implementation-aligned vector source.
import runpy
runpy.run_path(str(HERE / 'draw_architecture.py'), run_name='__main__')
print('Rendered 8 result figures plus the architecture from frozen records.')
