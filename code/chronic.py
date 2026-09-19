from style import *; import pandas as pd, numpy as np, json
from features import load
H=load(); H['day']=H.ts.dt.normalize()
D=H.groupby(['id','day']).agg(E=('y',lambda s:s.sum(min_count=24)),cs=('csn','sum')).reset_index()
D['k']=D.E/D.cs
K=D.pivot_table(index='day',columns='id',values='k')
kmed=K.median(axis=1); n=K.notna().sum(axis=1)
ok=(kmed>=0.05)&(n>=6)&K.index.month.isin([4,5,6,7,8,9,10])
R=np.log((K[ok]+0.02).div(kmed[ok]+0.02,axis=0))   # daily peer log-ratio
ref=R[(R.index>='2018-01-01')&(R.index<'2020-01-01')].median()
rel=np.exp(R-ref)                                     # peer ratio relative to 2018-19 reference
roll=rel.rolling('90D',min_periods=30).median().reindex(pd.date_range(rel.index.min(),rel.index.max()))
yr=rel[(rel.index>='2018-01-01')&(rel.index<'2023-01-01')].groupby(rel.index[(rel.index>='2018-01-01')&(rel.index<'2023-01-01')].year).median()
print(yr.round(2).T)
names=pd.read_csv('out/tab/sites_meta.csv').set_index('id').name
flag={int(s):[str(d.date()) for d in roll[s][roll[s]<0.85].index[[0]]] if (roll[s]<0.85).any() else [] for s in roll.columns}
share={int(s):float(100*(roll[s]<0.85).mean()) for s in roll.columns}
print('first sustained <0.85:',flag); print('share of days below 0.85',{k:round(v,1) for k,v in share.items()})
json.dump({'yearly_rel':{int(s):{int(y):float(v) for y,v in yr[s].items()} for s in yr.columns},'first_below_085':flag,'share_below_085':share},open('out/tab/chronic.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(W2,2.2))
hl={355827:PAL[0],577650:PAL[1],570079:PAL[2]}
for s in roll.columns:
    if s in hl: continue
    ax.plot(roll.index,roll[s],color='#c9c8c3',lw=0.8)
for s,c in hl.items(): ax.plot(roll.index,roll[s],color=c,lw=1.5,label=names[s])
ax.axhline(0.85,color='#8a8985',ls='--',lw=0.8); ax.text(roll.index[40],0.86,'15% loss',fontsize=6.5,color='#52514e')
ax.set_ylabel('90-day median peer ratio\n(relative to 2018–19)'); ax.set_ylim(0.3,1.3); ax.legend(frameon=False,fontsize=6.5,loc='lower left',ncol=3)
fig.savefig('out/fig/chronic.png')
