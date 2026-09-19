import pandas as pd, numpy as np, json
from features import load
from style import *
H=load(); H['day']=H.ts.dt.normalize()
D=H.groupby(['id','day']).agg(E=('y',lambda s:s.sum(min_count=1)),cs=('csn','sum')).reset_index()
D['miss']=D.E.isna(); D['k']=D.E/D.cs
fleet=D.groupby('day').agg(nrep=('miss',lambda s:(~s).sum()),nsite=('miss','size'),kmed=('k','median'))
D=D.join(fleet,on='day')
D['others_rep']=D.nrep-(~D.miss).astype(int); D['others']=D.nsite-1
M=D[D.miss].copy()
M['cls']=np.where(M.others_rep<0.5*M.others,'fleet-wide gap (portal/export)',
          np.where(M.kmed<0.05,'fleet-wide low production (snow/dark)','site-specific gap on productive day (candidate outage)'))
cnt=M.cls.value_counts(); print(cnt, len(M))
# run lengths of candidate outages
C=M[M.cls.str.startswith('site')].sort_values(['id','day'])
C['run']=(C.groupby('id').day.diff().dt.days!=1).cumsum()
runs=C.groupby('run').agg(id=('id','first'),start=('day','min'),n=('day','size'))
print('candidate-outage runs',len(runs),'len>=3',(runs.n>=3).sum(),'max',runs.n.max())
print(runs.sort_values('n',ascending=False).head(8))
M['month']=M.day.dt.month
winter=float(100*M.month.isin([11,12,1,2,3]).mean())
out={'n_missing':int(len(M)),'classes':{k:int(v) for k,v in cnt.items()},'candidate_runs':int(len(runs)),'candidate_runs_ge3':int((runs.n>=3).sum()),
     'candidate_max_run':int(runs.n.max()),'missing_in_NovMar_pct':winter,
     'cand_by_site':C.groupby('id').size().to_dict()}
M.to_csv('out/tab/missing_days_audit.csv',index=False)
# capacity proxy stability
H['yr']=H.ts.dt.year
cap=H.groupby('id').cap.first()
py=H[(H.yr>=2017)&(H.yr<=2022)].groupby(['id','yr']).kWh.quantile(0.995).unstack()
ratio=py.div(cap*0+H.groupby('id').apply(lambda g:g[g.ts<'2020-01-01'].kWh.quantile(0.995)),axis=0)
print(ratio.round(2)); ratio.round(3).to_csv('out/tab/capacity_stability.csv')
out['cap_ratio_range_by_site']={int(i):[float(ratio.loc[i].min()),float(ratio.loc[i].max())] for i in ratio.index}
# spatial decorrelation (8 sites with coordinates)
R=pd.read_csv('out/tab/rated_capacity.csv').set_index('id')
def hav(a,b):
    la1,lo1,la2,lo2=map(np.radians,[a.lat,a.lon,b.lat,b.lon]); d=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371*2*np.arcsin(np.sqrt(d))
K=D[(D.kmed>=0.05)].pivot_table(index='day',columns='id',values='k')
Hh=H[(H.elev>10)&H.ts.dt.month.isin([5,6,7,8,9])].copy(); Hh['kh']=Hh.y/Hh.csn
KH=Hh.pivot_table(index='ts',columns='id',values='kh')
rows=[]
ids=[i for i in R.index if i in K.columns]
for i in range(len(ids)):
    for j in range(i+1,len(ids)):
        a,b=ids[i],ids[j]; dist=hav(R.loc[a],R.loc[b])
        rows.append(dict(a=a,b=b,km=dist,r_daily=K[[a,b]].dropna().corr().iloc[0,1],r_hourly_summer=KH[[a,b]].dropna().corr().iloc[0,1]))
S=pd.DataFrame(rows); S.to_csv('out/tab/spatial_pairs.csv',index=False)
sl=np.polyfit(S.km,S.r_daily,1); slh=np.polyfit(S.km,S.r_hourly_summer,1)
print(S.describe().round(3)); print('slope daily per 10km',sl[0]*10,'hourly',slh[0]*10)
out.update({'n_pairs':len(S),'km_range':[float(S.km.min()),float(S.km.max())],'r_daily_range':[float(S.r_daily.min()),float(S.r_daily.max())],
 'r_hourly_range':[float(S.r_hourly_summer.min()),float(S.r_hourly_summer.max())],'slope_daily_per10km':float(sl[0]*10),'slope_hourly_per10km':float(slh[0]*10),
 'r_daily_mean':float(S.r_daily.mean()),'r_hourly_mean':float(S.r_hourly_summer.mean()),
 'corr_dist_rdaily':float(np.corrcoef(S.km,S.r_daily)[0,1]),'corr_dist_rhourly':float(np.corrcoef(S.km,S.r_hourly_summer)[0,1])})
fig,ax=plt.subplots(figsize=(W1,2.2))
ax.scatter(S.km,S.r_daily,s=14,color=PAL[0],label='Daily clear-sky ratio (all assessable days)',zorder=3)
ax.scatter(S.km,S.r_hourly_summer,s=14,color=PAL[1],marker='s',label='Hourly clear-sky index (May–Sep)',zorder=3)
xx=np.linspace(0,S.km.max()*1.05,10)
ax.plot(xx,np.polyval(sl,xx),color=PAL[0],lw=1); ax.plot(xx,np.polyval(slh,xx),color=PAL[1],lw=1)
ax.set_xlabel('Distance between sites (km)'); ax.set_ylabel('Pearson correlation'); ax.legend(frameon=False,fontsize=6.5,loc='lower left'); ax.set_ylim(0,1)
fig.savefig('out/fig/spatial.png')
json.dump(out,open('out/tab/audit.json','w'),indent=1,default=str)
