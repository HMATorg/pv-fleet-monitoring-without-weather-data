import pandas as pd, numpy as np, json
from sklearn.ensemble import HistGradientBoostingRegressor
from features import load
H=load(); rows=[]
for sid,g in H.groupby('id',sort=False):
    g=g.set_index('ts').sort_index(); y=g.y; cs=g.csn; el=g.elev
    issues=g.index[(g.index.hour==11)]
    for it in issues:
        D1=it.normalize()+pd.Timedelta(days=1)
        tg=pd.date_range(D1,periods=24,freq='h')
        o=y.get(it,np.nan)
        if np.isnan(o): continue
        w24=y.loc[it-pd.Timedelta(hours=23):it]; c24=cs.loc[it-pd.Timedelta(hours=23):it]
        base=dict(id=sid,issue=it,o=o,l1=y.get(it-pd.Timedelta(hours=1),np.nan),l2=y.get(it-pd.Timedelta(hours=2),np.nan),
                  l24=y.get(it-pd.Timedelta(hours=24),np.nan),m24=w24.mean(),x24=w24.max(),csi24=w24.sum()/max(c24.sum(),1e-3),
                  m168=y.loc[it-pd.Timedelta(hours=167):it].mean())
        for t in tg:
            if t not in y.index: continue
            hh=t.hour
            pers_t=(t-pd.Timedelta(days=1)) if hh<=11 else (t-pd.Timedelta(days=2))
            r=dict(base); r.update(ts=t,target=y[t],csn=cs[t],elev=el[t],lead=(t-it)/pd.Timedelta(hours=1),hour=hh,doy=t.dayofyear,pers=y.get(pers_t,np.nan))
            rows.append(r)
F=pd.DataFrame(rows); F['site']=F.id.astype('category').cat.codes
F['hs']=np.sin(2*np.pi*F.hour/24); F['hc']=np.cos(2*np.pi*F.hour/24); F['ds']=np.sin(2*np.pi*F.doy/365.25); F['dc']=np.cos(2*np.pi*F.doy/365.25)
feats=['o','l1','l2','l24','m24','x24','csi24','m168','csn','elev','lead','hs','hc','ds','dc','site']
out=[]
for Y in [2020,2021,2022]:
    tr=F[(F.ts<f'{Y}-01-01')&F.target.notna()&(F.elev>-5)]; te=F[(F.ts>=f'{Y}-01-01')&(F.ts<f'{Y+1}-01-01')&F.target.notna()&(F.elev>0)&F.pers.notna()].copy()
    tr=tr.assign(m=tr.ts.dt.month); te['m']=te.ts.dt.month
    clim=tr.groupby(['id','m','hour']).target.mean().rename('clim')
    a=tr.join(clim,on=['id','m','hour']).dropna(subset=['clim','pers']); a=a[a.elev>0]
    d=a.pers-a.clim; al=float(((a.target-a.clim)*d).sum()/(d*d).sum())
    te=te.join(clim,on=['id','m','hour']); te['CLIPER']=(al*te.pers+(1-al)*te.clim).clip(lower=0)
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.06,max_depth=8,random_state=0,categorical_features=[feats.index('site')]).fit(tr[feats],tr.target)
    te['HGB']=np.clip(m.predict(te[feats]),0,None); te['fold']=Y; te['alpha']=al; out.append(te[['id','ts','target','pers','clim','CLIPER','HGB','fold','alpha','lead']])
    print(Y,al,len(te),flush=True)
O=pd.concat(out).dropna(subset=['clim'])
def rm(p): return 100*np.sqrt(((p-O.target)**2).mean())
res={'nRMSE':{k:rm(O[k]) for k in ['pers','clim','CLIPER','HGB']},'n':len(O)}
res['skill_HGB']=100*(1-rm(O.HGB)/rm(O.CLIPER))
rng=np.random.default_rng(0); O['day']=O.ts.dt.normalize(); days=O.day.unique()
a=((O.HGB-O.target)**2).groupby(O.day).agg(['sum','count']); r=((O.CLIPER-O.target)**2).groupby(O.day).agg(['sum','count']); bs=[]
for b in range(500):
    dd=rng.choice(days,len(days)); A=a.loc[dd].sum(); R=r.loc[dd].sum(); bs.append(100*(1-np.sqrt(A['sum']/A['count'])/np.sqrt(R['sum']/R['count'])))
res['ci']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
print(res); json.dump(res,open('out/tab/dayahead_issue.json','w'),indent=1)
