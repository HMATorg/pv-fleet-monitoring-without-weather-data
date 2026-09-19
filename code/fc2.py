import pandas as pd, numpy as np, json, time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import load, make, FEATS_EXCL
H=load(); out={'curve':[], 'ridge':{}}
def fit_hgb(tr,feats):
    X=tr[feats].copy(); X['site']=X.site.cat.codes
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.06,max_depth=8,random_state=0,categorical_features=[feats.index('site')]); m.fit(X,tr.target); return m
def pred_hgb(m,te,feats):
    X=te[feats].copy(); X['site']=X.site.cat.codes; return np.clip(m.predict(X),0,None)
def cliper(tr,te):
    tr=tr.assign(m=tr.ts.dt.month,hr=tr.ts.dt.hour); te=te.assign(m=te.ts.dt.month,hr=te.ts.dt.hour)
    clim=tr.groupby(['id','m','hr']).target.mean().rename('clim')
    a_=tr.join(clim,on=['id','m','hr']); a_=a_[a_.elev>0].dropna(subset=['clim','o'])
    d=a_.o-a_.clim; al=float(((a_.target-a_.clim)*d).sum()/(d*d).sum())
    t2=te.join(clim,on=['id','m','hr'])
    return (al*t2.o+(1-al)*t2.clim).clip(lower=0).values, t2.clim.values, al
rows=[]; allp=[]
for h in [1,2,3,4,6,9,12,18,24]:
    F=make(H,h); F['o']=F[f'lag_{h}']
    feats=[c for c in F.columns if c not in FEATS_EXCL|{'o'}]
    for Y in [2020,2021,2022]:
        tr=F[(F.ts<f'{Y}-01-01')&F.target.notna()&(F.elev>-5)]
        te=F[(F.ts>=f'{Y}-01-01')&(F.ts<f'{Y+1}-01-01')&F.target.notna()&(F.elev>0)].dropna(subset=['o'])
        cl,clim,al=cliper(tr,te)
        p=pred_hgb(fit_hgb(tr,feats),te,feats)
        P=te[['id','ts','target','elev']].copy(); P['CLIPER']=cl; P['Climatology']=clim; P['Persistence']=te.o.values; P['HGB']=p; P['h']=h; P['fold']=Y; P['alpha']=al
        # hourly smart persistence with elevation floor (diagnosis of M13)
        cso=te.csn.values  # clear-sky at target
        cs_origin=H.set_index(['id','ts']).csn
        if h in (1,24):
            Xr=tr[feats].drop(columns=['site']).copy(); Xt=te[feats].drop(columns=['site']).copy()
            med=Xr.median(); Xr=Xr.fillna(med); Xt=Xt.fillna(med)
            Xr=pd.concat([Xr,pd.get_dummies(tr.site,prefix='s').astype(float)],axis=1); Xt=pd.concat([Xt,pd.get_dummies(te.site,prefix='s').astype(float)],axis=1)
            sc=StandardScaler().fit(Xr); rg=Ridge(alpha=1.0).fit(sc.transform(Xr),tr.target)
            P['Ridge']=np.clip(rg.predict(sc.transform(Xt)),0,None)
        allp.append(P); print(h,Y,round(al,3),flush=True)
A=pd.concat(allp); A.to_parquet('out/preds_curve.parquet')
