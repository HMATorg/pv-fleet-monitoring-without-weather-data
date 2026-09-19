import pandas as pd, numpy as np, lightgbm as lgb, json, time, sys
from sklearn.ensemble import HistGradientBoostingRegressor
from features import load, make, FEATS_EXCL
H=load()
FOLDS=[2020,2021,2022]
res=[]; preds_all=[]
for h in [1,24]:
    F=make(H,h)
    feats=[c for c in F.columns if c not in FEATS_EXCL]
    for Y in FOLDS:
        tr=F[(F.ts<f'{Y}-01-01')&F.target.notna()&(F.elev>-5)]
        te=F[(F.ts>=f'{Y}-01-01')&(F.ts<f'{Y+1}-01-01')&F.target.notna()&(F.elev>0)]
        te=te.dropna(subset=['bl_smart','bl_persist','bl_seasonal'])
        P=te[['id','ts','target','cap','elev']].copy()
        P['Persistence']=te.bl_persist.values; P['Seasonal naive']=te.bl_seasonal.values; P['Smart persistence']=te.bl_smart.values
        t0=time.time()
        m=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,min_child_samples=50,random_state=0,verbose=-1)
        m.fit(tr[feats],tr.target,categorical_feature=['site'])
        P['LightGBM']=np.clip(m.predict(te[feats]),0,None)
        X=tr[feats].copy(); X['site']=X.site.cat.codes; Xt=te[feats].copy(); Xt['site']=Xt.site.cat.codes
        hg=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.06,max_depth=8,random_state=0,categorical_features=[feats.index('site')])
        hg.fit(X,tr.target); P['HistGB']=np.clip(hg.predict(Xt),0,None)
        P['h']=h; P['fold']=Y
        preds_all.append(P)
        if Y==2022: m.booster_.save_model(f'out/lgb_h{h}.txt')
        print(h,Y,len(tr),len(te),round(time.time()-t0),flush=True)
pd.concat(preds_all).to_parquet('out/preds_tree.parquet')
