import pandas as pd, numpy as np
from features import load, make
H=load(); out=[]
for h in [1,24]:
    F=make(H,h); F['m']=F.ts.dt.month; F['hr']=F.ts.dt.hour
    for Y in [2020,2021,2022]:
        tr=F[(F.ts<f'{Y}-01-01')&F.target.notna()]
        clim=tr.groupby(['id','m','hr']).target.mean().rename('clim')
        G=F.join(clim,on=['id','m','hr'])
        t=G[(G.ts<f'{Y}-01-01')&G.target.notna()&G.bl_persist.notna()&(G.elev>0)]
        d=t.bl_persist-t.clim; a=float(((t.target-t.clim)*d).sum()/(d*d).sum())
        te=G[(G.ts>=f'{Y}-01-01')&(G.ts<f'{Y+1}-01-01')]
        out.append(pd.DataFrame({'id':te.id,'ts':te.ts,'h':h,'Climatology':te.clim,'CLIPER':(a*te.bl_persist+(1-a)*te.clim).clip(lower=0),'alpha':a}))
        print(h,Y,round(a,3))
pd.concat(out).to_parquet('out/preds_cliper.parquet')
