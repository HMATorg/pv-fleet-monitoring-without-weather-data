import pandas as pd, numpy as np, json
from features import load, make
H=load(); F=make(H,1)
te=F[(F.ts>='2020-01-01')&F.target.notna()&(F.elev>0)].copy()
te['o']=F.loc[te.index,'lag_1']
te['cso']=H.set_index(['id','ts']).csn.groupby(level=0).shift(1).reindex(pd.MultiIndex.from_arrays([te.id,te.ts])).values
res={}
def ev(p,m): e=(p-te.target)[m]; return 100*np.sqrt((e**2).mean())
m=te.o.notna()
res['persistence']=ev(te.o,m)
res['smart_24h_used_in_paper']=ev(te.bl_smart,m&te.bl_smart.notna())
for floor in [0.0,0.05,0.1,0.2]:
    csi=np.where(te.cso>floor, te.o/np.maximum(te.cso,1e-3), np.nan)
    p=np.where(np.isnan(csi), te.o, csi*te.csn)   # fallback to persistence below floor
    res[f'smart_hourly_floor{floor}']=ev(pd.Series(p,index=te.index),m)
    res[f'smart_hourly_floor{floor}_clip']=ev(pd.Series(np.where(np.isnan(csi),te.o,np.clip(csi,0,1.5)*te.csn),index=te.index),m)
print({k:round(v,2) for k,v in res.items()}); json.dump(res,open('out/tab/smartpers_diag.json','w'),indent=1)
