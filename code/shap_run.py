from style import *; import numpy as np, pandas as pd, lightgbm as lgb, shap, json
from features import load, make, FEATS_EXCL
H=load(); out={}
NICE={'lag_1':'Output at t−1 h','lag_2':'Output at t−2 h','lag_3':'Output at t−3 h','lag_4':'Output at t−4 h','lag_6':'Output at t−6 h','lag_24':'Output at t−24 h',
 'lag_25':'Output at t−25 h','lag_26':'Output at t−26 h','lag_27':'Output at t−27 h','lag_29':'Output at t−29 h','lag_47':'Output at t−47 h','lag_48':'Output at t−48 h','lag_168':'Output at t−168 h',
 'csn':'Clear-sky irradiance (target hour)','elev':'Solar elevation (target hour)','roll24_mean':'24-h mean output','roll24_max':'24-h max output','roll168_mean':'7-day mean output',
 'csi_origin':'Clear-sky index at origin','csi24':'24-h clear-sky index','hour_sin':'Hour (sin)','hour_cos':'Hour (cos)','doy_sin':'Day of year (sin)','doy_cos':'Day of year (cos)','site':'Site ID'}
fig,axs=plt.subplots(1,2,figsize=(W2,2.4))
for ax,h in zip(axs,[1,24]):
    F=make(H,h); feats=[c for c in F.columns if c not in FEATS_EXCL]
    te=F[(F.ts>='2022-01-01')&F.target.notna()&(F.elev>0)].sample(4000,random_state=0)
    b=lgb.Booster(model_file=f'out/lgb_h{h}.txt')
    X=te[feats].copy()
    sv=b.predict(X,pred_contrib=True)[:,:-1]
    imp=pd.Series(np.abs(sv).mean(0),index=feats).sort_values(ascending=False)
    out[f'h{h}']={k:float(v) for k,v in imp.items()}
    top=imp.head(8)[::-1]
    ax.barh([NICE.get(k,k) for k in top.index],top.values,color=PAL[0],height=0.6)
    ax.set_title(f'{"1-hour" if h==1 else "24-hour"}-ahead model'); ax.set_xlabel('Mean |SHAP value|'); ax.grid(axis='y',visible=False)
fig.tight_layout(); fig.savefig('out/fig/shap.png'); json.dump(out,open('out/tab/shap.json','w'),indent=1)
for k,v in out.items(): print(k,list(v.items())[:6])
