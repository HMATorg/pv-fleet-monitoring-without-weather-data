import pandas as pd, numpy as np, json, os
P=pd.read_parquet('out/preds_tree.parquet')
if os.path.exists('out/preds_lstm.parquet'):
    L=pd.read_parquet('out/preds_lstm.parquet'); L['ts']=pd.to_datetime(L.ts)
    P=P.merge(L[['id','ts','h','LSTM']],on=['id','ts','h'],how='inner')
C=pd.read_parquet('out/preds_cliper.parquet'); P=P.merge(C[['id','ts','h','Climatology','CLIPER']],on=['id','ts','h'],how='left')
models=[m for m in ['Climatology','CLIPER','Persistence','Seasonal naive','Smart persistence','HistGB','LightGBM','LSTM'] if m in P]
rows=[]
for h,g in P.groupby('h'):
    ref=np.sqrt(((g['CLIPER']-g.target)**2).mean())
    for m in models:
        e=g[m]-g.target
        rows.append(dict(h=h,model=m,nMAE=100*e.abs().mean(),nRMSE=100*np.sqrt((e**2).mean()),MBE=100*e.mean(),
            skill=100*(1-np.sqrt((e**2).mean())/ref),n=len(g)))
T=pd.DataFrame(rows); print(T.round(2).to_string()); T.to_csv('out/tab/forecast_overall.csv',index=False)
# per fold
rows=[]
for (h,f),g in P.groupby(['h','fold']):
    for m in models:
        e=g[m]-g.target; rows.append(dict(h=h,fold=f,model=m,nRMSE=100*np.sqrt((e**2).mean()),nMAE=100*e.abs().mean()))
pd.DataFrame(rows).to_csv('out/tab/forecast_folds.csv',index=False)
print(pd.DataFrame(rows).pivot_table(index=['h','model'],columns='fold',values='nRMSE').round(2))
# block bootstrap (day blocks) CI of skill for best models vs smart persistence
rng=np.random.default_rng(0); ci={}
for h,g in P.groupby('h'):
    g=g.assign(day=g.ts.dt.normalize()); days=g.day.unique()
    se={m:(g[m]-g.target)**2 for m in models}
    agg={m:se[m].groupby(g.day).agg(['sum','count']) for m in models}
    for m in models:
        if m=='CLIPER': continue
        bs=[]
        for b in range(500):
            d=rng.choice(days,len(days),replace=True)
            a=agg[m].loc[d].sum(); r=agg['CLIPER'].loc[d].sum()
            bs.append(100*(1-np.sqrt(a['sum']/a['count'])/np.sqrt(r['sum']/r['count'])))
        ci[f'h{h}_{m}']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
json.dump(ci,open('out/tab/skill_ci.json','w'),indent=1); print(ci)
# per-site best model
best=T.loc[T[T.model.isin(['LightGBM','HistGB','LSTM'])].groupby('h').nRMSE.idxmin()]
rows=[]
for (h,sid),g in P.groupby(['h','id']):
    for m in ['CLIPER','HistGB','LightGBM']+(['LSTM'] if 'LSTM' in P else []):
        e=g[m]-g.target; rows.append(dict(h=h,id=sid,model=m,nRMSE=100*np.sqrt((e**2).mean())))
S=pd.DataFrame(rows).pivot_table(index=['id'],columns=['h','model'],values='nRMSE'); S.round(2).to_csv('out/tab/forecast_sites.csv'); print(S.round(2))
