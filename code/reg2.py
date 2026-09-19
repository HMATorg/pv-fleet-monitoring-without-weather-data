import pandas as pd, numpy as np, json, torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import diptest
from daily import daily
P,CSd=daily(); P=P[P.notna().all(axis=1)]
emb=pd.read_parquet('out/emb.parquet'); X=P.loc[emb.index].values; Z=emb[[f'z{i}' for i in range(4)]].values
Zp=PCA(4,random_state=0).fit_transform(X); pca=PCA(4,random_state=0).fit(X)
rng=np.random.default_rng(0); sub=rng.choice(len(X),8000,replace=False)
res={}
# (1) silhouette in the common 24-D fingerprint space
for k in (3,4,5,6):
    lp=KMeans(k,n_init=10,random_state=0).fit_predict(Zp); la=KMeans(k,n_init=10,random_state=0).fit_predict(Z)
    res[f'sil24_k{k}']={'pca':float(silhouette_score(X[sub],lp[sub])),'ae':float(silhouette_score(X[sub],la[sub]))}
print(res,flush=True)
# (2) multimodality: dip test on PC1 and on daily energy
E=X.sum(1)
d1,p1=diptest.diptest(Zp[:,0]); dE,pE=diptest.diptest(E)
res['dip_pc1']=[float(d1),float(p1)]; res['dip_energy']=[float(dE),float(pE)]
res['evr']=pca.explained_variance_ratio_.tolist()
print('dip',res['dip_pc1'],res['dip_energy'],'evr',res['evr'],flush=True)
# (3) gap statistic vs uniform null in PCA space (Tibshirani), k=1..8
def Wk(D,k):
    km=KMeans(k,n_init=5,random_state=0).fit(D); return np.log(km.inertia_)
D=Zp[sub]; lo,hi=D.min(0),D.max(0); B=10
gap={};sk={}
for k in range(1,9):
    w=Wk(D,k); wb=np.array([Wk(rng.uniform(lo,hi,size=D.shape),k) for _ in range(B)])
    gap[k]=float(wb.mean()-w); sk[k]=float(wb.std()*np.sqrt(1+1/B))
kopt=[k for k in range(1,8) if gap[k]>=gap[k+1]-sk[k+1]]
res['gap']=gap; res['gap_sk']=sk; res['gap_kopt']=kopt[0] if kopt else None
print('gap',{k:round(v,3) for k,v in gap.items()},'kopt',res['gap_kopt'],flush=True)
# (4) are regimes just energy bins?
lab=KMeans(5,n_init=20,random_state=0).fit_predict(Zp)
eb=pd.qcut(E,5,labels=False); ek=KMeans(5,n_init=20,random_state=0).fit_predict(E.reshape(-1,1))
res['ari_regime_vs_energy_quintile']=float(adjusted_rand_score(lab,eb)); res['ari_regime_vs_energy_kmeans']=float(adjusted_rand_score(lab,ek))
shape=X/np.maximum(E[:,None],1e-6); ls=KMeans(5,n_init=20,random_state=0).fit_predict(PCA(4,random_state=0).fit_transform(shape[E>0.5]))
res['pc1_corr_energy']=float(np.corrcoef(Zp[:,0],E)[0,1])
print('ARI vs energy',res['ari_regime_vs_energy_quintile'],res['ari_regime_vs_energy_kmeans'],'pc1~E r',res['pc1_corr_energy'],flush=True)
# (5) synchrony with seasonal null: permute labels among site-days within same month-year
days=emb.index.get_level_values(1); ids=emb.index.get_level_values(0)
D2=pd.DataFrame({'day':days,'lab':lab,'my':days.to_period('M')})
def sync(df):
    n=df.groupby('day').size(); g=df.groupby('day').lab.agg(lambda s:s.value_counts(normalize=True).iloc[0])
    return float(100*(g[n>=6]>=0.7).mean())
obs=sync(D2); null=[]
for b in range(100):
    D3=D2.copy(); D3['lab']=D3.groupby('my').lab.transform(lambda s:rng.permutation(s.values)); null.append(sync(D3))
res['sync_obs']=obs; res['sync_null_mean']=float(np.mean(null)); res['sync_null_p95']=float(np.percentile(null,95)); res['sync_null_max']=float(np.max(null))
print('sync obs',obs,'null',np.mean(null),np.percentile(null,95),flush=True)
json.dump(res,open('out/tab/regime_tests.json','w'),indent=1)
