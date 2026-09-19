import pandas as pd, numpy as np, torch, json
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from daily import daily
torch.manual_seed(0); np.random.seed(0)
P,CSd=daily()
P=P[P.notna().all(1)]
X=P.values.astype(np.float32); ids=P.index.get_level_values(0); days=P.index.get_level_values(1)
print('complete site-days',len(X))
class AE(nn.Module):
    def __init__(s,z=4):
        super().__init__(); s.e=nn.Sequential(nn.Linear(24,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,z))
        s.d=nn.Sequential(nn.Linear(z,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,24))
    def forward(s,x): z=s.e(x); return s.d(z),z
def train_ae(X,epochs=40,seed=0):
    torch.manual_seed(seed); m=AE(); opt=torch.optim.Adam(m.parameters(),1e-3); Xt=torch.from_numpy(X)
    for ep in range(epochs):
        perm=torch.randperm(len(Xt))
        for b in range(0,len(Xt),256):
            i=perm[b:b+256]; opt.zero_grad(); r,_=m(Xt[i]); l=((r-Xt[i])**2).mean(); l.backward(); opt.step()
    m.eval(); return m
if __name__=='__main__':
    m=train_ae(X); torch.save(m.state_dict(),'out/ae.pt')
    with torch.no_grad(): R,Z=m(torch.from_numpy(X)); R=R.numpy(); Z=Z.numpy()
    rec=((R-X)**2).mean(1)
    pca=PCA(4,random_state=0).fit(X); Zp=pca.transform(X)
    res={'n_sitedays':int(len(X)),'ae_recon_mse_mean':float(rec.mean()),'pca4_recon_mse':float(((pca.inverse_transform(Zp)-X)**2).mean(1).mean()),'pca4_evr':float(pca.explained_variance_ratio_.sum())}
    rng=np.random.default_rng(0); sub=rng.choice(len(X),8000,replace=False)
    sil={}
    for k in range(3,11):
        sil[k]={'ae':float(silhouette_score(Z[sub],KMeans(k,n_init=10,random_state=0).fit_predict(Z)[sub])),
                'pca':float(silhouette_score(Zp[sub],KMeans(k,n_init=10,random_state=0).fit_predict(Zp)[sub]))}
    res['silhouette']=sil
    print(sil)
    # stability via bootstrap ARI for k=5..7
    stab={}
    for k in [4,5,6,7]:
        for name,E in [('ae',Z),('pca',Zp)]:
            base=KMeans(k,n_init=10,random_state=0).fit(E)
            aris=[]
            for s in range(10):
                bi=rng.choice(len(E),len(E),replace=True)
                km=KMeans(k,n_init=5,random_state=s).fit(E[bi]); aris.append(adjusted_rand_score(base.labels_,km.predict(E)))
            stab[f'{name}_k{k}']=(float(np.mean(aris)),float(np.std(aris)))
    res['stability_ari']=stab; print(stab)
    json.dump(res,open('out/tab/unsup.json','w'),indent=1)
    pd.DataFrame(Z,columns=[f'z{i}' for i in range(4)],index=P.index).assign(rec=rec).to_parquet('out/emb.parquet')
