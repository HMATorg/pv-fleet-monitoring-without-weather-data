from style import *; import numpy as np, pandas as pd, json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from daily import daily
P,CSd=daily(); P=P[P.notna().all(axis=1)]
emb=pd.read_parquet('out/emb.parquet'); Z=emb[[f'z{i}' for i in range(4)]].values; X=P.loc[emb.index].values
K=5
Zp=PCA(4,random_state=0).fit_transform(X)
lab=KMeans(K,n_init=20,random_state=0).fit_predict(Zp)      # primary: PCA
la=KMeans(K,n_init=20,random_state=0).fit_predict(Z)        # comparison: autoencoder
ari=adjusted_rand_score(lab,la)
syncs={}
for kk in (4,5,6):
    l2=KMeans(kk,n_init=20,random_state=0).fit_predict(Zp); D2=pd.DataFrame({'day':emb.index.get_level_values(1),'lab':l2})
    g2=D2.groupby('day').lab.agg(lambda s:s.value_counts(normalize=True).iloc[0]); n2=D2.groupby('day').size()
    syncs[kk]=float(100*(g2[n2>=6]>=0.7).mean())
print('sync by k',syncs)
E=X.sum(1); order=np.argsort([E[lab==c].mean() for c in range(K)])[::-1]
remap={c:i for i,c in enumerate(order)}; lab=np.array([remap[c] for c in lab])
days=emb.index.get_level_values(1); ids=emb.index.get_level_values(0)
kidx=E/pd.Series(days).map(CSd).values
info=[]
for c in range(K):
    m=lab==c; mon=pd.Series(days[m].month).value_counts(normalize=True)
    prof=X[m].mean(0); shape=prof/prof.sum()
    info.append(dict(regime=c+1,n=int(m.sum()),share=float(100*m.mean()),mean_daily_eqh=float(E[m].mean()),
        mean_clear_sky_ratio=float(np.nanmean(kidx[m])),peak_hour=int(prof.argmax()),
        summer_share=float(100*np.isin(days[m].month,[5,6,7,8]).mean()),winter_share=float(100*np.isin(days[m].month,[11,12,1,2]).mean()),
        top_months=[int(x) for x in mon.index[:3]], productive_hours=float((X[m]>0.02).sum(1).mean())))
I=pd.DataFrame(info); print(I.round(2).to_string()); I.to_csv('out/tab/regimes.csv',index=False)
# fleet synchrony: share of days on which >=70% of reporting sites share the modal regime
D=pd.DataFrame({'day':days,'lab':lab})
g=D.groupby('day').lab.agg(lambda s:s.value_counts(normalize=True).iloc[0]); n=D.groupby('day').size()
sync=float(100*(g[n>=6]>=0.7).mean())
json.dump({'primary':'PCA','sync_by_k':syncs,'ari_ae_vs_pca_k5':float(ari),'fleet_sync_pct':sync,'K':K},open('out/tab/regimes.json','w'),indent=1); print('ari',ari,'sync',sync)
names=[f'R{c+1}' for c in range(K)]
fig,axs=plt.subplots(1,2,figsize=(W2,2.1),gridspec_kw={'width_ratios':[1,1.15]})
for c in range(K):
    m=lab==c; mu=X[m].mean(0); q1,q9=np.quantile(X[m],[.1,.9],axis=0)
    axs[0].fill_between(range(24),q1,q9,color=PAL[c],alpha=0.12,lw=0); axs[0].plot(range(24),mu,color=PAL[c],lw=1.5,label=f'{names[c]} ({100*m.mean():.0f}%)')
axs[0].set_xlabel('Hour of day'); axs[0].set_ylabel('Normalised output'); axs[0].legend(frameon=False,fontsize=6.5); axs[0].set_xticks(range(0,24,4))
ct=pd.crosstab(days.month,lab,normalize='index')*100
bottom=np.zeros(12)
for c in range(K):
    axs[1].bar(range(1,13),ct[c].values,bottom=bottom,color=PAL[c],width=0.85,edgecolor='white',linewidth=0.8,label=names[c]); bottom+=ct[c].values
axs[1].set_xticks(range(1,13)); axs[1].set_xticklabels(list('JFMAMJJASOND')); axs[1].set_ylabel('Share of site-days (%)'); axs[1].set_ylim(0,100); axs[1].grid(axis='x',visible=False)
fig.tight_layout(); fig.savefig('out/fig/regimes.png')
# latent scatter
from sklearn.decomposition import PCA as P2
z2=Zp[:,:2]; rng=np.random.default_rng(0); s=rng.choice(len(z2),6000,replace=False)
fig,ax=plt.subplots(figsize=(W1,2.3))
for c in range(K): m=lab[s]==c; ax.scatter(z2[s][m,0],z2[s][m,1],s=3,color=PAL[c],alpha=0.6,lw=0,label=names[c])
ax.set_xlabel('Principal component 1'); ax.set_ylabel('Principal component 2'); ax.legend(frameon=False,markerscale=3,fontsize=6.5,ncol=5,loc='upper center',bbox_to_anchor=(0.5,1.12))
fig.savefig('out/fig/latent.png')
pd.Series(lab,index=emb.index,name='regime').to_frame().to_parquet('out/regime_labels.parquet')
