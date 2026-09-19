import pandas as pd, numpy as np, json, os, sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from daily import daily
P0,CSd=daily()
from features import load as _load
_H=_load(); _H['day']=_H.ts.dt.normalize()
_zf=_H[(_H.elev>10)&(~_H.raw_present)].groupby(['id','day']).size()
_rep=_H[_H.raw_present].groupby(['id','day']).size()
ZF=(_zf.reindex(_rep.index).fillna(0)).rename('zf')
REPS=int(os.environ.get('R',5)); EPS=float(os.environ.get('EPS',0.02)); DAMP=float(os.environ.get('DAMP',0.02))
KMIN=float(os.environ.get('KMIN',0.05)); ZM=float(os.environ.get('ZM',4.0)); SUF=os.environ.get('SUF','')
W=int(os.environ.get('W',60)); LOO=int(os.environ.get('LOO',1)); ZX=int(os.environ.get('ZX',0))
def table(P):
    E=P.sum(axis=1,min_count=24)
    t=pd.DataFrame({'E':E.values},index=P.index); t.index.names=['_id','_day']
    t['id']=t.index.get_level_values(0); t['day']=t.index.get_level_values(1)
    first=t[t.E.notna()].groupby('id').day.min(); last=t[t.E.notna()].groupby('id').day.max()
    t=t[(t.day>=t.id.map(first))&(t.day<=t.id.map(last))]
    t['k']=t.E/t.day.map(CSd)
    g=t.groupby('day'); t['nsite']=g.k.transform('size')
    if LOO:
        K=t.pivot_table(index='day',columns='id',values='k')
        med={c:K.drop(columns=c).median(axis=1) for c in K.columns}; cnt={c:K.drop(columns=c).notna().sum(axis=1) for c in K.columns}
        t['kmed']=[med[i].get(d,np.nan) for i,d in zip(t.id,t.day)]; t['nrep']=[cnt[i].get(d,0) for i,d in zip(t.id,t.day)]
    else:
        t['kmed']=g.k.transform('median'); t['nrep']=g.k.transform('count')
    return t
def robust_z(series, mask_flag, zm=None):
    """causal robust z; if mask_flag, days flagged (z>ZM) are excluded from later baselines"""
    v=series.values; z=np.full(len(v),np.nan); hist=[]
    for i,x in enumerate(v):
        if len(hist)>=20:
            h=np.array(hist[-W:]); med=np.median(h); mad=np.median(np.abs(h-med))
            if not np.isnan(x): z[i]=-(x-med)/(1.4826*mad+DAMP)
        if np.isnan(x): continue
        if mask_flag and not np.isnan(z[i]) and z[i]>zm: continue
        hist.append(x)
    return z
def scores(P,t,iso=None,ZMD=None):
    t=t.copy()
    t['lr']=np.log((t.k+EPS)/(t.kmed+EPS)); t['lk']=np.log(t.k+EPS)
    for col,out,m in [('lr','s_peer',False),('lr','s_peer_mask',True),('lk','s_single',False),('lk','s_single_mask',True)]:
        t[out]=np.nan
        for sid,g in t.groupby('id',sort=False):
            gs=g.sort_values('day'); zm=(ZMD or {}).get(out,1e9)
            t.loc[gs.index,out]=robust_z(gs[col],m,zm)
    # silent-day rule: site silent while fleet productive -> maximal peer score
    silent=t.E.isna()&(t.kmed>=0.05)&(t.nrep>=0.5*(t.nsite-1))&(t.nrep>=5)
    for c in ['s_peer','s_peer_mask']: t.loc[silent,c]=1e3
    t['silent']=silent
    V=P.loc[t.index].fillna(0).values
    F=pd.DataFrame({'E':t.E.values,'peak':V.max(1),'peakhr':V.argmax(1),'nprod':(V>0.01).sum(1),'k':t.k.values,
        'doys':np.sin(2*np.pi*t.day.dt.dayofyear/365.25),'doyc':np.cos(2*np.pi*t.day.dt.dayofyear/365.25)},index=t.index)
    s=pd.Series(np.nan,index=t.index); fitted={}
    for sid in t.id.unique():
        ok=(t.id==sid)&F.notna().all(axis=1)
        m=iso[sid] if iso else IsolationForest(n_estimators=200,random_state=0).fit(F[ok]); fitted[sid]=m
        s[ok]=-m.score_samples(F[ok])
    t['s_iso']=s
    return t,fitted
METH={'s_peer_mask':'Peer index, masked (proposed)','s_peer':'Peer index, unmasked','s_single_mask':'Single-site index, masked','s_single':'Single-site index','s_iso':'Isolation Forest'}
t0=table(P0); T0,iso=scores(P0,t0)
_b=T0[(T0.kmed>=KMIN)&(T0.nrep>=6)&T0.E.notna()]
ZMD={'s_peer_mask':float(_b.s_peer.quantile(0.99)),'s_single_mask':float(_b.s_single.quantile(0.99))}
T0,iso=scores(P0,t0,None,ZMD); print('mask thresholds',ZMD,flush=True)
fleet_ok=(T0.kmed>=KMIN)&(T0.nrep>=6)
T0['zf']=[ZF.get((i,d),np.nan) for i,d in zip(T0.id,T0.day)]
if ZX: fleet_ok=fleet_ok&~(T0.zf>2)
base=T0[fleet_ok&T0.E.notna()]            # evaluable unmodified site-days (site reported)
thr={m:base[m].quantile(0.99) for m in METH}
low=base.kmed<=base.kmed.quantile(0.10)
conf={METH[m]:float(100*low[base[m]>thr[m]].mean()) for m in METH}
real_silent=int((T0.silent&T0.E.isna()).sum())
print('confounded',conf,'real silent-day alarms',real_silent,flush=True)
rng=np.random.default_rng(1); KINDS=['outage','derate50','derate25','shading','gradual','clipping']
cand=base.index.to_list(); res=[]
p99=P0.stack().groupby(level=0).quantile(0.99)
for rep in range(REPS):
    Pi=P0.copy(); lab={}; kind={}; ev={}
    picks=rng.choice(len(cand),240,replace=False)
    for j,pi in enumerate(picks):
        sid,day=cand[pi]; kd=KINDS[j%6]
        L={'outage':int(rng.integers(1,3)),'gradual':30}.get(kd,int(rng.integers(3,11)))
        for n,d in enumerate(pd.date_range(day,periods=L)):
            if (sid,d) not in Pi.index: continue
            row=Pi.loc[(sid,d)].values.astype(float).copy()
            if np.isnan(row).any(): continue
            if kd=='outage': row[:]=np.nan            # realistic: logger writes nothing on a zero day
            elif kd=='derate50': row*=0.5
            elif kd=='derate25': row*=0.75
            elif kd=='shading': row[:14]*=0.35
            elif kd=='gradual': row*=1-0.4*(n+1)/L      # linear decline to 60% over 30 days
            elif kd=='clipping': row=np.minimum(row,0.5*p99[sid])  # loss only at high irradiance
            Pi.loc[(sid,d)]=row; lab[(sid,d)]=1; kind[(sid,d)]=kd; ev[(sid,d)]=j
    ti=table(Pi); Ti,_=scores(Pi,ti,iso,ZMD)
    evalset=Ti[((Ti.kmed>=KMIN)&(Ti.nrep>=5))&(Ti.E.notna()|pd.Series(Ti.index.isin(list(lab)),index=Ti.index))]
    if ZX:
        zz=pd.Series([ZF.get((i,d),0) for i,d in zip(evalset.id,evalset.day)],index=evalset.index); evalset=evalset[~(zz>2)|evalset.index.isin(list(lab))]
    y=pd.Series(0,index=evalset.index); y[y.index.isin(list(lab))]=1
    k=pd.Series(kind).reindex(evalset.index); e=pd.Series(ev).reindex(evalset.index)
    for m in METH:
        sc=evalset[m].fillna(-1e9)
        a=sc>thr[m]; r=dict(rep=rep,method=METH[m],AP=average_precision_score(y,sc),recall=float(a[y==1].mean()),
            false_alarm=float(a[y==0].mean()),precision=float(y[a].mean()) if a.any() else np.nan)
        ed=a[e.notna()].groupby(e[e.notna()]).any(); ek=k[e.notna()].groupby(e[e.notna()]).first()
        r['event_recall']=float(ed.mean())
        # persistence: for detected events longer than 2 days, share of days from first alarm to event end that stay alarmed
        pers=[]
        for evid,grp in a[e.notna()].groupby(e[e.notna()]):
            g2=grp.sort_index(level=1)
            if len(g2)>2 and g2.any():
                first=np.argmax(g2.values); pers.append(g2.values[first:].mean())
        r['persistence']=float(np.mean(pers)) if pers else np.nan
        win=evalset.day.dt.month.isin([11,12,1,2,3])
        for nm,msk in [('AP_winter',win),('AP_summer',~win)]:
            yy=y[msk]; r[nm]=average_precision_score(yy,sc[msk]) if yy.sum()>0 else np.nan
        r['n_pos_winter']=int(y[win].sum()); r['chance_AP']=float(y.mean())
        # causal threshold: 99th pct of this detector's scores on all previous days (>=180 days history)
        es=pd.DataFrame({'day':evalset.day.values,'s':sc.values},index=evalset.index).sort_values('day')
        dd=np.sort(es.day.unique()); thr_c=pd.Series(np.nan,index=dd); vals=[]
        byday=es.groupby('day').s.apply(np.array)
        hist=np.array([])
        for i,dday in enumerate(dd):
            if i>=180: thr_c[dday]=np.quantile(hist,0.99)
            hist=np.concatenate([hist,byday[dday]])
        tc=es.day.map(thr_c); ac=(es.s>tc)&tc.notna(); ac=ac.reindex(evalset.index)
        okc=tc.reindex(evalset.index).notna()
        r['recall_causal']=float(ac[(y==1)&okc].mean()); r['precision_causal']=float(y[ac&okc].mean()) if (ac&okc).any() else np.nan
        r['false_alarm_causal']=float(ac[(y==0)&okc].mean())
        edc=ac[e.notna()&okc].groupby(e[e.notna()&okc]).any(); r['event_recall_causal']=float(edc.mean())
        no=(k!='outage')
        r['AP_nonoutage']=average_precision_score(y[no],sc[no])
        for kd in KINDS: r['ev_'+kd]=float(ed[ek==kd].mean())
        # detection delay (days from event start to first alarm) for multi-day events
        res.append(r)
    print('rep',rep,flush=True)
R=pd.DataFrame(res); S=R.drop(columns='rep').groupby('method').agg(['mean','std'])
S.to_csv(f'out/tab/anomaly3{SUF}.csv'); print(R.groupby('method').mean(numeric_only=True).round(3).to_string())
json.dump({'confounded_alarm_pct':conf,'real_silent_day_alarms':real_silent,'n_eval_base':int(len(base)),
  'pct_fleet_assessable':float(100*fleet_ok.mean()),'eps':EPS,'mask_thresholds':ZMD,'damp':DAMP,'kmin':KMIN,'zmask':ZM},open(f'out/tab/anomaly3_meta{SUF}.json','w'),indent=1)
T0.to_parquet(f'out/daily_scores3{SUF}.parquet')
