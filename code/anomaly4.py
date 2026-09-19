"""Detector benchmark, third-round revision.

Changes vs anomaly3.py, all prompted by the second-round review:
 (1) Real silent days are no longer dropped from the evaluation set. Two sets are
     scored: UB = reporting days + injected fault days (previous behaviour, silent
     days treated as unlabelled); LB = the same plus real silent days as negatives.
     AP and day-level precision are reported for both.
 (2) The peer-count threshold is harmonised at NREP (6) for the calibration set,
     the evaluation set and the silent-day rule.
 (3) CALW>0 switches the robust-z baseline from "last W retained observations" to
     "retained observations within the last CALW calendar days", so alarm masking
     no longer stretches the baseline backwards over the seasonal cycle.
 (4) Operational alarm statistics on the unmodified data are recorded, so the real
     alarm rate can be quoted next to the 1% design budget.
 (5) Positives per repetition are recorded so the chance level is auditable.
"""
import pandas as pd, numpy as np, json, os
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
KMIN=float(os.environ.get('KMIN',0.05)); SUF=os.environ.get('SUF','')
W=int(os.environ.get('W',60)); LOO=int(os.environ.get('LOO',1)); ZX=int(os.environ.get('ZX',0))
CALW=int(os.environ.get('CALW',0))
NREP=int(os.environ.get('NREP',6))
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
def robust_z(series, days, mask_flag, zm=None):
    v=series.values; dd=days.values.astype('datetime64[D]').astype(int)
    z=np.full(len(v),np.nan); hv=[]; hd=[]
    for i,x in enumerate(v):
        if len(hv)>=20:
            if CALW:
                a=np.array(hd); sel=a>=dd[i]-CALW; h=np.array(hv)[sel]
                if len(h)<20: h=np.array(hv[-W:])
            else:
                h=np.array(hv[-W:])
            med=np.median(h); mad=np.median(np.abs(h-med))
            if not np.isnan(x): z[i]=-(x-med)/(1.4826*mad+DAMP)
        if np.isnan(x): continue
        if mask_flag and not np.isnan(z[i]) and z[i]>zm: continue
        hv.append(x); hd.append(dd[i])
    return z
def scores(P,t,iso=None,ZMD=None):
    t=t.copy()
    t['lr']=np.log((t.k+EPS)/(t.kmed+EPS)); t['lk']=np.log(t.k+EPS)
    for col,out,m in [('lr','s_peer',False),('lr','s_peer_mask',True),('lk','s_single',False),('lk','s_single_mask',True)]:
        t[out]=np.nan
        for sid,g in t.groupby('id',sort=False):
            gs=g.sort_values('day'); zm=(ZMD or {}).get(out,1e9)
            t.loc[gs.index,out]=robust_z(gs[col],gs['day'],m,zm)
    silent=t.E.isna()&(t.kmed>=KMIN)&(t.nrep>=0.5*(t.nsite-1))&(t.nrep>=NREP)
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
_b=T0[(T0.kmed>=KMIN)&(T0.nrep>=NREP)&T0.E.notna()]
ZMD={'s_peer_mask':float(_b.s_peer.quantile(0.99)),'s_single_mask':float(_b.s_single.quantile(0.99))}
T0,iso=scores(P0,t0,None,ZMD); print('mask thresholds',ZMD,flush=True)
fleet_ok=(T0.kmed>=KMIN)&(T0.nrep>=NREP)
T0['zf']=[ZF.get((i,d),np.nan) for i,d in zip(T0.id,T0.day)]
if ZX: fleet_ok=fleet_ok&~(T0.zf>2)
base=T0[fleet_ok&T0.E.notna()]
thr={m:base[m].quantile(0.99) for m in METH}
low=base.kmed<=base.kmed.quantile(0.10)
conf={METH[m]:float(100*low[base[m]>thr[m]].mean()) for m in METH}
real_silent=int((T0.silent&T0.E.isna()).sum())
op={}
for m in METH:
    n_thr=int((base[m]>thr[m]).sum()); n_sil=real_silent if m.startswith('s_peer') else 0
    op[METH[m]]={'threshold_alarm_days':n_thr,'silent_alarm_days':n_sil,'alarm_days':n_thr+n_sil,
                 'denominator':int(len(base))+n_sil,'alarm_rate_pct':100*(n_thr+n_sil)/(len(base)+n_sil)}
print('confounded',conf,'real silent-day alarms',real_silent,flush=True)
print('operational',json.dumps(op[METH['s_peer_mask']]),flush=True)
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
            if kd=='outage': row[:]=np.nan
            elif kd=='derate50': row*=0.5
            elif kd=='derate25': row*=0.75
            elif kd=='shading': row[:14]*=0.35
            elif kd=='gradual': row*=1-0.4*(n+1)/L
            elif kd=='clipping': row=np.minimum(row,0.5*p99[sid])
            Pi.loc[(sid,d)]=row; lab[(sid,d)]=1; kind[(sid,d)]=kd; ev[(sid,d)]=j
    ti=table(Pi); Ti,_=scores(Pi,ti,iso,ZMD)
    inlab=pd.Series(Ti.index.isin(list(lab)),index=Ti.index)
    fleet=(Ti.kmed>=KMIN)&(Ti.nrep>=NREP)
    sets={'ub':Ti[fleet&(Ti.E.notna()|inlab)],'lb':Ti[fleet&(Ti.E.notna()|inlab|Ti.silent)]}
    if ZX:
        for nm in list(sets):
            S=sets[nm]; zz=pd.Series([ZF.get((i,d),0) for i,d in zip(S.id,S.day)],index=S.index)
            sets[nm]=S[~(zz>2)|S.index.isin(list(lab))]
    for m in METH:
        r=dict(rep=rep,method=METH[m])
        for nm,S in sets.items():
            y=pd.Series(0,index=S.index); y[y.index.isin(list(lab))]=1
            sc=S[m].fillna(-1e9); a=sc>thr[m]
            r['AP_'+nm]=average_precision_score(y,sc)
            r['precision_'+nm]=float(y[a].mean()) if a.any() else np.nan
            r['alarm_rate_'+nm]=float(a.mean()); r['chance_'+nm]=float(y.mean())
            r['n_pos_'+nm]=int(y.sum()); r['n_'+nm]=int(len(y))
            # AP excluding outage-like days on BOTH sides: injected outages and real silent days.
            # This is the quantity the silent-day filter cannot touch.
            no=(pd.Series(kind).reindex(S.index)!='outage')&(~S.silent)
            r['AP_nonoutage_'+nm]=average_precision_score(y[no],sc[no])
        S=sets['ub']; y=pd.Series(0,index=S.index); y[y.index.isin(list(lab))]=1
        sc=S[m].fillna(-1e9); a=sc>thr[m]
        k=pd.Series(kind).reindex(S.index); e=pd.Series(ev).reindex(S.index)
        r['recall']=float(a[y==1].mean()); r['false_alarm']=float(a[y==0].mean())
        ed=a[e.notna()].groupby(e[e.notna()]).any(); ek=k[e.notna()].groupby(e[e.notna()]).first()
        r['event_recall']=float(ed.mean())
        pers=[]
        for evid,grp in a[e.notna()].groupby(e[e.notna()]):
            g2=grp.sort_index(level=1)
            if len(g2)>2 and g2.any():
                first=np.argmax(g2.values); pers.append(g2.values[first:].mean())
        r['persistence']=float(np.mean(pers)) if pers else np.nan
        win=S.day.dt.month.isin([11,12,1,2,3])
        for nm2,msk in [('AP_winter',win),('AP_summer',~win)]:
            yy=y[msk]; r[nm2]=average_precision_score(yy,sc[msk]) if yy.sum()>0 else np.nan
        for kd in KINDS: r['ev_'+kd]=float(ed[ek==kd].mean())
        # causal threshold: 99th pct of this detector's scores on all previous days (>=180 days history)
        for nm,Sc in sets.items():
            yc=pd.Series(0,index=Sc.index); yc[yc.index.isin(list(lab))]=1
            scc=Sc[m].fillna(-1e9)
            es=pd.DataFrame({'day':Sc.day.values,'s':scc.values},index=Sc.index).sort_values('day')
            dd=np.sort(es.day.unique()); thr_c=pd.Series(np.nan,index=dd)
            byday=es.groupby('day').s.apply(np.array); hist=np.array([])
            for i,dday in enumerate(dd):
                if i>=180: thr_c[dday]=np.quantile(hist,0.99)
                hist=np.concatenate([hist,byday[dday]])
            tc=es.day.map(thr_c); ac=(es.s>tc)&tc.notna(); ac=ac.reindex(Sc.index); okc=tc.reindex(Sc.index).notna()
            r['recall_causal_'+nm]=float(ac[(yc==1)&okc].mean())
            r['precision_causal_'+nm]=float(yc[ac&okc].mean()) if (ac&okc).any() else np.nan
            ec=pd.Series(ev).reindex(Sc.index)
            edc=ac[ec.notna()&okc].groupby(ec[ec.notna()&okc]).any(); r['event_recall_causal_'+nm]=float(edc.mean())
        res.append(r)
    print('rep',rep,flush=True)
R=pd.DataFrame(res); S=R.drop(columns='rep').groupby('method').agg(['mean','std'])
S.to_csv(f'out/tab/anomaly4{SUF}.csv')
print(R.groupby('method').mean(numeric_only=True).round(3).to_string())
json.dump({'confounded_alarm_pct':conf,'real_silent_day_alarms':real_silent,'n_eval_base':int(len(base)),
  'operational':op,'pct_fleet_assessable':float(100*fleet_ok.mean()),'eps':EPS,'mask_thresholds':ZMD,
  'damp':DAMP,'kmin':KMIN,'nrep':NREP,'calw':CALW,'W':W,'loo':LOO},open(f'out/tab/anomaly4_meta{SUF}.json','w'),indent=1)
T0.to_parquet(f'out/daily_scores4{SUF}.parquet')
