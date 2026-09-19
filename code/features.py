import pandas as pd, numpy as np
def load():
    H=pd.read_parquet('hourly.parquet').sort_values(['id','ts']).reset_index(drop=True)
    cap=H[H.ts<'2020-01-01'].groupby('id').kWh.quantile(0.995).rename('cap')
    H=H.merge(cap,on='id'); H['y']=H.kWh/H.cap
    H['csn']=H.ghi_cs/1000.0
    return H
def make(H,h):
    out=[]
    for sid,g in H.groupby('id',sort=False):
        g=g.copy(); y=g.y
        f=pd.DataFrame({'id':sid,'ts':g.ts,'target':y,'elev':g.elev,'csn':g.csn,'cap':g.cap})
        for k in [0,1,2,3,5,23]:
            f[f'lag_{h+k}']=y.shift(h+k)
        for L in [24,48,168]:
            if L>h and f'lag_{L}' not in f: f[f'lag_{L}']=y.shift(L)
        o=y.shift(h)
        f['roll24_mean']=o.rolling(24,min_periods=18).mean()
        f['roll24_max']=o.rolling(24,min_periods=18).max()
        f['roll168_mean']=o.rolling(168,min_periods=120).mean()
        cs_o=g.csn.shift(h)
        f['csi_origin']=np.where(cs_o>0.05,o/np.maximum(cs_o,1e-3),np.nan)
        # smart persistence (clear-sky index persistence) baseline
        # use daily-mean ratio over the 24h window at origin to be robust at night origins
        num=o.rolling(24,min_periods=18).sum(); den=cs_o.rolling(24,min_periods=18).sum()
        f['csi24']=num/den
        f['bl_smart']=(f['csi24']*g.csn).clip(lower=0)
        f['bl_persist']=o if h==1 else y.shift(24)   # h=1: last value; h=24: same hour yesterday
        f['bl_seasonal']=y.shift(24) if h<=24 else y.shift(h)
        hr=g.ts.dt.hour; doy=g.ts.dt.dayofyear
        f['hour_sin']=np.sin(2*np.pi*hr/24); f['hour_cos']=np.cos(2*np.pi*hr/24)
        f['doy_sin']=np.sin(2*np.pi*doy/365.25); f['doy_cos']=np.cos(2*np.pi*doy/365.25)
        out.append(f)
    F=pd.concat(out,ignore_index=True)
    F['site']=F.id.astype('category')
    return F
FEATS_EXCL={'id','ts','target','cap','bl_smart','bl_persist','bl_seasonal'}
