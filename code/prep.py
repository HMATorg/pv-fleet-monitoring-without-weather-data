import pandas as pd, numpy as np, pvlib, json
LAT,LON=51.05,-114.07
df=pd.read_csv('data.csv')
t=pd.to_datetime(df['date'],format='%Y/%m/%d %I:%M:%S %p')
df['ts']=t.dt.round('h'); df['offhour']=(t.dt.minute!=0).astype(int)
# some sites carry a duplicated feed stamped at :15/:30/:45 alongside the :00 feed -> keep the :00 record
n0=len(df); df=df.sort_values(['id','ts','offhour']).drop_duplicates(['id','ts'],keep='first')
print('dropped duplicate off-hour records:',n0-len(df))
df=df[df.id!=594148]  # Telus Spark: 317 rows only
meta=df.groupby('id').agg(name=('name','first'),inst=('installationDate','first')).reset_index()
h=df[['id','ts','kWh']]
frames=[]
for sid,g in h.groupby('id'):
    d0=g.ts.min().normalize()+pd.Timedelta(days=1); d1=g.ts.max().normalize()
    idx=pd.date_range(d0,d1+pd.Timedelta(hours=23),freq='h')
    s=g.set_index('ts').kWh.reindex(idx)
    day=idx.normalize()
    has=pd.Series(s.notna().values,index=idx).groupby(day).transform('any').values
    # logger reports only non-zero hours: missing hour within a reporting day => 0; whole-day absent => gap
    v=np.where(s.notna(),s.values,np.where(has,0.0,np.nan))
    frames.append(pd.DataFrame({'id':sid,'ts':idx,'kWh':v,'raw_present':s.notna().values}))
H=pd.concat(frames,ignore_index=True)
# clear-sky potential (Haurwitz, zenith only), hour-centred, local time America/Edmonton
t=pd.date_range(H.ts.min(),H.ts.max(),freq='h')
tl=(t+pd.Timedelta(minutes=30)).tz_localize('America/Edmonton',ambiguous='NaT',nonexistent='shift_forward')
ok=~tl.isna()
sp=pvlib.solarposition.get_solarposition(tl[ok],LAT,LON)
cs=pvlib.clearsky.haurwitz(sp['apparent_zenith'])
C=pd.DataFrame({'ts':t[ok],'elev':sp['apparent_elevation'].values,'ghi_cs':cs['ghi'].values})
H=H.merge(C,on='ts',how='left')
H.to_parquet('hourly.parquet')
meta.to_csv('out/tab/sites_meta.csv',index=False)
# stats
st=[]
for sid,g in H.groupby('id'):
    dd=g.groupby(g.ts.dt.normalize()).kWh.sum(min_count=1)
    st.append(dict(id=sid,name=meta.set_index('id').name[sid],start=g.ts.min().date(),end=g.ts.max().date(),
      days=len(dd),gap_days=int(dd.isna().sum()),gap_pct=round(100*dd.isna().mean(),1),
      total_MWh=round(g.kWh.sum()/1000,1),mean_daily_kWh=round(dd.mean(),1),
      p95_daily_kWh=round(dd.quantile(.95),1),peak_hour_kWh=round(g.kWh.max(),1)))
st=pd.DataFrame(st).sort_values('total_MWh',ascending=False)
st.to_csv('out/tab/site_stats.csv',index=False)
print(st.to_string()); print(H.shape, H.kWh.isna().mean(), (H.raw_present).sum())
print('night-with-production (elev<-5 & kWh>0):',((H.elev<-5)&(H.kWh>0)).mean(), H[(H.elev<-5)].kWh.mean())
