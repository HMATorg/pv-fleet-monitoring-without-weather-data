from style import *; import numpy as np, pandas as pd
from features import load
H=load()
def acf(x,L):
    x=x-np.nanmean(x); d=np.nansum(x*x); return np.array([np.nansum(x[:len(x)-k]*x[k:])/d for k in range(L+1)])
raw=pd.read_csv('data.csv'); raw=raw[raw.id==314106]
raw['t']=pd.to_datetime(raw['date'],format='%Y/%m/%d %I:%M:%S %p'); r0=acf(raw.sort_values('t').kWh.values,72)
g=H[H.id==314106].y.values; r1=acf(g,72)
fig,ax=plt.subplots(figsize=(W1,1.9))
ax.plot(range(73),r0,color=PAL[1],marker='o',ms=2,lw=1,label='Rows as logged (night hours absent)')
ax.plot(range(73),r1,color=PAL[0],marker='o',ms=2,lw=1,label='Regular hourly grid (this study)')
for k in (24,48,72): ax.axvline(k,color='#8a8985',lw=0.6,ls='--')
ax.set_xlabel('Lag'); ax.set_ylabel('Autocorrelation'); ax.legend(frameon=False,loc='upper right'); ax.set_xticks([0,12,24,36,48,60,72])
fig.savefig('out/fig/acf_fix.png'); print('acf lag24 raw/fixed',r0[24].round(3),r1[24].round(3),'argmax raw 10-30',10+np.argmax(r0[10:30]))
json_out={'acf24_raw':float(r0[24]),'acf24_grid':float(r1[24]),'acf_peak_raw_lag':int(10+np.argmax(r0[10:30])),'acf168_grid':float(acf(g,168)[168])}
# month x hour heatmap (fleet mean normalised output)
H['m']=H.ts.dt.month; H['hr']=H.ts.dt.hour
M=H.pivot_table(index='m',columns='hr',values='y',aggfunc='mean')
fig,ax=plt.subplots(figsize=(W1,2.2))
im=ax.imshow(M.values,aspect='auto',cmap=SEQ,origin='upper'); ax.grid(False)
ax.set_yticks(range(12)); ax.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_xticks(range(0,24,3)); ax.set_xlabel('Hour of day (local time)')
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label('Mean normalised output',fontsize=7); cb.outline.set_visible(False)
fig.savefig('out/fig/month_hour.png')
# daily normalised energy per site: monthly medians, coverage
D=H.groupby(['id',H.ts.dt.normalize()]).y.sum(min_count=1).rename('e').reset_index()
D['mon']=D.ts.dt.to_period('M')
Mo=D.groupby('mon').e.agg(['median',lambda s:s.quantile(.1),lambda s:s.quantile(.9)]); Mo.columns=['med','p10','p90']
x=Mo.index.to_timestamp()
fig,ax=plt.subplots(figsize=(W2,1.8))
ax.fill_between(x,Mo.p10,Mo.p90,color=PAL[0],alpha=0.2,lw=0,label='10th–90th percentile (all sites)')
ax.plot(x,Mo.med,color=PAL[0],lw=1.4,label='Median')
ax.set_ylabel('Daily energy / capacity proxy (h)'); ax.legend(frameon=False,ncol=2,loc='upper left')
fig.savefig('out/fig/daily_fleet.png')
# summer vs winter summary numbers
H['season']=np.where(H.m.isin([6,7]),'JunJul',np.where(H.m.isin([12,1]),'DecJan','o'))
d2=D.merge(H[['id','ts']].assign(ts=H.ts.dt.normalize()).drop_duplicates(),on=['id','ts'])
D['m']=D.ts.dt.month
json_out['summer_winter_ratio']=float(D[D.m.isin([6,7])].e.median()/D[D.m.isin([12,1])].e.median())
json_out['median_daily_eqh_jun_jul']=float(D[D.m.isin([6,7])].e.median()); json_out['median_daily_eqh_dec_jan']=float(D[D.m.isin([12,1])].e.median())
json_out['zero_days_dec_jan_pct']=float(100*(D[D.m.isin([12,1])].e<0.05).mean())
import json; json.dump(json_out,open('out/tab/eda.json','w'),indent=1); print(json_out)
