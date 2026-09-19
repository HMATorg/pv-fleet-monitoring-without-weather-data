from style import *; import numpy as np, pandas as pd, matplotlib.dates as mdates
T=pd.read_parquet('out/daily_scores2.parquet')
b=T[(T.kmed>=0.05)&(T.nrep>=6)&T.E.notna()]
thr=b.s_peer_mask.quantile(.99)
def case(ax,sid,a,z,title):
    s=T[(T.id==sid)&(T.day>=a)&(T.day<=z)].sort_values('day')
    ax.plot(s.day,s.kmed,color='#8a8985',lw=1.2,label='Fleet median')
    ax.plot(s.day,s.k,color=PAL[0],lw=1.4,label='Site')
    f=s[(s.s_peer_mask>thr)&s.E.notna()]; g=s[(s.s_peer_mask>thr)&s.E.isna()]
    ax.scatter(f.day,f.k,color=PAL[7],s=14,zorder=3,label='Flagged (masked peer index)',edgecolor='white',linewidth=0.6)
    ax.scatter(g.day,np.zeros(len(g)),color=PAL[7],marker='x',s=16,zorder=3,label='Flagged silent day (no records)')
    ax.set_title(title); ax.set_ylabel('Daily clear-sky ratio')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
fig,axs=plt.subplots(1,2,figsize=(W2,2.1),sharey=True)
case(axs[0],308057,'2017-05-20','2017-07-20','Hillhurst Sunnyside, 2017')
case(axs[1],319086,'2018-08-10','2018-10-10','Whitehorn MSC, 2018')
axs[1].set_ylabel(''); h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(0.5,1.06))
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig('out/fig/case.png')
R=pd.read_csv('out/tab/anomaly2.csv',header=[0,1],index_col=0)
order=['Peer index, masked (proposed)','Peer index, unmasked','Single-site index','Isolation Forest']
lab={order[0]:'Peer index\n(masked)',order[1]:'Peer index\n(unmasked)',order[2]:'Single-site\nclear-sky index',order[3]:'Isolation\nForest'}
fig,axs=plt.subplots(1,2,figsize=(W2,2.0),gridspec_kw={'width_ratios':[1,1.6]})
x=np.arange(4)
axs[0].bar(x,[R.loc[o,('AP','mean')] for o in order],yerr=[R.loc[o,('AP','std')] for o in order],color=PAL[0],width=0.6,capsize=2,error_kw={'lw':0.8})
axs[0].set_xticks(x); axs[0].set_xticklabels([lab[o] for o in order],fontsize=6.5); axs[0].set_ylabel('Average precision'); axs[0].grid(axis='x',visible=False)
kinds=[('ev_outage','Outage\n(silent)'),('ev_derate50','50%\nderate'),('ev_derate25','25%\nderate'),('ev_shading','Morning\nshading'),('ev_gradual','Gradual\ndecline'),('ev_clipping','Peak\nclipping')]
w=0.2
for i,o in enumerate(order):
    axs[1].bar(np.arange(6)+(i-1.5)*w,[R.loc[o,(k,'mean')] for k,_ in kinds],width=w*0.9,color=PAL[i],label=lab[o].replace('\n',' '))
axs[1].set_xticks(range(6)); axs[1].set_xticklabels([n for _,n in kinds],fontsize=6.5); axs[1].set_ylabel('Event recall'); axs[1].legend(frameon=False,fontsize=6); axs[1].grid(axis='x',visible=False)
fig.tight_layout(); fig.savefig('out/fig/anomaly_bench.png')
