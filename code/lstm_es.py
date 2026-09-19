import pandas as pd, numpy as np, torch, time
from torch import nn
from features import load
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
H=load(); L=48
def build(h):
    Xs,Xf,Y,meta=[],[],[],[]
    for sid,g in H.groupby('id',sort=False):
        y=g.y.values.astype(np.float32); cs=g.csn.values.astype(np.float32); hr=g.ts.dt.hour.values; doy=g.ts.dt.dayofyear.values
        el=g.elev.values; ts=g.ts.values
        seq=np.stack([y,cs,np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24)],1).astype(np.float32)
        idx=np.arange(L+h-1,len(y))
        idx=idx[(el[idx]>0)&~np.isnan(y[idx])]
        win=np.lib.stride_tricks.sliding_window_view(seq,(L,4))[:,0]  # win[i]=seq[i:i+L]
        s=idx-h-L+1
        W=win[s]; ok=~np.isnan(W).any((1,2)); idx=idx[ok]; W=W[ok]
        fut=np.stack([cs[idx],np.sin(2*np.pi*hr[idx]/24),np.cos(2*np.pi*hr[idx]/24),np.sin(2*np.pi*doy[idx]/365.25),np.cos(2*np.pi*doy[idx]/365.25)],1)
        Xs.append(W); Xf.append(fut.astype(np.float32)); Y.append(y[idx]); meta.append(pd.DataFrame({'id':sid,'ts':ts[idx]}))
    return np.concatenate(Xs),np.concatenate(Xf),np.concatenate(Y),pd.concat(meta,ignore_index=True)
class Net(nn.Module):
    def __init__(s,hid=64):
        super().__init__(); s.lstm=nn.LSTM(4,hid,num_layers=2,batch_first=True,dropout=0.1)
        s.head=nn.Sequential(nn.Linear(hid+5,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x,f):
        o,_=s.lstm(x); return s.head(torch.cat([o[:,-1],f],1)).squeeze(1)
out=[]
for h in [24,1]:
    Xs,Xf,Y,M=build(h); print('h',h,Xs.shape,flush=True)
    for Yr in ([2020,2021,2022] if h==24 else [2022]):
        tr=np.where(M.ts<np.datetime64(f'{Yr}-01-01'))[0]; te=np.where((M.ts>=np.datetime64(f'{Yr}-01-01'))&(M.ts<np.datetime64(f'{Yr+1}-01-01')))[0]
        net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); t0=time.time()
        Xs_t=torch.from_numpy(Xs); Xf_t=torch.from_numpy(Xf); Y_t=torch.from_numpy(Y)
        cut=np.datetime64(f'{Yr-1}-01-01')
        trn=tr[M.ts.values[tr]<cut]; val=tr[M.ts.values[tr]>=cut]
        best=1e9; bad=0; best_ep=0; import copy; best_state=None
        for ep in range(20):
            perm=np.random.permutation(trn); net.train()
            for b in range(0,len(perm),1024):
                i=perm[b:b+1024]; opt.zero_grad()
                loss=nn.functional.mse_loss(net(Xs_t[i],Xf_t[i]),Y_t[i]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl=float(np.mean([nn.functional.mse_loss(net(Xs_t[val[b:b+8192]],Xf_t[val[b:b+8192]]),Y_t[val[b:b+8192]]).item() for b in range(0,len(val),8192)]))
            if vl<best-1e-6: best=vl; bad=0; best_ep=ep+1; best_state=copy.deepcopy(net.state_dict())
            else: bad+=1
            print('  ep',ep+1,'val',round(vl,6),flush=True)
            if bad>=3: break
            for gp in opt.param_groups: gp['lr']*=0.85
        net.load_state_dict(best_state); tl=best
        net.eval(); P=[]
        with torch.no_grad():
            for b in range(0,len(te),8192):
                i=te[b:b+8192]; P.append(net(Xs_t[i],Xf_t[i]).numpy())
        d=M.iloc[te].copy(); d['LSTM']=np.clip(np.concatenate(P),0,None); d['h']=h; d['fold']=Yr; out.append(d)
        print(h,Yr,len(tr),len(te),'best_ep',best_ep,'val',best,round(time.time()-t0),flush=True)
pd.concat(out).to_parquet('out/preds_lstm_es.parquet')
