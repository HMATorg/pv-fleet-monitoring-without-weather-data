import pandas as pd, numpy as np
from features import load
def daily():
    H=load(); H['day']=H.ts.dt.normalize(); H['hr']=H.ts.dt.hour
    P=H.pivot_table(index=['id','day'],columns='hr',values='y',aggfunc='first',dropna=False)
    P=P.reindex(columns=range(24))
    CS=H.groupby('day').csn.first()  # same coords for all sites
    CSd=H.groupby(['day','hr']).csn.first().groupby('day').sum()
    return P, CSd
