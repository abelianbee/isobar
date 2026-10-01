"""Diagnostics for the Kalshi->Polymarket trade. Execution at RAW Polymarket price (not strip-normalised). Signal = logit(lp, lx[, lpm])."""
import pandas as pd, numpy as np, statsmodels.api as sm, warnings, sys; warnings.filterwarnings('ignore')
R='/Users/abebraham/Desktop/resume/edge/data/'
Y=pd.read_parquet(R+'kalshi/xvenue_poly_scored.parquet')
K=pd.read_parquet(R+'kalshi/xvenue_panel.parquet').drop_duplicates(['event_ticker','ts'])[['city','target','ts','h','vol','oi']]
Kh=K.groupby(['city','target','ts']).agg(h=('h','first')).reset_index(); Y=Y.merge(Kh,on=['city','target','ts'],how='left')
print('psum distribution (raw strip sum):', Y.psum.describe()[['mean','25%','50%','75%']].round(3).to_dict())
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
def wf(df,cols,warm=40):
    out=[]
    for m in sorted(df.month.unique()):
        tr=df[df.month<m]; te=df[df.month==m]
        if tr.slug.nunique()<warm: continue
        mod=sm.Logit(tr.y,sm.add_constant(tr[cols])).fit(disp=0); out.append(te.assign(ph=mod.predict(sm.add_constant(te[cols])),**{'b_'+c:mod.params[c] for c in cols}))
    return pd.concat(out)
def signals(t,thr,hs,exec_col='p',first=True):
    px=t[exec_col]
    by=t[t.ph-(px+hs)>thr].copy(); by['side']='Y'; by['cost']=px[by.index]+hs; by['pnl']=by.y-by.cost; by['edge']=by.ph-by.cost
    bn=t[(px-hs)-t.ph>thr].copy(); bn['side']='N'; bn['cost']=1-(px[bn.index]-hs); bn['pnl']=(1-bn.y)-bn.cost; bn['edge']=(1-bn.ph)-bn.cost
    a=pd.concat([by,bn]).sort_values('ts')
    return a.drop_duplicates(['slug','lo']) if first else a
def rep(a,label):
    if len(a)<5: print(f'{label:55s} N={len(a)}'); return
    daily=a.groupby('target').pnl.sum(); alld=pd.Series(0.0,index=sorted(Y.target.unique())); alld.loc[daily.index]=daily.values
    print(f'{label:55s} N={len(a):5d} pnl/ct={a.pnl.mean():+.4f} sd={a.pnl.std():.3f} t={a.pnl.mean()/a.pnl.std()*np.sqrt(len(a)):4.1f} win={(a.pnl>0).mean():.2f} dSharpe(all days)={alld.mean()/alld.std()*np.sqrt(365):.2f} ROI={a.pnl.sum()/a.cost.sum():+.3f}')
T2=wf(Y,['lp','lx']); T3=wf(Y,['lp','lx','lpm'])
print('\n=== RAW-price execution (the honest number), hs=2c ===')
for thr in [0.05,0.08]:
    rep(signals(T2,thr,0.02,'pn'),f'lp+lx thr {thr} exec at NORMALISED price (as before)')
    rep(signals(T2,thr,0.02,'p'),f'lp+lx thr {thr} exec at RAW price')
    rep(signals(T3,thr,0.02,'p'),f'lp+lx+lpm thr {thr} exec at RAW price')
A=signals(T2,0.05,0.02,'p')
print('\n=== lp+lx thr .05 hs 2c raw: breakdowns ===')
print('by side:'); print(A.groupby('side').pnl.agg(['mean','size']).round(4).to_string())
print('by season x hour block:')
A['blk']=np.select([(A.dayrel==-1)&(A.hour<=21),((A.dayrel==-1)&(A.hour>=22))|((A.dayrel==0)&(A.hour<=3))],['D-1 <=21Z','22Z-03Z'],'D 04-06Z')
print(A.pivot_table(index='season',columns='blk',values='pnl',aggfunc=['mean','size']).round(4).to_string())
print('by Kalshi hours-since-open h:'); A['hb']=pd.cut(A.h,[0,1,2,4,8,12,20,40]); print(A.groupby('hb',observed=True).pnl.agg(['mean','size']).round(4).T.to_string())
print('by month:'); print(A.groupby('month').pnl.agg(['mean','size','sum']).round(3).T.to_string())
print('by city x season:'); print(A.pivot_table(index='city',columns='season',values='pnl',aggfunc=['mean','size']).round(4).to_string())
print('by model edge bin:'); A['eb']=pd.cut(A.edge,[0.05,0.07,0.10,0.15,0.25,1]); print(A.groupby('eb',observed=True).pnl.agg(['mean','size']).round(4).T.to_string())
print('by Poly price level:'); A['pb']=pd.cut(A.p,[0,0.05,0.15,0.3,0.5,0.7,0.85,1]); print(A.groupby('pb',observed=True).pnl.agg(['mean','size']).round(4).T.to_string())
print('mean refit coefs: lx',T2.b_lx.mean().round(2),'lp',T2.b_lp.mean().round(2))
# ---- exit at convergence instead of settlement: exit after k hours at raw Poly price (no spread on exit -> add hs)
print('\n=== exit after k hours at Poly price (pay hs again) vs hold to settlement ===')
Ys=Y.set_index(['slug','lo','ts']).p
for k in [3,6,12]:
    idx=pd.MultiIndex.from_arrays([A.slug,A.lo,A.ts+pd.Timedelta(hours=k)]); pk=Ys.reindex(idx).values
    ok=~np.isnan(pk); a=A[ok].copy(); pk=pk[ok]
    a['pnl_exit']=np.where(a.side=='Y',(pk-0.02)-a.cost,(1-(pk+0.02))-a.cost)
    print(f'k={k:2d}h: N={len(a)} exit pnl/ct={a.pnl_exit.mean():+.4f} (t={a.pnl_exit.mean()/a.pnl_exit.std()*np.sqrt(len(a)):.1f}) vs hold {a.pnl.mean():+.4f}')
# ---- placebos
print('\n=== placebos ===')
from scipy.stats import norm
Kp=Y.drop_duplicates(['city','target','ts'])[['city','target','ts','mu_k','sig_k','dm','ds']].copy()
# (a) Kalshi from the previous city-day (shift target and ts by +1 day) -> stale by a day, wrong day
Z=Kp.copy(); Z['target']=Z.target+pd.Timedelta(days=1); Z['ts']=Z.ts+pd.Timedelta(days=1)
W=Y.drop(columns=['mu_k','sig_k']).merge(Z[['city','target','ts','mu_k','sig_k']],on=['city','target','ts'])
s=np.sqrt(W.sig_k**2+W.ds**2); W['pxk']=norm.cdf((W.hi-(W.mu_k-W.dm))/s)-norm.cdf((W.lo-(W.mu_k-W.dm))/s); W['lx']=lg(W.pxk)
tw=wf(W,['lp','lx']); rep(signals(tw,0.05,0.02,'p'),'placebo A: Kalshi of previous day'); print('   lx coef',round(tw.b_lx.mean(),2))
# (b) stale Kalshi: same day, Kalshi price 6h earlier
Z=Kp.copy(); Z['ts']=Z.ts+pd.Timedelta(hours=6)
W=Y.drop(columns=['mu_k','sig_k']).merge(Z[['city','target','ts','mu_k','sig_k']],on=['city','target','ts'])
s=np.sqrt(W.sig_k**2+W.ds**2); W['pxk']=norm.cdf((W.hi-(W.mu_k-W.dm))/s)-norm.cdf((W.lo-(W.mu_k-W.dm))/s); W['lx']=lg(W.pxk)
tw=wf(W,['lp','lx']); rep(signals(tw,0.05,0.02,'p'),'placebo B: Kalshi 6h stale (same day)'); print('   lx coef',round(tw.b_lx.mean(),2))
# (c) shuffled Kalshi across days within city (shift ts by the same offset so hours line up)
rng=np.random.default_rng(0); Z=Kp.copy(); Z['target']=pd.to_datetime(Z.target)
parts=[]
for c,g in Z.groupby('city'):
    days=np.sort(g.target.unique()); perm=dict(zip(days,rng.permutation(days))); nt=g.target.map(perm); parts.append(g.assign(ts=g.ts+(nt-g.target), target=nt))
Z=pd.concat(parts); Y['target_dt']=pd.to_datetime(Y.target)
W=Y.drop(columns=['mu_k','sig_k']).merge(Z.rename(columns={'target':'target_dt'})[['city','target_dt','ts','mu_k','sig_k']],on=['city','target_dt','ts'])
s=np.sqrt(W.sig_k**2+W.ds**2); W['pxk']=norm.cdf((W.hi-(W.mu_k-W.dm))/s)-norm.cdf((W.lo-(W.mu_k-W.dm))/s); W['lx']=lg(W.pxk)
tw=wf(W,['lp','lx']); rep(signals(tw,0.05,0.02,'p'),'placebo C: Kalshi shuffled across days'); print('   lx coef',round(tw.b_lx.mean(),2), 'rows', len(W))
# ---- capacity proxy: Polymarket per-bracket total volume ($) for signalled brackets
print('\n=== capacity proxy: Polymarket total $ volume of signalled brackets ===')
print(A.bvol.describe()[['25%','50%','75%']].round(0).to_dict(), 'signals/day', round(len(A)/A.target.nunique(),2), 'days', A.target.nunique())
print('signals/day by city:', A.groupby('city').apply(lambda g: round(len(g)/g.target.nunique(),2)).to_dict())
A.drop(columns=['hb','eb','pb'],errors='ignore').to_parquet(R+'kalshi/xvenue_trades_v3.parquet')
