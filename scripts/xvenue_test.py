"""Cross-venue tests. Usage: python3 xvenue_test.py [thr]"""
import pandas as pd, numpy as np, statsmodels.api as sm, sys, warnings; warnings.filterwarnings('ignore')
R='/Users/abebraham/Desktop/resume/edge/data/'
THR=float(sys.argv[1]) if len(sys.argv)>1 else 0.05
X=pd.read_parquet(R+'kalshi/xvenue_panel.parquet')
X=X[(X.ask>0)&(X.ask<1)&(X.bid<1)&(X.bid>=0)].copy()
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
X=X.dropna(subset=['mu_k','sig_k','px']); X['lmid']=lg(X.mid); X['lpx']=lg(X.px); X['lpk']=lg(X.pk_fit); X['tgt']=pd.to_datetime(X.target); X['month']=X.tgt.dt.to_period('M')
X['hour']=X.ts.dt.hour; X['dayrel']=(X.ts.dt.normalize()-X.tgt.dt.tz_localize('UTC')).dt.days   # -1 = D-1, 0 = D
X['season']=np.where(X.tgt.dt.month.isin([11,12,1,2,3]),'winter',np.where(X.tgt.dt.month.isin([6,7,8,9]),'summer','shoulder'))
print('rows',len(X),'events',X.event_ticker.nunique(),'range',X.target.min(),X.target.max()); print(X.groupby('city').event_ticker.nunique().to_dict()); print(X.groupby('season').event_ticker.nunique().to_dict())
def clus(mod,g): return mod.fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(g)[0]})
# ---------- 1. information beyond the Kalshi price, by hour block
print('\n=== 1. Does the Polymarket-implied bracket probability carry information beyond the Kalshi mid? (logit, event-clustered) ===')
for lab,sel in [('D-1 14-17Z',(X.dayrel==-1)&X.hour.between(14,17)),('D-1 18-21Z',(X.dayrel==-1)&X.hour.between(18,21)),('D-1 22Z-23Z',(X.dayrel==-1)&X.hour.between(22,23)),('D 00-03Z',(X.dayrel==0)&X.hour.between(0,3)),('D 04-05Z',(X.dayrel==0)&X.hour.between(4,5)),('ALL',X.dayrel.isin([-1,0]))]:
    s=X[sel]
    if len(s)<200: continue
    m1=clus(sm.Logit(s.y,sm.add_constant(s[['lmid','lpx']])),s.event_ticker)
    m0=clus(sm.Logit(s.y,sm.add_constant(s[['lmid','lpk']])),s.event_ticker)   # control: Kalshi's own smoothed fit
    print(f'{lab:12s} n={len(s):6d} ev={s.event_ticker.nunique():4d} | lmid {m1.params.lmid:.2f} lpx {m1.params.lpx:.2f} (z={m1.tvalues.lpx:4.1f}) | control own-fit lpk {m0.params.lpk:.2f} (z={m0.tvalues.lpk:4.1f}) | Brier mid {((s.mid-s.y)**2).mean():.4f} px {((s.px-s.y)**2).mean():.4f} comb {((m1.predict(sm.add_constant(s[["lmid","lpx"]]))-s.y)**2).mean():.4f}')
# ---------- 2. lead-lag / error correction on implied means
print('\n=== 2. Lead-lag on implied means (per event, hourly): dmu_K(t+1) ~ gap(t)=mu_P_adj-mu_K, dmu_K(t), dmu_P(t) ; and reverse ===')
E=X.drop_duplicates(['event_ticker','ts'])[['event_ticker','city','ts','mu_k','mu_p_adj','sig_k','sig_p','dayrel','hour','season']].dropna().sort_values(['event_ticker','ts']).copy()
g=E.groupby('event_ticker')
E['gap']=E.mu_p_adj-E.mu_k
E['dk']=g.mu_k.diff(); E['dp']=g.mu_p_adj.diff(); E['dk1']=g.dk.shift(-1); E['dp1']=g.dp.shift(-1)
E['contig']=(g.ts.diff()==pd.Timedelta(hours=1))&(g.ts.diff(-1)==-pd.Timedelta(hours=1))
E2=E[E.contig].dropna(subset=['dk','dp','dk1','dp1','gap'])
E2=E2[(E2.gap.abs()<8)&(E2.dk.abs()<8)&(E2.dp.abs()<8)]
for lab,s in [('all',E2),('D-1',E2[E2.dayrel==-1]),('D',E2[E2.dayrel==0])]+[(c,E2[E2.city==c]) for c in E2.city.unique()]:
    if len(s)<300: continue
    mk=sm.OLS(s.dk1,sm.add_constant(s[['gap','dk','dp']])).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.event_ticker)[0]})
    mp=sm.OLS(-s.dp1,sm.add_constant(s[['gap','dp','dk']])).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.event_ticker)[0]})   # -dp1 so that positive coef = Poly moves toward Kalshi
    aK=mk.params.gap; aP=mp.params.gap; gg=aK/(aK+aP) if (aK+aP)!=0 else np.nan
    print(f'{lab:9s} n={len(s):6d} | Kalshi->Poly-price: gap {aK:+.3f} (t={mk.tvalues.gap:4.1f}) lagged dP {mk.params.dp:+.3f} (t={mk.tvalues.dp:4.1f}) own dK {mk.params.dk:+.3f} | Poly->Kalshi-price: gap {aP:+.3f} (t={mp.tvalues.gap:4.1f}) lagged dK {mp.params.dk:+.3f} (t={mp.tvalues.dk:4.1f}) | share of adjustment done by Kalshi {gg:.2f} | corr(dk,dp) {s.dk.corr(s.dp):.2f} sd gap {s.gap.std():.2f}')
# ---------- 3. walk-forward trading on Kalshi, taker at ask/bid, hold to settlement
print(f'\n=== 3. Walk-forward trading Kalshi (monthly refit on prior months, warm-up 40 events), thr={THR}, taker fee, one position per bracket-side per event (first signal) ===')
def wf(df,cols,warm=40):
    preds=[]
    for m in sorted(df.month.unique()):
        tr=df[df.month<m]; te=df[df.month==m]
        if tr.event_ticker.nunique()<warm: continue
        mod=sm.Logit(tr.y,sm.add_constant(tr[cols])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[cols])),**{'b_'+c:mod.params[c] for c in cols}))
    return pd.concat(preds) if preds else None
def trade(t,label,thr=THR,first_only=True,cost_extra=0.0):
    t=t.sort_values('ts'); by=t[t.ph-t.ask>thr].copy(); bn=t[t.bid-t.ph>thr].copy()
    by['side']='Y'; bn['side']='N'
    if first_only: by=by.drop_duplicates('ticker'); bn=bn.drop_duplicates('ticker')
    by['pnl']=by.y-by.ask-fee(by.ask)-cost_extra; bn['pnl']=(1-bn.y)-(1-bn.bid)-fee(bn.bid)-cost_extra
    a=pd.concat([by,bn])
    if len(a)<5: print(f'{label:60s} N={len(a)}'); return a
    daily=pd.Series(0.0,index=t.target.unique()); dd=a.groupby('target').pnl.sum(); daily.loc[dd.index]=dd.values
    print(f'{label:60s} N={len(a):5d} ({len(by)}Y/{len(bn)}N) pnl/ct={a.pnl.mean():+.4f} t={a.pnl.mean()/a.pnl.std()*np.sqrt(len(a)):4.1f} win={(a.pnl>0).mean():.2f} dailySharpe={daily.mean()/daily.std()*np.sqrt(365):.2f} days={len(daily)} | ' + ' '.join(f'{c}:{v:+.3f}({n})' for c,v,n in a.groupby('city').pnl.agg(['mean','size']).itertuples()))
    return a
for lab,sel in [('D-1 all hours',X.dayrel==-1),('D-1 22Z..D 03Z',((X.dayrel==-1)&(X.hour>=22))|((X.dayrel==0)&(X.hour<=3))),('D 00-05Z',X.dayrel==0),('all hours',X.dayrel.isin([-1,0]))]:
    t=wf(X[sel],['lmid','lpx'])
    if t is None: continue
    a=trade(t,f'{lab} | mid+poly')
    if lab=='all hours':
        for ssn in ['winter','shoulder','summer']:
            if (a.season==ssn).any(): print(f'    {ssn:8s} N={(a.season==ssn).sum():4d} pnl/ct={a[a.season==ssn].pnl.mean():+.4f}')
        print('    mean coef lpx over refits', round(t.b_lpx.mean(),2))
        trade(t,f'{lab} | mid+poly, +2c slippage',cost_extra=0.02)
        trade(t,f'{lab} | mid+poly, thr .08',thr=0.08)
        trade(t,f'{lab} | mid+poly, every hour (not first-only)',first_only=False)
    tc=wf(X[sel],['lmid','lpk']); 
    if tc is not None: trade(tc,f'{lab} | CONTROL mid+own normal fit')
# ---------- 4. placebo: Poly from the previous city-day
print('\n=== 4. Placebo: Polymarket distribution taken from the SAME city, PREVIOUS target day ===')
Pm=X.drop_duplicates(['city','target','ts'])[['city','target','ts','mu_p_adj','sig_p','ds']].copy(); Pm['target']=Pm.target+pd.Timedelta(days=1); Pm['ts']=Pm.ts+pd.Timedelta(days=1)
Z=X.drop(columns=['mu_p_adj','sig_p']).merge(Pm.drop(columns=['ds']),on=['city','target','ts'])
from scipy.stats import norm
sx=np.sqrt(Z.sig_p**2+Z.ds**2); Z['px']=norm.cdf((Z.hi-Z.mu_p_adj)/sx)-norm.cdf((Z.lo-Z.mu_p_adj)/sx); Z['lpx']=lg(Z.px)
tz=wf(Z[Z.dayrel.isin([-1,0])],['lmid','lpx'])
if tz is not None: trade(tz,'placebo (yesterday\'s Polymarket) all hours'); print('    placebo lpx coef', round(tz.b_lpx.mean(),2))
