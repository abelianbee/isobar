"""Identical contracts on two venues: Kalshi and Polymarket 2F brackets with the same edges, same station (Miami, Atlanta, Seattle).
Model-free price discovery: error-correction speeds, Gonzalo-Granger and Hasbrouck information shares, outcome-information logit,
settlement-source disagreement, and a model-free laggard trade with pre-specified thresholds."""
import pandas as pd, numpy as np, statsmodels.api as sm, warnings, sys; warnings.filterwarnings('ignore')
R='/Users/abebraham/Desktop/resume/edge/data/'
K=pd.read_parquet(R+'kalshi/xvenue_panel.parquet')[['city','target','ts','h','ticker','lo','hi','y','bid','ask','mid','vol','oi']].rename(columns={'y':'y_k'})
Pp=pd.read_parquet(R+'kalshi/xvenue_panel_poly.parquet')[['city','target','ts','slug','lo','hi','y','p','pn','bvol']].rename(columns={'y':'y_p'})
SAME=['miami','atlanta','seattle']+(sys.argv[1].split(',') if len(sys.argv)>1 else [])
J=K[K.city.isin(SAME)].merge(Pp[Pp.city.isin(SAME)],on=['city','target','ts','lo','hi'])
J=J[(J.ask>0)&(J.ask<1)&(J.bid>=0)&(J.p>0)&(J.p<1)].copy()
print('identical-contract rows',len(J),'events',J.groupby(['city','target']).ngroups,J.groupby('city').apply(lambda g: g[['target']].drop_duplicates().shape[0]).to_dict(),'brackets/event',round(len(J)/J.groupby(['city','target','ts']).ngroups,2))
# settlement disagreement (Kalshi NWS CLI vs Polymarket Wunderground)
u=J.drop_duplicates(['ticker'])[['city','target','lo','hi','y_k','y_p']]
print(f'settlement disagreement on identical brackets: {(u.y_k!=u.y_p).mean():.3%} of {len(u)} brackets; events with any disagreement: {u.assign(d=u.y_k!=u.y_p).groupby(["city","target"]).d.any().mean():.3%}')
print('mean |Kalshi mid - Poly mid|:', (J.mid-J.p).abs().mean().round(4), ' median Kalshi spread', (J.ask-J.bid).median(), ' corr(mid,p)', J.mid.corr(J.p).round(3))
# error correction on identical contracts
J=J.sort_values(['ticker','ts']); g=J.groupby('ticker')
J['dK']=g.mid.diff(); J['dP']=g.p.diff(); J['dK1']=g.mid.shift(-1)-J.mid; J['dP1']=g.p.shift(-1)-J.p
J['ok']=(g.ts.shift(-1)-J.ts==pd.Timedelta(hours=1))&(J.ts-g.ts.shift(1)==pd.Timedelta(hours=1))
J['gap']=J.mid-J.p   # Kalshi minus Poly
E=J[J.ok].dropna(subset=['dK','dP','dK1','dP1'])
def ecm(E,lab):
    cl={'groups':pd.factorize(E.ticker)[0]}
    mP=sm.OLS(E.dP1,sm.add_constant(E[['gap','dP','dK']])).fit(cov_type='cluster',cov_kwds=cl)    # Poly moves toward Kalshi if gap coef > 0
    mK=sm.OLS(E.dK1,sm.add_constant(E[['gap','dK','dP']])).fit(cov_type='cluster',cov_kwds=cl)    # Kalshi moves toward Poly if gap coef < 0
    aP=mP.params.gap; aK=mK.params.gap
    # Gonzalo-Granger: alpha=(aK,aP) on (p_K - p_P); alpha_perp ∝ (aP, -aK); share of Kalshi = aP/(aP-aK)
    gg=aP/(aP-aK) if (aP-aK)!=0 else np.nan
    # Hasbrouck IS bounds
    res=np.vstack([mK.resid.values,mP.resid.values]).T; Om=np.cov(res.T); psi=np.array([aP,-aK])/(aP-aK)
    def IS(order):
        idx=list(order); O=Om[np.ix_(idx,idx)]; F=np.linalg.cholesky(O); ps=psi[idx]; c=ps@F; s=c**2/(c**2).sum(); out=np.empty(2); out[idx]=s; return out
    is1=IS([0,1]); is2=IS([1,0]); lo_k=min(is1[0],is2[0]); hi_k=max(is1[0],is2[0])
    print(f'{lab:10s} n={len(E):6d} | Poly->Kalshi speed {aP:+.3f}/h (t={mP.tvalues.gap:4.1f}) | Kalshi->Poly speed {-aK:+.3f}/h (t={-mK.tvalues.gap:4.1f}) | GG share Kalshi {gg:.2f} | Hasbrouck IS Kalshi [{lo_k:.2f},{hi_k:.2f}] | corr(dK,dP) {E.dK.corr(E.dP):.2f} | Poly half-life {np.log(.5)/np.log(1-aP) if 0<aP<1 else np.nan:.1f}h')
print('\n=== error-correction / information shares (identical contracts, hourly) ===')
ecm(E,'all')
for c in SAME:
    if (E.city==c).sum()>200: ecm(E[E.city==c],c)
E['blk']=np.where(E.h<=8,'h<=8','h>8'); 
for b in ['h<=8','h>8']: ecm(E[E.blk==b],b)
# outcome information
print('\n=== which venue predicts the outcome? logit(y) ~ logit(p_K) + logit(p_P), event-clustered ===')
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
J['lk']=lg(J.mid); J['lp']=lg(J.p)
for lab,s in [('all',J),('h<=8',J[J.h<=8]),('h 9-16',J[J.h.between(9,16)]),('h>16',J[J.h>16])]:
    if len(s)<100: continue
    m=sm.Logit(s.y_p,sm.add_constant(s[['lk','lp']])).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(s.ticker)[0]})
    print(f'{lab:7s} n={len(s):6d} Kalshi {m.params.lk:.2f} (z={m.tvalues.lk:.1f})  Poly {m.params.lp:.2f} (z={m.tvalues.lp:.1f}) | Brier Kalshi {((s.mid-s.y_p)**2).mean():.4f} Poly {((s.p-s.y_p)**2).mean():.4f}')
# model-free trade on Poly (settle on Poly outcome y_p), pre-specified thresholds, first signal per bracket-side
print('\n=== model-free laggard trade: buy Polymarket toward Kalshi mid when |Kalshi mid - Poly mid| > thr; cost = Poly mid + hs; hold to settlement ===')
def mf(J,thr,hs,lab,delay=False):
    t=J.sort_values('ts').copy()
    if delay:
        t['p_exec']=t.groupby('ticker').p.shift(-1); t=t.dropna(subset=['p_exec'])
    else: t['p_exec']=t.p
    by=t[t.mid-t.p>thr].drop_duplicates('ticker'); bn=t[t.p-t.mid>thr].drop_duplicates('ticker')
    a=pd.concat([by.assign(pnl=by.y_p-(by.p_exec+hs),side='Y'),bn.assign(pnl=(1-bn.y_p)-(1-(bn.p_exec-hs)),side='N')])
    if len(a)<5: print(lab,'N',len(a)); return a
    daily=a.groupby('target').pnl.sum(); alld=pd.Series(0.0,index=sorted(J.target.unique())); alld.loc[daily.index]=daily.values
    print(f'{lab:34s} thr={thr:.2f} hs={hs:.2f}{" DELAY1h" if delay else ""} N={len(a):4d} pnl/ct={a.pnl.mean():+.4f} t={a.pnl.mean()/a.pnl.std()*np.sqrt(len(a)):4.1f} win={(a.pnl>0).mean():.2f} dSharpe={alld.mean()/alld.std()*np.sqrt(365):.2f} | '+' '.join(f'{c}:{v:+.3f}({n})' for c,v,n in a.groupby('city').pnl.agg(['mean','size']).itertuples()))
    return a
for thr in [0.05,0.08,0.12]:
    for hs in [0.01,0.02,0.03]: mf(J,thr,hs,'Poly toward Kalshi')
    mf(J,thr,0.02,'Poly toward Kalshi',delay=True)
    mf(J[J.h<=8],thr,0.02,'Poly toward Kalshi, h<=8 only')
# reverse: Kalshi toward Poly, taker fee
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
for thr in [0.05,0.08]:
    t=J.sort_values('ts'); by=t[t.p-t.ask>thr].drop_duplicates('ticker'); bn=t[t.bid-t.p>thr].drop_duplicates('ticker')
    a=pd.concat([by.y_k-by.ask-fee(by.ask),(1-bn.y_k)-(1-bn.bid)-fee(bn.bid)])
    print(f'REVERSE Kalshi toward Poly (taker+fee)  thr={thr:.2f} N={len(a):4d} pnl/ct={a.mean():+.4f} t={a.mean()/a.std()*np.sqrt(len(a)) if len(a)>2 else 0:4.1f}')
J.to_parquet(R+'kalshi/xvenue_same.parquet')
