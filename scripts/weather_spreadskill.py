"""Improve bracket probabilities: heteroscedastic residual scale from inter-model spread + season, per station. Compare Brier and edge vs fixed empirical residuals. Also maker-execution variant (fill at mid if the bracket trades in the next hour; maker fee 0)."""
import pandas as pd, numpy as np, statsmodels.api as sm, sys
sys.argv=['x','w','ext']
src=open('/Users/abebraham/Desktop/resume/edge/scripts/weather_edge2.py').read().split("def run(h):")[0]
# patch: keep per-row model spread and the residual pool with covariates
src=src.replace("rows.append(pd.DataFrame({'station':st,'target':te.date.values,'mu':mu.values,'sig_tr':res.std(),'year':y,'res_key':st+str(y),'ncols':len(cols)}))",
"spr_te=te[cols].std(axis=1).values; spr_tr=tr[cols].std(axis=1).values; rows.append(pd.DataFrame({'station':st,'target':te.date.values,'mu':mu.values,'spread':spr_te,'sig_tr':res.std(),'year':y,'res_key':st+str(y),'ncols':len(cols)})); RESX.setdefault(st+str(y),(res, spr_tr, tr.doy.values))")
src=src.replace("rows=[]\nfor st,g in P.groupby('station'):","rows=[]; RESX={}\nfor st,g in P.groupby('station'):")
src=src.replace("c=c.merge(M[['station','target','mu','res_key']],on=['station','target'])","c=c.merge(M[['station','target','mu','res_key','spread']],on=['station','target'])")
exec(src)
c=c[pd.to_datetime(c.target)>=pd.Timestamp('2025-01-01')].copy()
# heteroscedastic sigma model per res_key: |res| ~ a + b*spread + c*cos(doy)
def hetero_pm(row_mu, row_spread, row_doy, lo, hi, key):
    res,spr,doy=RESX[key]
    X=np.column_stack([np.ones(len(res)), spr, np.cos(2*np.pi*doy/365)]); beta=np.linalg.lstsq(X, np.abs(res), rcond=None)[0]
    sig_hat=X@beta; z=res/np.maximum(sig_hat,0.5)   # standardized residuals
    s_now=max(0.5, beta[0]+beta[1]*row_spread+beta[2]*np.cos(2*np.pi*row_doy/365))
    v=row_mu+z*s_now; return ((v>lo)&(v<hi)).mean()
c['doy']=pd.to_datetime(c.target).dt.dayofyear
print('computing heteroscedastic probabilities...', flush=True)
c['pm2']=[hetero_pm(m,s,d,l,h,k) for m,s,d,l,h,k in zip(c.mu,c.spread,c.doy,c.lo,c.hi,c.res_key)]
c['lpm2']=lg(c.pm2)
def evaluate(h, col, label, maker=False):
    s=c[c.h==h]; preds=[]
    for m in sorted(s.month.unique()):
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<120: continue
        mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid',col]])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[['lmid',col]])), b=mod.params[col]))
    t=pd.concat(preds); thr=0.05
    if maker:
        nxt=c[c.h==h+1][['ticker','vol']].rename(columns={'vol':'vol_next'}); t=t.merge(nxt,on='ticker',how='left').fillna({'vol_next':0})
        by=t[(t.ph-t.mid>thr)&(t.vol_next>0)].copy(); bn=t[(t.mid-t.ph>thr)&(t.vol_next>0)].copy()
        by['pnl']=by.y-by.mid; bn['pnl']=(1-bn.y)-(1-bn.mid)
    else:
        by=t[t.ph-t.ask>thr].copy(); bn=t[t.bid-t.ph>thr].copy(); by['pnl']=by.y-by.ask-fee(by.ask); bn['pnl']=(1-bn.y)-(1-bn.bid)-fee(bn.bid)
    sig=pd.concat([by,bn]); sig['mon']=sig.tgt.dt.month; sig['winter']=sig.mon.isin([11,12,1,2,3])
    daily=pd.Series(0.0,index=t.target.unique()); dd=sig.groupby('target').pnl.sum(); daily.loc[dd.index]=dd.values
    w=sig[sig.winter]; dw=pd.Series(0.0,index=t[t.tgt.dt.month.isin([11,12,1,2,3])].target.unique()); ddw=w.groupby('target').pnl.sum(); dw.loc[ddw.index]=ddw.values
    pcol=t[col.replace('l','',1)] if col!='lpm' else t.pm
    print(f'h={h} {label:28s} Brier mkt={((t.mid-t.y)**2).mean():.4f} model={((pcol-t.y)**2).mean():.4f} comb={((t.ph-t.y)**2).mean():.4f} coef={t.b.mean():.2f} | ALL N={len(sig)} pnl/ct={sig.pnl.mean():+.4f} Sharpe={daily.mean()/daily.std()*np.sqrt(365):.2f} | WINTER N={len(w)} pnl/ct={w.pnl.mean():+.4f} Sharpe={dw.mean()/dw.std()*np.sqrt(365):.2f}')
for h in [8,10]:
    evaluate(h,'lpm','fixed residuals (baseline)')
    evaluate(h,'lpm2','spread-skill heteroscedastic')
    evaluate(h,'lpm','baseline, MAKER at mid',maker=True)
    evaluate(h,'lpm2','spread-skill, MAKER at mid',maker=True)
