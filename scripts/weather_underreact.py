import pandas as pd, numpy as np, statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize
R='/Users/abebraham/Desktop/resume/edge/data/'
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[c.h<=37]
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
# fit normal to bracket mids per event-hour
def fit_normal(lo, hi, p):
    p=np.asarray(p); p=p/p.sum(); 
    cen=np.where(np.isinf(lo), hi-1, np.where(np.isinf(hi), lo+1, (lo+hi)/2)); mu0=(cen*p).sum(); 
    def loss(x):
        mu,ls=x; s=np.exp(ls); q=norm.cdf(hi,mu,s)-norm.cdf(lo,mu,s); return ((q-p)**2).sum()
    r=minimize(loss,[mu0,np.log(2.0)],method='Nelder-Mead',options={'xatol':1e-3,'fatol':1e-7,'maxiter':300}); return r.x[0], np.exp(r.x[1])
rows=[]
for (e,h),g in c.groupby(['event_ticker','h']):
    if len(g)!=6: continue
    mu,s=fit_normal(g.lo.values, g.hi.values, g.mid.values+1e-4)
    rows.append((e,h,mu,s,g.truth.iloc[0],g.target.iloc[0],g.station.iloc[0],g.ts.max(),g.tz.iloc[0]))
f=pd.DataFrame(rows,columns=['event_ticker','h','mu','sig','truth','target','station','ts','tz'])
f['local_hour']=[pd.Timestamp(t,unit='s',tz='UTC').tz_convert(tz).hour for t,tz in zip(f.ts,f.tz)]
f['day_rel']=[(pd.Timestamp(t,unit='s',tz='UTC').tz_convert(tz).date()-tg).days for t,tz,tg in zip(f.ts,f.tz,f.target)]
f.to_parquet(R+'weather/snap_normal.parquet')
print('fitted', len(f), 'sig by h:', f.groupby('h').sig.median().round(2).to_dict())
f['res']=f.truth-f.mu
print('mean res by h (calibration of fitted mean):', f.groupby('h').res.mean().round(2).to_dict())
print('sd res by h:', f.groupby('h').res.std().round(2).to_dict())
base=f[f.h==2][['event_ticker','mu']].rename(columns={'mu':'m0'})
print('\n--- underreaction test with normal-fit mean: res_h ~ (mu_h - mu_2) ---')
for h in [8,14,20,24,26,28,30,32]:
    s=f[f.h==h].merge(base,on='event_ticker'); s['chg']=s.mu-s.m0
    m=sm.OLS(s.res, sm.add_constant(s.chg)).fit(cov_type='HC1'); print(f'h={h}: coef={m.params.iloc[1]:+.3f} (se {m.bse.iloc[1]:.3f}) t={m.tvalues.iloc[1]:.1f} N={len(s)} sd chg={s.chg.std():.2f}')
# also lagged change: res_h ~ (mu_h - mu_{h-6})
print('--- res_h ~ (mu_h - mu_{h-6}) ---')
for h in [14,20,24,26,28,30]:
    s=f[f.h==h].merge(f[f.h==h-6][['event_ticker','mu']].rename(columns={'mu':'mlag'}),on='event_ticker'); s['chg']=s.mu-s.mlag
    m=sm.OLS(s.res, sm.add_constant(s.chg)).fit(cov_type='HC1'); print(f'h={h}: coef={m.params.iloc[1]:+.3f} (se {m.bse.iloc[1]:.3f}) t={m.tvalues.iloc[1]:.1f} N={len(s)}')
# --- direct trading test (no fitted mean needed): at hour h, if mu_h - m0 > thr, buy YES on brackets whose center > mu_h (the 'upper' brackets) at ask; if < -thr buy lower brackets
print('\n--- momentum trade: buy the bracket adjacent above (below) the fitted mean when market rose (fell) since open ---')
cc=c.merge(f[['event_ticker','h','mu']],on=['event_ticker','h']).merge(base,on='event_ticker'); cc['chg']=cc.mu-cc.m0
cc['center']=np.where(np.isinf(cc.lo), cc.hi-1, np.where(np.isinf(cc.hi), cc.lo+1, (cc.lo+cc.hi)/2))
for h in [14,20,24,26,28]:
    for thr in [0.5,1.0]:
        s=cc[(cc.h==h)&(cc.chg.abs()>thr)].copy()
        # choose, per event, the bracket on the side of the move: first bracket with center beyond mu in direction of chg
        s['side_ok']=np.where(s.chg>0, s.center>s.mu, s.center<s.mu); s=s[s.side_ok]
        s['dist']=(s.center-s.mu).abs(); pick=s.sort_values('dist').groupby('event_ticker').first().reset_index()
        pnl=pick.y-pick.ask-fee(pick.ask); d=pd.DataFrame({'t':pick.target,'p':pnl}).groupby('t').p.sum()
        print(f'h={h} |chg|>{thr}: N={len(pick)} ask={pick.ask.mean():.3f} mid={pick.mid.mean():.3f} hit={pick.y.mean():.3f} pnl/ct={pnl.mean():+.4f} ROI={pnl.sum()/pick.ask.sum()*100:+.1f}% Sharpe={d.mean()/d.std()*np.sqrt(365):.2f}')
    # control: same but opposite side (fade)
    s=cc[(cc.h==h)&(cc.chg.abs()>0.5)].copy(); s['side_ok']=np.where(s.chg>0, s.center<s.mu, s.center>s.mu); s=s[s.side_ok]; s['dist']=(s.center-s.mu).abs(); pick=s.sort_values('dist').groupby('event_ticker').first().reset_index()
    pnl=pick.y-pick.ask-fee(pick.ask); print(f'   FADE control h={h}: N={len(pick)} hit={pick.y.mean():.3f} mid={pick.mid.mean():.3f} pnl/ct={pnl.mean():+.4f}')
