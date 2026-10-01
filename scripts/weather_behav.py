import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/'
agg=pd.read_parquet(R+'weather/snap_market.parquet'); c=pd.read_parquet(R+'weather/candles_ev.parquet')
agg=agg[agg.h<=37]; c=c[c.h<=37]
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
print('events', agg.event_ticker.nunique(), agg.target.min(), agg.target.max())
# ---------- 1. favourite-longshot: P&L by hour and price bucket
c['pb']=pd.cut(c.mid,[0,0.03,0.06,0.1,0.2,0.35,0.5,0.65,0.8,0.9,1.0])
cal=c.groupby('pb',observed=True).agg(n=('y','size'), mid=('mid','mean'), ask=('ask','mean'), bid=('bid','mean'), hit=('y','mean')).round(3)
cal['yes_edge_at_ask']=(cal.hit-cal.ask-fee(cal.ask)).round(3); cal['no_edge_at_bid']=((1-cal.hit)-(1-cal.bid)-fee(cal.bid)).round(3)
print('\nCalibration all hours (edge per contract after taker fee):'); print(cal.to_string())
# by phase: D-1 (h<=13), overnight (14-23), D morning (24-30), D afternoon (31-37)
c['phase']=pd.cut(c.h,[-1,13,23,30,37],labels=['D-1 day','overnight','D morning','D afternoon'])
for ph,g in c.groupby('phase',observed=True):
    cal=g.groupby('pb',observed=True).agg(n=('y','size'), ask=('ask','mean'), bid=('bid','mean'), hit=('y','mean'))
    cal['yes_edge']=(cal.hit-cal.ask-fee(cal.ask)).round(3); cal['no_edge']=((1-cal.hit)-(1-cal.bid)-fee(cal.bid)).round(3)
    print(f'\n--- phase {ph} ---'); print(cal.round(3).to_string())
# ---------- 2. strategy sims with daily P&L and Sharpe: one trade per event at hour h
def sharpe(daily):
    d=daily.groupby('target').sum(); return d.mean()/d.std()*np.sqrt(365) if d.std()>0 else np.nan, d.mean(), len(d)
print('\n=== Strategy A: buy YES on favourite at ask, 1 contract, at hour h ===')
for h in [1,4,8,12,16,20,24,26,28,30,32]:
    s=agg[agg.h==h]; pnl=s.fav_y-s.fav_ask-fee(s.fav_ask); sr,mu,n=sharpe(pd.DataFrame({'target':s.target,'p':pnl}).set_index('target').p)
    print(f'h={h:2d} N={len(s):4d} avg ask={s.fav_ask.mean():.3f} hit={s.fav_y.mean():.3f} pnl/contract={pnl.mean():+.4f} ROI={pnl.sum()/s.fav_ask.sum()*100:+.1f}% Sharpe={sr:.2f}')
print('\n=== Strategy B: buy NO on every bracket with ask<=0.06 (sell longshots), 1 contract each, at hour h ===')
for h in [1,4,8,12,16,20,24,26,28,30,32]:
    s=c[(c.h==h)&(c.ask<=0.06)&(c.bid>0)]; pnl=(1-s.y)-(1-s.bid)-fee(s.bid); sr,mu,n=sharpe(pd.DataFrame({'target':s.target,'p':pnl}).set_index('target').p)
    print(f'h={h:2d} N={len(s):4d} avg bid={s.bid.mean():.3f} hit(yes)={s.y.mean():.3f} pnl/contract={pnl.mean():+.4f} ROI on capital={pnl.sum()/(1-s.bid).sum()*100:+.2f}% Sharpe={sr:.2f} days={n}')
# ---------- 3. momentum/reversal of implied mean: residual (truth - imp_mean_h) vs change (imp_mean_h - imp_mean_1)
base=agg[agg.h==2][['event_ticker','imp_mean']].rename(columns={'imp_mean':'m0'})
for h in [8,14,20,24,26,28,30]:
    s=agg[agg.h==h].merge(base,on='event_ticker'); s['res']=s.truth-s.imp_mean; s['chg']=s.imp_mean-s.m0
    m=sm.OLS(s.res, sm.add_constant(s.chg)).fit(cov_type='HC1'); print(f'h={h}: res ~ chg: coef={m.params.iloc[1]:+.3f} (se {m.bse.iloc[1]:.3f}) N={len(s)}  mean res={s.res.mean():+.2f} sd res={s.res.std():.2f}  sd chg={s.chg.std():.2f}')
# ---------- 4. H15 hot-morning: at local hour 10-12 on D, residual vs (T_now - mu1) and vs (imp_mean - mu1)
snap=pd.read_parquet(R+'weather/snap_obs.parquet'); snap['target']=snap.date
ev=pd.read_parquet(R+'weather/events.parquet')
for H in [9,10,11,12,13]:
    s=agg[(agg.day_rel==0)&(agg.local_hour==H)].merge(snap[snap.H==H][['station','target','T','RM','mu1']], on=['station','target'])
    s['res']=s.truth-s.imp_mean; s['hot']=s['T']-s.mu1; s['mkt_dev']=s.imp_mean-s.mu1
    m=sm.OLS(s.res, sm.add_constant(s[['hot']])).fit(cov_type='HC1'); m2=sm.OLS(s.res, sm.add_constant(s[['mkt_dev']])).fit(cov_type='HC1')
    print(f'local {H}:00 N={len(s)} res~hot coef={m.params.hot:+.3f} (se {m.bse.hot:.3f}) | res~(imp_mean-mu1) coef={m2.params.mkt_dev:+.3f} (se {m2.bse.mkt_dev:.3f}) | corr(imp_mean, truth)={np.corrcoef(s.imp_mean,s.truth)[0,1]:.3f} sd res={s.res.std():.2f}')
