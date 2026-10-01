import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/'
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[c.h<=37]
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
base=c[c.h==2][['ticker','mid']].rename(columns={'mid':'m0'})
print('--- bracket-level: (y - mid_h) ~ (mid_h - mid_2), brackets with mid in [0.1,0.9] ---')
for h in [8,14,20,24,26,28,30]:
    s=c[c.h==h].merge(base,on='ticker'); s=s[(s.mid>0.1)&(s.mid<0.9)]; s['d']=s.mid-s.m0; s['res']=s.y-s.mid
    m=sm.OLS(s.res, sm.add_constant(s.d)).fit(cov_type='cluster', cov_kwds={'groups':pd.factorize(s.event_ticker)[0]})
    print(f'h={h}: coef={m.params.iloc[1]:+.3f} (se {m.bse.iloc[1]:.3f}) N={len(s)} mean res={s.res.mean():+.3f}')
    # trade: buy YES at ask on brackets whose mid rose >0.15 since open; and buy NO on brackets whose mid fell >0.15
    up=s[s.d>0.15]; dn=s[s.d<-0.15]
    pu=up.y-up.ask-fee(up.ask); pdn=(1-dn.y)-(1-dn.bid)-fee(dn.bid)
    print(f'     risers>+.15: N={len(up)} mid={up.mid.mean():.3f} hit={up.y.mean():.3f} buyYES@ask pnl/ct={pu.mean():+.4f} | fallers<-.15: N={len(dn)} mid={dn.mid.mean():.3f} hit={dn.y.mean():.3f} buyNO@bid pnl/ct={pdn.mean():+.4f}')
# lagged 6h change
print('--- (y - mid_h) ~ (mid_h - mid_{h-6}) ---')
for h in [14,20,24,26,28,30]:
    s=c[c.h==h].merge(c[c.h==h-6][['ticker','mid']].rename(columns={'mid':'ml'}),on='ticker'); s=s[(s.mid>0.1)&(s.mid<0.9)]; s['d']=s.mid-s.ml; s['res']=s.y-s.mid
    m=sm.OLS(s.res, sm.add_constant(s.d)).fit(cov_type='cluster', cov_kwds={'groups':pd.factorize(s.event_ticker)[0]})
    print(f'h={h}: coef={m.params.iloc[1]:+.3f} (se {m.bse.iloc[1]:.3f}) N={len(s)}')
