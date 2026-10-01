"""Final evaluation of the surviving strategy on all cities with candles: safe-timing snapshots (h=8,10,12 = D-1 22Z .. D 02Z),
13-model previous_day1 blend (weights from prior years), rolling monthly logistic combination, taker fees, fills at candle ask/bid.
Reports per station, per season, per month, capacity."""
import pandas as pd, numpy as np, statsmodels.api as sm, sys
sys.argv=['x','w','ext']
exec(open('/Users/abebraham/Desktop/resume/edge/scripts/weather_edge2.py').read().split("def run(h):")[0])
c=c[pd.to_datetime(c.target)>=pd.Timestamp('2025-01-01')]
print('\nevents 2025+ with candles+forecasts:', c.drop_duplicates('event_ticker').groupby('station').size().to_dict())
def preds_for(h, warm=120):
    s=c[c.h==h]; preds=[]
    for m in sorted(s.month.unique()):
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<warm: continue
        mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid','lpm']])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[['lmid','lpm']])), b=mod.params.lpm))
    return pd.concat(preds)
for h in [8,10,12]:
    t=preds_for(h); thr=0.05
    by=t[t.ph-t.ask>thr].copy(); bn=t[t.bid-t.ph>thr].copy(); by['pnl']=by.y-by.ask-fee(by.ask); bn['pnl']=(1-bn.y)-(1-bn.bid)-fee(bn.bid)
    sig=pd.concat([by,bn]); sig['mon']=sig.tgt.dt.month; sig['season']=np.where(sig.mon.isin([11,12,1,2,3]),'Nov-Mar',np.where(sig.mon.isin([4,5,10]),'Apr,May,Oct','Jun-Sep'))
    daily=pd.Series(0.0,index=t.target.unique()); dd=sig.groupby('target').pnl.sum(); daily.loc[dd.index]=dd.values
    print(f'\n===== h={h} (thr {thr}) OOS events={t.event_ticker.nunique()} days={len(daily)} | Brier mkt={((t.mid-t.y)**2).mean():.4f} blend={((t.pm-t.y)**2).mean():.4f} comb={((t.ph-t.y)**2).mean():.4f} lpm coef={t.b.mean():.2f}')
    print(f'ALL: N={len(sig)} pnl/ct={sig.pnl.mean():+.4f} ROI={sig.pnl.sum()/np.where(sig.ph-sig.ask>thr,sig.ask,1-sig.bid).sum()*100:+.1f}% Sharpe(all days)={daily.mean()/daily.std()*np.sqrt(365):.2f} win={ (sig.pnl>0).mean():.2f}')
    print('by station:', {k:(round(v,3),n) for k,v,n in sig.groupby('station').pnl.agg(['mean','size']).itertuples()})
    print('by season :', {k:(round(v,3),n) for k,v,n in sig.groupby('season').pnl.agg(['mean','size']).itertuples()})
    print('by station x season:'); print(sig.pivot_table(index='station',columns='season',values='pnl',aggfunc=['mean','size']).round(3).to_string())
    print('by month  :', {str(k):(round(v,3),n) for k,v,n in sig.groupby(sig.tgt.dt.to_period('M')).pnl.agg(['mean','size']).itertuples()})
    w=sig[sig.season=='Nov-Mar']; dw=pd.Series(0.0,index=t[t.tgt.dt.month.isin([11,12,1,2,3])].target.unique()); ddw=w.groupby('target').pnl.sum(); dw.loc[ddw.index]=ddw.values
    print(f'WINTER ONLY: N={len(w)} pnl/ct={w.pnl.mean():+.4f} Sharpe={dw.mean()/dw.std()*np.sqrt(365):.2f} days={len(dw)} | by station:', {k:(round(v,3),n) for k,v,n in w.groupby('station').pnl.agg(['mean','size']).itertuples()})
    nxt=c[(c.h>h)&(c.h<=h+6)].groupby('ticker').vol.sum().rename('vol6'); sig=sig.join(nxt,on='ticker').fillna({'vol6':0})
    for part,cap in [(0.10,100)]:
        sig['n']=np.minimum(cap,np.floor(part*sig.vol6)); sig['usd']=sig.n*sig.pnl; ncity=t.station.nunique()
        print(f'capacity @10%/cap100: contracts/signal={sig.n.mean():.0f}, $/city-day={sig.usd.sum()/len(daily)/ncity:+.2f} => ${sig.usd.sum()/len(daily)*365:,.0f}/yr across {ncity} cities; winter-only $/city-day={sig[sig.season=="Nov-Mar"].usd.sum()/max(1,len(dw))/ncity:+.2f}')
