import pandas as pd, numpy as np, statsmodels.api as sm, sys
exec(open('/Users/abebraham/Desktop/resume/edge/scripts/weather_edge2.py').read().split("def run(h):")[0].replace("MODE=sys.argv[1] if len(sys.argv)>1 else 'w'","MODE='w'").replace("HOURS=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [1,2,4,8,12,14,18,22]","HOURS=[2,8]"))
def preds_for(h):
    s=c[c.h==h]; preds=[]
    for m in sorted(s.month.unique()):
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<120: continue
        mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid','lpm']])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[['lmid','lpm']]))))
    return pd.concat(preds)
for h in [2,8]:
    t=preds_for(h); thr=0.05
    by=t[t.ph-t.ask>thr].copy(); bn=t[t.bid-t.ph>thr].copy()
    by['pnl']=by.y-by.ask-fee(by.ask); bn['pnl']=(1-bn.y)-(1-bn.bid)-fee(bn.bid)
    print(f'\n=== h={h} thr={thr} ===')
    print(f'YES side: N={len(by)} avg ask={by.ask.mean():.3f} avg ph={by.ph.mean():.3f} hit={by.y.mean():.3f} pnl/ct={by.pnl.mean():+.4f} | edge buckets:', by.assign(e=pd.cut(by.ph-by.ask,[0.05,0.08,0.12,0.2,1])).groupby('e',observed=True).pnl.agg(['mean','size']).round(3).to_dict('index'))
    print(f'NO  side: N={len(bn)} avg bid={bn.bid.mean():.3f} avg ph={bn.ph.mean():.3f} hit={bn.y.mean():.3f} pnl/ct={bn.pnl.mean():+.4f} | edge buckets:', bn.assign(e=pd.cut(bn.bid-bn.ph,[0.05,0.08,0.12,0.2,1])).groupby('e',observed=True).pnl.agg(['mean','size']).round(3).to_dict('index'))
    print('YES by price:', by.assign(pb=pd.cut(by.ask,[0,0.1,0.2,0.35,0.5,0.7,1])).groupby('pb',observed=True).pnl.agg(['mean','size']).round(3).to_dict('index'))
    print('NO by price (bid):', bn.assign(pb=pd.cut(bn.bid,[0,0.1,0.2,0.35,0.5,0.7,1])).groupby('pb',observed=True).pnl.agg(['mean','size']).round(3).to_dict('index'))
    # worse fills: +1c and +2c
    for slip in [0.01,0.02]:
        p2=pd.concat([by.y-(by.ask+slip)-fee(by.ask+slip), (1-bn.y)-(1-(bn.bid-slip))-fee(bn.bid-slip)]); print(f'slippage {slip*100:.0f}c: pnl/ct={p2.mean():+.4f} total={p2.sum():+.1f}')
    # strike type of traded brackets
    print('YES by strike type:', by.groupby('strike_type').pnl.agg(['mean','size']).round(3).to_dict('index'), ' NO by strike type:', bn.groupby('strike_type').pnl.agg(['mean','size']).round(3).to_dict('index'))
    # capacity: volume in signaled bracket over the next 6 hours
    nxt=c[(c.h>h)&(c.h<=h+6)].groupby('ticker').vol.sum().rename('vol6')
    sig=pd.concat([by,bn]).join(nxt,on='ticker'); print(f'volume in signalled bracket over next 6h: median={sig.vol6.median():.0f} p25={sig.vol6.quantile(.25):.0f} p75={sig.vol6.quantile(.75):.0f}; contracts/day signalled={len(sig)/t.event_ticker.nunique():.2f}')
    # concentration: share of total pnl from top 10% trades
    allp=pd.concat([by.pnl,bn.pnl]).sort_values(ascending=False); print(f'top-10% trades share of total pnl: {allp.head(int(len(allp)*0.1)).sum()/allp.sum():.2f}; win rate={(allp>0).mean():.3f}; median pnl={allp.median():+.3f}')
