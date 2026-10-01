import pandas as pd, numpy as np, statsmodels.api as sm, sys
exec(open('/Users/abebraham/Desktop/resume/edge/scripts/weather_edge2.py').read().split("def run(h):")[0].replace("MODE=sys.argv[1] if len(sys.argv)>1 else 'w'","MODE='w'").replace("HOURS=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [1,2,4,8,12,14,18,22]","HOURS=[]"))
print('\n=== PURE MODEL (no logistic fit): buy YES if pm-ask>thr, NO if bid-pm>thr ===')
for h in [2,8,12,14,22]:
    t=c[(c.h==h)&(c.tgt>=pd.Timestamp('2025-07-01'))]
    for thr in [0.05,0.08,0.12]:
        by=t[t.pm-t.ask>thr]; bn=t[t.bid-t.pm>thr]; allp=pd.concat([by.y-by.ask-fee(by.ask),(1-bn.y)-(1-bn.bid)-fee(bn.bid)])
        daily=pd.Series(0.0,index=t.target.unique()); dd=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum(); daily.loc[dd.index]=dd.values
        print(f'h={h} thr={thr}: N={len(by)}Y+{len(bn)}N pnl/ct={allp.mean():+.4f} ROI={allp.sum()/max(1e-9,by.ask.sum()+(1-bn.bid).sum())*100:+.1f}% Sharpe={daily.mean()/daily.std()*np.sqrt(365):.2f}')
# shrunk model: p = 0.5*mid + 0.5*pm (no fit)
print('\n=== 50/50 blend of mid and model prob, trade vs ask/bid ===')
for h in [2,8,12,14,22]:
    t=c[(c.h==h)&(c.tgt>=pd.Timestamp('2025-07-01'))]; pb=0.5*t.mid+0.5*t.pm
    for thr in [0.03,0.05]:
        by=t[pb-t.ask>thr]; bn=t[t.bid-pb>thr]; allp=pd.concat([by.y-by.ask-fee(by.ask),(1-bn.y)-(1-bn.bid)-fee(bn.bid)])
        daily=pd.Series(0.0,index=t.target.unique()); dd=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum(); daily.loc[dd.index]=dd.values
        print(f'h={h} thr={thr}: N={len(by)}Y+{len(bn)}N pnl/ct={allp.mean():+.4f} Sharpe={daily.mean()/daily.std()*np.sqrt(365):.2f}')
