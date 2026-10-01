import pandas as pd, numpy as np, statsmodels.api as sm, sys
sys.argv=['x','w','ext']
exec(open('/Users/abebraham/Desktop/resume/edge/scripts/weather_edge2.py').read().split("def run(h):")[0])
c=c[pd.to_datetime(c.target)>=pd.Timestamp('2025-01-01')].copy()
def preds_for(h, col='lpm'):
    s=c[c.h==h]; preds=[]
    for m in sorted(s.month.unique()):
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<120: continue
        mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid',col]])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[['lmid',col]])), b=mod.params[col]))
    return pd.concat(preds)
def evaluate(t, label, thr=0.05, fill='close', delay=0):
    # fill: 'close' = ask/bid close of signal hour; 'worst' = next-hour ask_high / bid_low, require next-hour volume>0
    if delay or fill=='worst':
        nxt=c[c.h==t.h.iloc[0]+max(1,delay)][['ticker','ask','bid','vol']].rename(columns={'ask':'a1','bid':'b1','vol':'v1'}); t=t.merge(nxt,on='ticker')
        if fill=='worst': t=t[t.v1>0]; pa,pb=np.minimum(0.99,t.a1+0.01),np.maximum(0.01,t.b1-0.01)
        else: pa,pb=t.a1,t.b1
    else: pa,pb=t.ask,t.bid
    by=t[t.ph-t.ask>thr]; bn=t[t.bid-t.ph>thr]
    py=by.y-pa[by.index]-fee(pa[by.index]); pn=(1-bn.y)-(1-pb[bn.index])-fee(pb[bn.index]); allp=pd.concat([py,pn])
    daily=pd.Series(0.0,index=t.target.unique()); dd=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum(); daily.loc[dd.index]=dd.values
    print(f'{label:58s} N={len(allp):5d} pnl/ct={allp.mean():+.4f} sd/trade={allp.std():.3f} t-stat={allp.mean()/allp.std()*np.sqrt(len(allp)):5.1f} win={(allp>0).mean():.2f} dailySharpe={daily.mean()/daily.std()*np.sqrt(365):.2f}')
    return allp
t8=preds_for(8)
print('=== STRESS TESTS, Kalshi 18 cities, D-1 22Z, thr .05 ===')
evaluate(t8,'baseline: fill at signal-hour ask/bid close')
evaluate(t8,'1-hour execution delay: fill at next-hour close',delay=1)
evaluate(t8,'WORST fill: next-hour close +1c slippage, need next-hour volume>0',fill='worst')
evaluate(t8,'2-hour delay',delay=2)
print('\n=== temporal halves (same walk-forward, just split the OOS period) ===')
mid=pd.Timestamp('2025-12-01')
evaluate(t8[t8.tgt<mid],'OOS Jan-Nov 2025 only'); evaluate(t8[t8.tgt>=mid],'OOS Dec 2025-Sep 2026 only')
print('\n=== PLACEBOS (should be ~zero) ===')
# placebo 1: shuffle blend across events within station (destroys day-specific info, keeps climatology)
rng=np.random.default_rng(0); cp=c.copy()
for st,g in cp.groupby('station'):
    ev=g.event_ticker.unique(); perm=dict(zip(ev, rng.permutation(ev)))
    mu_map=g.drop_duplicates('event_ticker').set_index('event_ticker').mu
    cp.loc[g.index,'mu']=g.event_ticker.map(lambda e: mu_map[perm[e]]).values
pm=np.empty(len(cp))
for i,(m,l,h,k) in enumerate(zip(cp.mu,cp.lo,cp.hi,cp.res_key)): v=m+RES[k]; pm[i]=((v>l)&(v<h)).mean()
cp['lpm_sh']=lg(pm); c_backup=c; c=cp; tp=preds_for(8,'lpm_sh'); print(f'  shuffled-day blend: lpm coef={tp.b.mean():.2f}'); evaluate(tp,'placebo A: blend shuffled across days (same station)'); c=c_backup
# placebo 2: lead-2 (stale) blend was already ~0 (see summary). placebo 3: climatology-only prob (no forecast): use station monthly mean high as mu
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); cli['mon']=pd.to_datetime(cli.date).dt.month; clim=cli[pd.to_datetime(cli.date)<pd.Timestamp('2025-01-01')].groupby(['station','mon']).high.mean()
CLIm={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL','ATL':'KATL','BOS':'KBOS','DFW':'KDFW','DCA':'KDCA','HOU':'KHOU','LAS':'KLAS','MSP':'KMSP','OKC':'KOKC','PHX':'KPHX','SEA':'KSEA','SFO':'KSFO'}
cq=c.copy(); cq['mu']=[clim.get((CLIm[s],m),np.nan) for s,m in zip(cq.station,cq.tgt.dt.month)]; cq=cq.dropna(subset=['mu'])
pm=np.empty(len(cq))
for i,(m,l,h,k) in enumerate(zip(cq.mu,cq.lo,cq.hi,cq.res_key)): v=m+RES[k]*3; pm[i]=((v>l)&(v<h)).mean()
cq['lpm_cl']=lg(pm); c=cq; tq=preds_for(8,'lpm_cl'); print(f'  climatology blend: lpm coef={tq.b.mean():.2f}'); evaluate(tq,'placebo B: climatology instead of forecast'); c=c_backup
print('\n=== per-trade honesty: why daily Sharpe is high ===')
allp=evaluate(t8,'baseline again')
print(f'  trades/day={len(allp)/t8.target.nunique():.1f}; per-trade mean/sd={allp.mean()/allp.std():.3f}; sqrt(trades/day)*that*sqrt(365)~{allp.mean()/allp.std()*np.sqrt(len(allp)/t8.target.nunique()*365):.1f} (if trades were independent) vs realised dailySharpe above (correlation across cities/brackets lowers it)')
