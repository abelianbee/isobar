import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/'
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[(c.h<=37)&(c.ask>0)&(c.ask<1)&(c.bid<1)]
ev=pd.read_parquet(R+'weather/events.parquet'); snap=pd.read_parquet(R+'weather/snap_obs.parquet')
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date'])
fd['train']=pd.to_datetime(fd.date)<pd.Timestamp('2025-01-01'); fd['err']=fd.tmax_fc-fd.high
bias=fd[fd.train].groupby(['station','model','lead']).err.mean().rename('bias').reset_index(); fd=fd.merge(bias,on=['station','model','lead']); fd['bc']=fd.tmax_fc-fd.bias
E=fd[fd.lead==1].groupby(['station','date']).agg(mu1=('bc','mean'), high=('high','first'), train=('train','first')).reset_index()
res1=(E[E.train].high-E[E.train].mu1).values
def pbr(mu,lo,hi,res): v=mu+res; return ((v>lo)&(v<hi)).mean()
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
c=c.merge(E[['station','date','mu1']].rename(columns={'date':'target'}), on=['station','target'])
c['lmid']=np.log(np.clip(c.mid,0.01,0.99)/(1-np.clip(c.mid,0.01,0.99)))
c['tgt']=pd.to_datetime(c.target); c['split']=np.where(c.tgt<pd.Timestamp('2026-01-01'),'fit','test')
print('events fit/test:', c[c.split=='fit'].event_ticker.nunique(), c[c.split=='test'].event_ticker.nunique())
for h in [2,8,14,20]:
    s=c[c.h==h].copy(); s['pm']=[pbr(m,l,hh,res1) for m,l,hh in zip(s.mu1,s.lo,s.hi)]; s['lpm']=np.log(np.clip(s.pm,0.005,0.995)/(1-np.clip(s.pm,0.005,0.995)))
    f=s[s.split=='fit']; t=s[s.split=='test']
    m=sm.Logit(f.y, sm.add_constant(f[['lmid','lpm']])).fit(disp=0)
    t=t.assign(ph=m.predict(sm.add_constant(t[['lmid','lpm']])))
    bm=((t.mid-t.y)**2).mean(); bc=((t.ph-t.y)**2).mean()
    print(f'\nh={h}: fit coefs lmid={m.params.lmid:.3f} lpm={m.params.lpm:.3f} (z={m.tvalues.lpm:.1f}) | TEST Brier market={bm:.4f} combined={bc:.4f} N={len(t)}')
    for thr in [0.02,0.04,0.06]:
        by=t[t.ph-t.ask>thr]; bn=t[t.bid-t.ph>thr]
        py=(by.y-by.ask-fee(by.ask)); pn=((1-bn.y)-(1-bn.bid)-fee(bn.bid)); allp=pd.concat([py,pn])
        d=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum()
        print(f'   thr={thr}: N={len(by)}+{len(bn)} pnl/ct={allp.mean():+.4f} total={allp.sum():+.1f} Sharpe={d.mean()/d.std()*np.sqrt(365) if len(d)>2 else np.nan:.2f} days={len(d)}')
# day D with obs: local hours 9-13
print('\n=== day D: market + obs-model (mu1, T, RM) ===')
c['ltime']=pd.to_datetime(c.ts,unit='s',utc=True); c['lhour']=[t.tz_convert(tz).hour for t,tz in zip(c.ltime,c.tz)]; c['ldate']=[t.tz_convert(tz).date() for t,tz in zip(c.ltime,c.tz)]
D=c[c.ldate==c.target]
for H in [9,11,13,15]:
    s=D[D.lhour==H].merge(snap[snap.H==H][['station','date','T','RM']].rename(columns={'date':'target'}), on=['station','target'])
    # obs model fit on 2021-2024 (no market): high ~ mu1 + T + RM per station
    tr=snap[(snap.H==H)&snap.train&(snap.station=='NYC')].dropna(subset=['high']); om=sm.OLS(tr.high, sm.add_constant(tr[['mu1','T','RM']])).fit(); rs=(tr.high-om.predict(sm.add_constant(tr[['mu1','T','RM']]))).values
    s['muo']=om.predict(sm.add_constant(s[['mu1','T','RM']])); s['po']=[pbr(m,l,hh,rs) for m,l,hh in zip(s.muo,s.lo,s.hi)]
    s['lpo']=np.log(np.clip(s.po,0.005,0.995)/(1-np.clip(s.po,0.005,0.995)))
    f=s[s.split=='fit']; t=s[s.split=='test']
    m=sm.Logit(f.y, sm.add_constant(f[['lmid','lpo']])).fit(disp=0); t=t.assign(ph=m.predict(sm.add_constant(t[['lmid','lpo']])))
    print(f'local {H}: obs-model sigma={rs.std():.2f} | fit lmid={m.params.lmid:.3f} lpo={m.params.lpo:.3f} (z={m.tvalues.lpo:.1f}) | TEST Brier mkt={((t.mid-t.y)**2).mean():.4f} obs-only={((t.po-t.y)**2).mean():.4f} comb={((t.ph-t.y)**2).mean():.4f} N={len(t)}')
    for thr in [0.03,0.06]:
        by=t[t.ph-t.ask>thr]; bn=t[t.bid-t.ph>thr]; py=(by.y-by.ask-fee(by.ask)); pn=((1-bn.y)-(1-bn.bid)-fee(bn.bid)); allp=pd.concat([py,pn])
        print(f'   thr={thr}: N={len(by)}+{len(bn)} pnl/ct={allp.mean():+.4f} total={allp.sum():+.1f}')
