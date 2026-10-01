import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/'
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[(c.h<=37)&(c.ask>0)&(c.ask<1)&(c.bid<1)]
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date']); fd['err']=fd.tmax_fc-fd.high; fd['year']=pd.to_datetime(fd.date).dt.year
b=fd[fd.year<2025].groupby(['station','model','lead']).err.mean().rename('bias').reset_index(); fd=fd.merge(b,on=['station','model','lead'])
e1=fd[fd.lead==1].groupby(['station','date']).agg(mu_raw=('tmax_fc','mean'), mu_bc=(('tmax_fc'),'mean'), high=('high','first')).reset_index()
e1['mu_bc']=fd[fd.lead==1].assign(bc=fd.tmax_fc-fd.bias).groupby(['station','date']).bc.mean().values
# per-model too
pm=fd[fd.lead==1].pivot_table(index=['station','date'],columns='model',values='tmax_fc').reset_index()
c=c.merge(e1.rename(columns={'date':'target'}),on=['station','target']).merge(pm.rename(columns={'date':'target'}),on=['station','target'],how='left')
c['center']=np.where(np.isinf(c.lo), c.hi-1, np.where(np.isinf(c.hi), c.lo+1, (c.lo+c.hi)/2))
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
# implied mean via mids (crude) and residual
h=2; s=c[c.h==h]
g=s.groupby('event_ticker').apply(lambda d: pd.Series({'imp':(d.mid*d.center).sum()/d.mid.sum(),'truth':d.truth.iloc[0],'mu_raw':d.mu_raw.iloc[0],'mu_bc':d.mu_bc.iloc[0],'year':pd.Timestamp(d.target.iloc[0]).year}))
print(f'h={h}: mean(truth-imp)={ (g.truth-g.imp).mean():+.2f}  mean(truth-mu_raw)={(g.truth-g.mu_raw).mean():+.2f}  mean(truth-mu_bc)={(g.truth-g.mu_bc).mean():+.2f}  mean(mu_bc-imp)={(g.mu_bc-g.imp).mean():+.2f}')
print('   by year:', g.groupby('year').apply(lambda d: pd.Series({'truth-imp':(d.truth-d.imp).mean(),'mu_bc-imp':(d.mu_bc-d.imp).mean(),'n':len(d)})).round(2).to_dict('index'))
print('   corr(truth-imp, mu_bc-imp)=', np.corrcoef(g.truth-g.imp, g.mu_bc-g.imp)[0,1].round(3), ' OLS slope:', sm.OLS(g.truth-g.imp, sm.add_constant(g.mu_bc-g.imp)).fit().params.iloc[1].round(3))
# ---- simple rule, no fitting: at hour h, dev = mu_bc - imp. If dev>thr: buy YES on brackets with center>imp+? ... simpler: buy the bracket containing mu_bc if its ask < model prob? Instead: buy YES on bracket containing mu_bc when dev>thr ; buy NO on favourite when |dev|>thr
for h in [2,8,14]:
    s=c[c.h==h].copy(); imp=s.groupby('event_ticker').apply(lambda d:(d.mid*d.center).sum()/d.mid.sum()).rename('imp'); s=s.merge(imp,on='event_ticker'); s['dev']=s.mu_bc-s.imp
    for thr in [1.0,1.5,2.0]:
        t=s[(s.dev.abs()>thr)]
        # bracket containing mu_bc
        cont=t[(t.mu_bc>t.lo)&(t.mu_bc<t.hi)]; py=cont.y-cont.ask-fee(cont.ask)
        # favourite bracket (highest mid) -> buy NO
        fav=t.sort_values('mid').groupby('event_ticker').last(); pn=(1-fav.y)-(1-fav.bid)-fee(fav.bid)
        print(f'h={h} |mu_bc-imp|>{thr}: events={t.event_ticker.nunique()} | buy YES bracket containing mu_bc: N={len(cont)} ask={cont.ask.mean():.3f} hit={cont.y.mean():.3f} pnl/ct={py.mean():+.4f} | buy NO on market favourite: N={len(fav)} bid={fav.bid.mean():.3f} hit={fav.y.mean():.3f} pnl/ct={pn.mean():+.4f}')
# ---- which models matter? per-model bias-corrected deviation from imp, regress residual
s=c[c.h==2].copy(); imp=s.groupby('event_ticker').apply(lambda d:(d.mid*d.center).sum()/d.mid.sum()).rename('imp'); s=s.merge(imp,on='event_ticker').drop_duplicates('event_ticker')
models=[m for m in ['gfs_seamless','ecmwf_ifs025','icon_seamless','ncep_hrrr_conus','ncep_nbm_conus','gem_seamless','ukmo_seamless'] if m in s]
X=s[models].sub(s.imp,axis=0); X=X.dropna(); y=(s.truth-s.imp).loc[X.index]
print('\nresidual (truth-imp) ~ (model - imp) per model, h=2, N=',len(X)); print(sm.OLS(y, sm.add_constant(X)).fit().summary().tables[1])
for m in models:
    xx=(s[m]-s.imp).dropna(); print(f'  {m}: N={len(xx)} mean(model-imp)={xx.mean():+.2f} corr with residual={np.corrcoef(xx,(s.truth-s.imp).loc[xx.index])[0,1]:+.3f}')
