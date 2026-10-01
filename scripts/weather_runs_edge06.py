"""Strictly run-time-aware backtest: for target day D use only the D-1 00Z run blend for snapshots h<=6 (open..20Z),
and the D-1 12Z run blend for h>=8 (>=22Z). Blend weights fitted on the previous-runs archive (2024-2025, lead-1) per station;
residual scale re-estimated on the 2026 run data by leave-one-month-out."""
import pandas as pd, numpy as np, statsmodels.api as sm, json, glob, re, sys
R='/Users/abebraham/Desktop/resume/edge/data/'
CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
MODELS=['gfs_seamless','ecmwf_ifs','icon_seamless','ncep_nbm_conus','ncep_hrrr_conus','gem_seamless','ukmo_seamless','meteofrance_seamless','jma_seamless','knmi_seamless','dmi_seamless','ecmwf_aifs025_single']
# ---- run-specific daily max for target day
rows=[]
for f in glob.glob(R+'weather/runs/*.json'):
    st,d,hh=re.match(r'.*/([A-Z]{3})_(\d{8})_(\d{2})\.json',f).groups(); j=json.load(open(f))
    if 'hourly' not in j: continue
    h=j['hourly']; t=pd.to_datetime(h['time']); target=pd.Timestamp(d).date()
    sel=[i for i,x in enumerate(t) if x.date()==target]
    rec={'station':st,'target':target,'run':hh}
    for k,v in h.items():
        if k=='time': continue
        vals=[v[i] for i in sel if v[i] is not None]
        if len(vals)>=20: rec[k.replace('temperature_2m_','')]=max(vals)
    rows.append(rec)
RF=pd.DataFrame(rows); print('run forecasts', len(RF), RF.groupby('run').size().to_dict(), 'stations', RF.station.nunique())
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); RF['cli']=RF.station.map(CLI); RF=RF.merge(cli[['station','date','high']].rename(columns={'station':'cli','date':'target'}), on=['cli','target'])
for m in MODELS:
    if m in RF: print(f'  {m}: coverage {RF[m].notna().mean():.2f}  MAE 00Z {(RF[RF.run=="00"][m]-RF[RF.run=="00"].high).abs().mean():.2f}  12Z {(RF[RF.run=="12"][m]-RF[RF.run=="12"].high).abs().mean():.2f}')
# ---- blend weights from previous-runs archive (lead-1, 2024-2025), per station, same model set
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date; fd['cli']=fd.station.map(CLI)
fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date']); fd['year']=pd.to_datetime(fd.date).dt.year
P=fd[(fd.lead==1)&(fd.year<=2025)].pivot_table(index=['station','date','high'],columns='model',values='tmax_fc').reset_index()
def addseas(d): d=d.copy(); doy=pd.to_datetime(d['target' if 'target' in d else 'date']).dt.dayofyear; d['s1']=np.sin(2*np.pi*doy/365); d['c1']=np.cos(2*np.pi*doy/365); return d
def design(d, cols):
    X=d[cols].copy(); X=X.apply(lambda r: r.fillna(r.mean()), axis=1); X['s1']=d.s1; X['c1']=d.c1; return sm.add_constant(X, has_constant='add')
P=addseas(P); RF=addseas(RF)
FIT={}
for st,g in P.groupby('station'):
    g=g[g[[m for m in MODELS if m in g]].notna().sum(axis=1)>=4]
    cols=[m for m in MODELS if m in g and m in RF.columns and g[m].notna().mean()>0.6]; tr=g.dropna(subset=['high']).dropna(subset=cols, thresh=int(0.7*len(cols)))
    X=design(tr,cols); ok=X.notna().all(axis=1); m=sm.OLS(tr.high[ok], X[ok]).fit(); FIT[st]=(m,cols); print(f'  blend {st}: {len(cols)} models, {ok.sum()} training days, resid sd {np.std(tr.high[ok]-m.predict(X[ok])):.2f}')
RF['mu']=np.nan
for st,(m,cols) in FIT.items():
    idx=RF.station==st; sub=RF[idx]
    ok=sub[cols].notna().sum(axis=1)>=max(3,int(0.6*len(cols)))
    RF.loc[sub[ok].index,'mu']=m.predict(design(sub[ok],cols)).values
RF=RF.dropna(subset=['mu','high']); RF['err']=RF.high-RF.mu; RF['month']=pd.to_datetime(RF.target).dt.to_period('M')
print('blend MAE by run:', RF.groupby('run').err.apply(lambda s: s.abs().mean()).round(2).to_dict(), ' by station (12Z):', RF[RF.run=='12'].groupby('station').err.apply(lambda s: s.abs().mean()).round(2).to_dict())
# ---- market snapshots
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[(c.h<=37)&(c.ask>0)&(c.ask<1)&(c.bid<1)]
c=c[pd.to_datetime(c.target)>=pd.Timestamp('2026-04-02')]
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
c['lmid']=lg(c.mid); c['month']=pd.to_datetime(c.target).dt.to_period('M')
def pm_col(s, run):
    f=RF[RF.run==run][['station','target','mu','err','month']]
    s=s.merge(f[['station','target','mu']], on=['station','target'])
    pm=np.empty(len(s))
    for i,(st,mo,mu,lo,hi) in enumerate(zip(s.station,s.month,s.mu,s.lo,s.hi)):
        res=f[(f.month!=mo)].err.values   # leave-one-month-out residuals (pooled stations)
        v=mu+res; pm[i]=((v>lo)&(v<hi)).mean()
    s['pm']=pm; s['lpm']=lg(s.pm); return s
def run_h(h, run, coefs=(1.05,0.45)):
    s=pm_col(c[c.h==h], run)
    # fixed logistic coefficients from prior NYC backtest (no fitting on this sample), plus a refit check
    b1,b2=coefs; s['ph']=1/(1+np.exp(-(b1*s.lmid+b2*s.lpm)))
    mod=sm.Logit(s.y, sm.add_constant(s[['lmid','lpm']])).fit(disp=0)
    print(f'\nh={h:2d} run={run}Z events={s.event_ticker.nunique()} Brier mkt={((s.mid-s.y)**2).mean():.4f} model={((s.pm-s.y)**2).mean():.4f} comb(fixed)={((s.ph-s.y)**2).mean():.4f} | in-sample refit lpm coef={mod.params.lpm:.2f} (z={mod.tvalues.lpm:.1f})')
    for thr in [0.03,0.05,0.08]:
        by=s[s.ph-s.ask>thr]; bn=s[s.bid-s.ph>thr]; allp=pd.concat([by.y-by.ask-fee(by.ask),(1-bn.y)-(1-bn.bid)-fee(bn.bid)])
        daily=pd.Series(0.0,index=s.target.unique()); dd=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum(); daily.loc[dd.index]=dd.values
        bys=pd.concat([by.assign(p=by.y-by.ask-fee(by.ask)),bn.assign(p=(1-bn.y)-(1-bn.bid)-fee(bn.bid))])
        print(f'   thr={thr}: N={len(by)}Y+{len(bn)}N pnl/ct={allp.mean():+.4f} ROI={allp.sum()/max(1e-9,by.ask.sum()+(1-bn.bid).sum())*100:+.1f}% Sharpe={daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else np.nan:.2f} | by station: '+', '.join(f'{k}:{v:+.3f}(n={n})' for k,v,n in bys.groupby('station').p.agg(['mean','size']).itertuples())+' | by month: '+', '.join(f'{k}:{v:+.3f}' for k,v in bys.groupby('month').p.mean().items()))
for h in [1,2,4,6]: run_h(h,'06')
pass
