import pandas as pd, numpy as np, statsmodels.api as sm, sys
R='/Users/abebraham/Desktop/resume/edge/data/'
MODE=sys.argv[1] if len(sys.argv)>1 else 'w'   # 'w' weighted ensemble, 'bc' equal-weight bias-corrected
HOURS=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 and sys.argv[2][0].isdigit() else [2,8,12,14,22]
KIND='low' if 'low' in sys.argv else 'high'
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[(c.h<=37)&(c.ask>0)&(c.ask<1)&(c.bid<1)]
if 'kind' in c: c=c[c.kind==KIND]
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL','ATL':'KATL','BOS':'KBOS','DFW':'KDFW','DCA':'KDCA','HOU':'KHOU','LAS':'KLAS','MSP':'KMSP','OKC':'KOKC','PHX':'KPHX','SEA':'KSEA','SFO':'KSFO'}
if KIND=='low': fd['tmax_fc']=fd['tmin_fc']; cli['high']=cli['low']
fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date']); fd['year']=pd.to_datetime(fd.date).dt.year
ALLM=sorted(fd[fd.lead==1].model.unique()); EXT=('ext' in sys.argv)
MODELS=[m for m in ALLM if EXT or m in ['gfs_seamless','ecmwf_ifs025','icon_seamless','ncep_hrrr_conus','ncep_nbm_conus','gem_seamless','ukmo_seamless']]
print('models:', MODELS)
LEAD=int([a for a in sys.argv if a.startswith('lead=')][0][5:]) if any(a.startswith('lead=') for a in sys.argv) else 1
print('LEAD', LEAD)
P=fd[fd.lead==LEAD].pivot_table(index=['station','date','year','high'],columns='model',values='tmax_fc').reset_index()
MODELS=[m for m in MODELS if m in P.columns]
P['doy']=pd.to_datetime(P.date).dt.dayofyear; P['s1']=np.sin(2*np.pi*P.doy/365); P['c1']=np.cos(2*np.pi*P.doy/365)
# forecast combination per station, trained on years < y (fill missing models with row mean of others)
def design(d, cols):
    X=d[cols].copy(); X=X.apply(lambda r: r.fillna(r.mean()), axis=1); X['s1']=d.s1; X['c1']=d.c1; return sm.add_constant(X, has_constant='add')
rows=[]
for st,g in P.groupby('station'):
    for y in range(2023,2027):
        tr0=g[g.year<y]; te=g[g.year==y]
        tr=tr0[tr0[MODELS].notna().sum(axis=1)>=4]
        cols=[m for m in MODELS if tr[m].notna().mean()>0.6] if len(tr)>=250 else []
        if len(tr)<250 or len(cols)<3:   # fallback: GFS+HRRR only (archive since 2021)
            cols=[m for m in ['gfs_seamless','ncep_hrrr_conus'] if m in tr0 and tr0[m].notna().mean()>0.6]; tr=tr0.dropna(subset=cols)
        else:
            tr=tr.dropna(subset=cols, thresh=max(3,int(0.7*len(cols))))
        if len(tr)<250 or len(te)==0: continue
        if MODE=='w':
            m=sm.OLS(tr.high, design(tr,cols)).fit(); mu=m.predict(design(te,cols)); res=(tr.high-m.predict(design(tr,cols))).values
        else:
            bias=(tr[cols].sub(tr.high,axis=0)).mean(); mu=te[cols].sub(bias).mean(axis=1); res=(tr.high-tr[cols].sub(bias).mean(axis=1)).values
        rows.append(pd.DataFrame({'station':st,'target':te.date.values,'mu':mu.values,'sig_tr':res.std(),'year':y,'res_key':st+str(y),'ncols':len(cols)}))
        globals().setdefault('RES',{})[st+str(y)]=res
M=pd.concat(rows); M['err']=M.merge(P[['station','date','high']].rename(columns={'date':'target'}),on=['station','target']).eval('high-mu').values
print(f'MODE={MODE}: forecast OOS MAE by station/year:'); print(M.groupby(['station','year']).err.apply(lambda s: s.abs().mean()).unstack().round(2).to_string()); print('n models used:', M.groupby(['station','year']).ncols.first().unstack().to_string())
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
c=c.merge(M[['station','target','mu','res_key']],on=['station','target']); c['tgt']=pd.to_datetime(c.target); c['month']=c.tgt.dt.to_period('M'); c['lmid']=lg(c.mid)
c['center']=np.where(np.isinf(c.lo), c.hi-1, np.where(np.isinf(c.hi), c.lo+1, (c.lo+c.hi)/2))
pm=np.empty(len(c))
for i,(m,l,h,k) in enumerate(zip(c.mu,c.lo,c.hi,c.res_key)): v=m+RES[k]; pm[i]=((v>l)&(v<h)).mean()
c['pm']=pm; c['lpm']=lg(c.pm)
print('events', c.event_ticker.nunique(), c.drop_duplicates('event_ticker').groupby('station').size().to_dict())
def run(h):
    s=c[c.h==h]; preds=[]
    for m in sorted(s.month.unique()):
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<120: continue
        mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid','lpm']])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[['lmid','lpm']])), b=mod.params.lpm))
    if not preds: return
    t=pd.concat(preds)
    print(f'\nh={h:2d} OOS events={t.event_ticker.nunique()} Brier mkt={((t.mid-t.y)**2).mean():.4f} model={((t.pm-t.y)**2).mean():.4f} comb={((t.ph-t.y)**2).mean():.4f} b_lpm={t.b.mean():.2f}')
    for thr in [0.03,0.05,0.08]:
        by=t[t.ph-t.ask>thr]; bn=t[t.bid-t.ph>thr]; py=(by.y-by.ask-fee(by.ask)); pn=((1-bn.y)-(1-bn.bid)-fee(bn.bid)); allp=pd.concat([py,pn])
        daily=pd.Series(0.0,index=t.target.unique()); dd=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum(); daily.loc[dd.index]=dd.values
        stake=by.ask.sum()+(1-bn.bid).sum(); sr=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else np.nan
        vols=pd.concat([by.vol,bn.vol]); ois=pd.concat([by.oi,bn.oi])
        bys=t.assign(sig=np.where(t.ph-t.ask>thr,1,np.where(t.bid-t.ph>thr,-1,0)))
        print(f'   thr={thr}: N={len(by)}Y+{len(bn)}N pnl/ct={allp.mean():+.4f} ROI={allp.sum()/max(stake,1e-9)*100:+.1f}% Sharpe={sr:.2f} | by station: '+', '.join(f"{k}:{v:+.3f}(n={n})" for k,v,n in bys[bys.sig!=0].assign(p=np.where(bys[bys.sig!=0].sig==1, bys[bys.sig!=0].y-bys[bys.sig!=0].ask-fee(bys[bys.sig!=0].ask), (1-bys[bys.sig!=0].y)-(1-bys[bys.sig!=0].bid)-fee(bys[bys.sig!=0].bid))).groupby('station').p.agg(['mean','size']).itertuples()) + f' | hourly vol at signal: median={vols.median():.0f} p25={vols.quantile(.25):.0f} zero-share={(vols==0).mean():.2f}; OI median={ois.median():.0f}')
        yr=bys[bys.sig!=0].assign(p=np.where(bys[bys.sig!=0].sig==1, bys[bys.sig!=0].y-bys[bys.sig!=0].ask-fee(bys[bys.sig!=0].ask), (1-bys[bys.sig!=0].y)-(1-bys[bys.sig!=0].bid)-fee(bys[bys.sig!=0].bid))).groupby(bys[bys.sig!=0].tgt.dt.to_period('Q')).p.agg(['mean','size'])
        if thr==0.05: print('      by quarter:', {str(k):(round(v,3),n) for k,v,n in yr.itertuples()})
for h in HOURS: run(h)
