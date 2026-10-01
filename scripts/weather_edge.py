import pandas as pd, numpy as np, statsmodels.api as sm, sys, glob
R='/Users/abebraham/Desktop/resume/edge/data/'
REQ_VOL=('vol' in sys.argv)
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[(c.h<=37)&(c.ask>0)&(c.ask<1)&(c.bid<1)]
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date'])
fd['err']=fd.tmax_fc-fd.high; fd['dt']=pd.to_datetime(fd.date)
# bias per station/model/lead estimated from data strictly before the target year (expanding, by year)
fd['year']=fd.dt.dt.year
bias=[]
for y in range(2022, 2027):
    b=fd[fd.year<y].groupby(['station','model','lead']).err.mean().rename('bias').reset_index(); b['year']=y; bias.append(b)
fd=fd.merge(pd.concat(bias), on=['station','model','lead','year'], how='left'); fd=fd.dropna(subset=['bias']); fd['bc']=fd.tmax_fc-fd.bias
E={}
for L in [1,2]:
    e=fd[fd.lead==L].groupby(['station','date']).agg(mu=('bc','mean'), high=('high','first'), year=('year','first')).reset_index(); e['res']=e.high-e.mu; E[L]=e
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
def pbr_vec(mu, lo, hi, res):
    out=np.empty(len(mu))
    for i,(m,l,h) in enumerate(zip(mu,lo,hi)): v=m+res; out[i]=((v>l)&(v<h)).mean()
    return out
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
c['tgt']=pd.to_datetime(c.target); c['year']=c.tgt.dt.year; c['month']=c.tgt.dt.to_period('M')
for L in [1,2]:
    c=c.merge(E[L][['station','date','mu']].rename(columns={'date':'target','mu':f'mu{L}'}), on=['station','target'], how='left')
c=c.dropna(subset=['mu1','mu2']); c['lmid']=lg(c.mid)
allev=c[['event_ticker','target','station']].drop_duplicates('event_ticker')
print('events', len(allev), allev.groupby('station').size().to_dict(), 'years', allev.target.map(lambda d: d.year).value_counts().sort_index().to_dict())
def run(h, L, thr_list=(0.03,0.05,0.08), label=''):
    s=c[c.h==h].copy()
    # residuals for probability mapping: from E[L] rows with year < target year
    out=[]
    for y,g in s.groupby('year'):
        res=E[L][(E[L].year<y)].res.values
        if len(res)<200: continue
        g=g.copy(); g['pm']=pbr_vec(g[f'mu{L}'].values, g.lo.values, g.hi.values, res); out.append(g)
    s=pd.concat(out); s['lpm']=lg(s.pm)
    # rolling monthly fit
    preds=[]
    months=sorted(s.month.unique())
    for m in months:
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<120: continue
        mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid','lpm']])).fit(disp=0)
        te=te.assign(ph=mod.predict(sm.add_constant(te[['lmid','lpm']])), b_lpm=mod.params.lpm); preds.append(te)
    if not preds: print(f'h={h} L={L}: not enough data'); return None
    t=pd.concat(preds)
    if REQ_VOL: t=t[t.vol>0]
    bm=((t.mid-t.y)**2).mean(); bc=((t.ph-t.y)**2).mean(); bmod=((t.pm-t.y)**2).mean()
    line=f'h={h:2d} L={L} {label}| OOS events={t.event_ticker.nunique()} Brier mkt={bm:.4f} model={bmod:.4f} comb={bc:.4f} mean b_lpm={t.b_lpm.mean():.2f}'
    for thr in thr_list:
        by=t[t.ph-t.ask>thr]; bn=t[t.bid-t.ph>thr]
        py=(by.y-by.ask-fee(by.ask)); pn=((1-bn.y)-(1-bn.bid)-fee(bn.bid)); allp=pd.concat([py,pn])
        daily=pd.Series(0.0, index=t.target.unique()); dd=pd.DataFrame({'t':pd.concat([by.target,bn.target]),'p':allp}).groupby('t').p.sum(); daily.loc[dd.index]=dd.values
        stake=by.ask.sum()+(1-bn.bid).sum()
        sr=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else np.nan
        line+=f'\n      thr={thr}: N={len(by)}Y+{len(bn)}N pnl/ct={allp.mean():+.4f} total=${allp.sum():+.1f} ROI={allp.sum()/max(stake,1e-9)*100:+.1f}% Sharpe(all days)={sr:.2f} days={len(daily)} warm-side share={((by.center>by.mu1).sum()+(bn.center<bn.mu1).sum())/max(1,len(allp)):.2f}'
    print(line); return t
c['center']=np.where(np.isinf(c.lo), c.hi-1, np.where(np.isinf(c.hi), c.lo+1, (c.lo+c.hi)/2))
for h in [1,2,4,8,12,14,18,20,22,24]:
    run(h,1)
print('\n=== CONTROL: strictly stale lead-2 forecast (issued ~2 days before) ===')
for h in [2,8,14,20]:
    run(h,2,label='(lead2)')
