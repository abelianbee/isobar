import pandas as pd, numpy as np, glob
from scipy.stats import norm
R='/Users/abebraham/Desktop/resume/edge/data/'
ev=pd.read_parquet(R+'weather/events.parquet'); b=pd.read_parquet(R+'kalshi/brackets.parquet'); c=pd.read_parquet(R+'kalshi/candles_long.parquet')
ev['truth']=ev.cli_high
x=b.merge(ev[['event_ticker','truth']], on='event_ticker').dropna(subset=['truth']); x['py']=(x.truth>x.lo)&(x.truth<x.hi)
print('bracket result vs CLI-high consistency:', ((x.result=='yes')==x.py).mean().round(4), 'N', len(x))
bad=x[(x.result=='yes')!=x.py]; print('inconsistent events', bad.event_ticker.nunique()); good_ev=set(x.event_ticker)-set(bad.event_ticker); ev=ev[ev.event_ticker.isin(good_ev)]; print(bad.groupby(bad.event_ticker.str.slice(0,10)).size().head(10).to_dict())
# ---- forecast model: bias-correct per station/model/lead using rolling past (train on days < 2025-01-01 for a fixed correction), ensemble mean, sigma by lead
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); m={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
fd['cli']=fd.station.map(m); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date'])
fd['err']=fd.tmax_fc-fd.high; fd['train']=pd.to_datetime(fd.date)<pd.Timestamp('2025-01-01')
bias=fd[fd.train].groupby(['station','model','lead']).err.mean().rename('bias').reset_index()
fd=fd.merge(bias, on=['station','model','lead']); fd['fc_bc']=fd.tmax_fc-fd.bias
MODELS_L1=['gfs_seamless','icon_seamless','gem_seamless','ncep_hrrr_conus','ncep_nbm_conus','ecmwf_ifs025','ukmo_seamless']
def ens(lead, models):
    s=fd[(fd.lead==lead)&fd.model.isin(models)].groupby(['station','date']).agg(mu=('fc_bc','mean'), n=('fc_bc','size'), high=('high','first'), train=('train','first')).reset_index()
    s['res']=s.high-s.mu; return s
for lead in [0,1,2]:
    e=ens(lead, MODELS_L1); tr=e[e.train]; te=e[~e.train]
    print(f'lead{lead}: ens MAE train {tr.res.abs().mean():.2f} test {te.res.abs().mean():.2f}  sigma train {tr.res.std():.2f} test {te.res.std():.2f}  bias test {te.res.mean():.2f}')
E1=ens(1, MODELS_L1); E0=ens(0, MODELS_L1)
sig={1:E1[E1.train].res.std(), 0:E0[E0.train].res.std()}
# empirical residual distribution (train) for discrete probabilities
res1=E1[E1.train].res.values; res0=E0[E0.train].res.values
def bracket_prob(mu, lo, hi, res):
    # P(lo < high < hi) where high = round(mu+res) integer -> use empirical residuals
    v=mu+res; return ((v>lo)&(v<hi)).mean()
# ---- snapshots
ev=ev.merge(E1[['station','date','mu']].rename(columns={'date':'target','mu':'mu1'}), on=['station','target'], how='left').merge(E0[['station','date','mu']].rename(columns={'date':'target','mu':'mu0'}), on=['station','target'], how='left')
c=c.merge(b[['ticker','lo','hi']], on='ticker')
c['open_ts']=(pd.to_datetime(c.open_time, format='ISO8601', utc=True)-pd.Timestamp(0, tz='UTC')).dt.total_seconds().astype(int); c['close_ts']=(pd.to_datetime(c.close_time, format='ISO8601', utc=True)-pd.Timestamp(0, tz='UTC')).dt.total_seconds().astype(int); print('h_from_open range', ((c.ts-c.open_ts)//3600).min(), ((c.ts-c.open_ts)//3600).max())
c['h_from_open']=(c.ts-c.open_ts)//3600; c['h_to_close']=(c.close_ts-c.ts)//3600
def fee(p, n=1): return np.ceil(100*0.07*p*(1-p)*n)/100
out=[]
for label, sel in [('open+2h', c.h_from_open==2),('open+8h (D-1 evening)', c.h_from_open==8),('D 06Z-ish (h_to_close=23)', c.h_to_close==23),('D 12Z-ish (h_to_close=17)', c.h_to_close==17),('D 15Z (h_to_close=14)', c.h_to_close==14),('D 17Z (h_to_close=12)', c.h_to_close==12),('D 19Z (h_to_close=10)', c.h_to_close==10)]:
    s=c[sel].merge(ev[['event_ticker','truth','mu1','mu0','target','station']], on='event_ticker').dropna(subset=['truth','mu1','mu0'])
    s=s[(s.ask>0)&(s.ask<1)]
    s['y']=((s.truth>s.lo)&(s.truth<s.hi)).astype(float); s['mid']=(s.bid+s.ask)/2
    use0='D ' in label
    s['pm']=[bracket_prob(mu0 if use0 else mu1, lo, hi, res0 if use0 else res1) for mu0,mu1,lo,hi in zip(s.mu0,s.mu1,s.lo,s.hi)]
    brier_mkt=((s.mid-s.y)**2).mean(); brier_mod=((s.pm-s.y)**2).mean()
    # trading: buy yes if pm-ask>thr; buy no if (1-pm)-(1-bid)>thr => bid-pm>thr
    pnl=[]; 
    for thr in [0.03,0.06,0.10]:
        by=s[s.pm-s.ask>thr]; bn=s[s.bid-s.pm>thr]
        py=(by.y-by.ask-fee(by.ask)).sum(); pn=((1-bn.y)-(1-bn.bid)-fee(bn.bid)).sum()
        stake=by.ask.sum()+(1-bn.bid).sum()
        pnl.append(f'thr{thr}: N={len(by)+len(bn)} pnl/contract={ (py+pn)/max(1,len(by)+len(bn)):.3f} ROI={(py+pn)/max(stake,1e-9)*100:.1f}%')
    print(f'\n[{label}] N brackets={len(s)} events={s.event_ticker.nunique()} | Brier market {brier_mkt:.4f} model {brier_mod:.4f} | mid-price sum per event {s.groupby("event_ticker").mid.sum().mean():.3f}')
    print('   ', ' | '.join(pnl))
    # calibration of market by price bucket
    s['pb']=pd.cut(s.mid,[0,0.05,0.15,0.3,0.5,0.7,0.85,0.95,1]); print('    market calib (mid->hit):', s.groupby('pb',observed=True).y.mean().round(3).to_dict())
    print('    model calib:', s.assign(pb=pd.cut(s.pm,[0,0.05,0.15,0.3,0.5,0.7,0.85,0.95,1])).groupby('pb',observed=True).y.mean().round(3).to_dict())
