import pandas as pd, numpy as np
import statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/'
TZ={'NYC':'America/New_York','MDW':'America/Chicago','MIA':'America/New_York','AUS':'America/Chicago','LAX':'America/Los_Angeles','DEN':'America/Denver','PHL':'America/Philadelphia'}
TZ['PHL']='America/New_York'
CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
asos=pd.read_parquet(R+'asos_hourly.parquet' if False else R+'weather/asos_hourly.parquet')
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
# bias-corrected lead-1 ensemble (train bias on <2025)
fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date'])
fd['train']=pd.to_datetime(fd.date)<pd.Timestamp('2025-01-01'); fd['err']=fd.tmax_fc-fd.high
bias=fd[fd.train].groupby(['station','model','lead']).err.mean().rename('bias').reset_index(); fd=fd.merge(bias,on=['station','model','lead']); fd['bc']=fd.tmax_fc-fd.bias
mu1=fd[fd.lead==1].groupby(['station','date']).agg(mu1=('bc','mean'), high=('high','first')).reset_index()
# hourly obs in local time
rows=[]
for st,g in asos.groupby('station'):
    g=g.copy(); lt=g.valid.dt.tz_convert(TZ[st]); g['ldate']=lt.dt.date; g['lhour']=lt.dt.hour+lt.dt.minute/60
    g=g.sort_values('valid'); g['runmax']=g.groupby('ldate').tmpf.cummax()
    g['station']=st; rows.append(g[['station','ldate','lhour','tmpf','runmax','valid']])
ob=pd.concat(rows)
ob.to_parquet(R+'weather/asos_local.parquet')
# snapshot at local hours H: last obs with lhour<=H
out=[]
for H in [7,8,9,10,11,12,13,14,15]:
    s=ob[ob.lhour<=H].sort_values('valid').groupby(['station','ldate']).last().reset_index()[['station','ldate','tmpf','runmax']].rename(columns={'ldate':'date','tmpf':f'T','runmax':'RM'})
    s['H']=H; out.append(s)
snap=pd.concat(out).merge(mu1, on=['station','date'])
snap['month']=pd.to_datetime(snap.date).dt.month; snap['train']=pd.to_datetime(snap.date)<pd.Timestamp('2025-01-01')
snap['doy']=pd.to_datetime(snap.date).dt.dayofyear; snap['s1']=np.sin(2*np.pi*snap.doy/365); snap['c1']=np.cos(2*np.pi*snap.doy/365)
snap.to_parquet(R+'weather/snap_obs.parquet')
print('snapshot rows', len(snap))
# fit per (station,H): high ~ mu1 + T + RM + seasonal ; report residual sigma train/test vs forecast-only
res=[]
for (st,H),g in snap.groupby(['station','H']):
    tr=g[g.train]; te=g[~g.train]
    if len(tr)<300 or len(te)<100: continue
    X=lambda d: sm.add_constant(d[['mu1','T','RM','s1','c1']])
    m=sm.OLS(tr.high, X(tr)).fit(); pred=m.predict(X(te))
    res.append({'station':st,'H':H,'sig_fc_test':(te.high-te.mu1).std(),'sig_obs_test':(te.high-pred).std(),'mae_fc':(te.high-te.mu1).abs().mean(),'mae_obs':(te.high-pred).abs().mean(),'b_mu1':m.params['mu1'],'b_T':m.params['T'],'b_RM':m.params['RM'],'n_test':len(te)})
res=pd.DataFrame(res); print(res.round(2).to_string())
print('\nmean across stations by H:'); print(res.groupby('H')[['sig_fc_test','sig_obs_test','mae_fc','mae_obs']].mean().round(2).to_string())
