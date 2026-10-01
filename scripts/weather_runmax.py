import pandas as pd, numpy as np
R='/Users/abebraham/Desktop/resume/edge/data/'
ob=pd.read_parquet(R+'weather/asos_local.parquet'); cli=pd.read_parquet(R+'weather/cli_daily.parquet')
CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
ob['cli']=ob.station.map(CLI); ob=ob.merge(cli[['station','date','high','high_time']].rename(columns={'station':'cli','date':'ldate'}), on=['cli','ldate'])
# final hourly running max at end of day vs CLI high
fin=ob.sort_values('valid').groupby(['station','ldate']).agg(RMend=('runmax','max'), high=('high','first')).reset_index(); fin['gap']=fin.high-fin.RMend
print('CLI high minus max of hourly METAR temps (all stations):'); print(fin.gap.value_counts(normalize=True).sort_index().round(3).head(8).to_dict())
print(fin.groupby('station').gap.agg(['mean', lambda s:(s>=1).mean()]).round(3).to_string())
# by local hour: P(final high > RM_h) and P(final high >= RM_h + 2)
out=[]
for H in [13,14,15,16,17,18]:
    s=ob[ob.lhour<=H].sort_values('valid').groupby(['station','ldate']).last().reset_index()
    s['gap']=s.high-s.runmax; out.append(pd.DataFrame({'H':H,'station':s.station,'gap':s.gap}))
o=pd.concat(out); print('\nP(high - RM_H >= k) by local hour H (all stations):')
print(o.groupby('H').gap.apply(lambda g: pd.Series({'>=1':(g>=1).mean(),'>=2':(g>=2).mean(),'>=3':(g>=3).mean(),'mean':g.mean()})).unstack().round(3).to_string())
# ---- market at local hours 14-17 on day D: bracket containing RM vs bracket above vs below
c=pd.read_parquet(R+'weather/candles_ev.parquet'); c=c[c.h<=37]
c['ltime']=pd.to_datetime(c.ts,unit='s',utc=True); c['lhour']=[t.tz_convert(tz).hour for t,tz in zip(c.ltime,c.tz)]; c['ldate']=[t.tz_convert(tz).date() for t,tz in zip(c.ltime,c.tz)]
c=c[c.ldate==c.target]
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
for H in [13,14,15,16,17]:
    s=ob[ob.lhour<=H].sort_values('valid').groupby(['station','ldate']).last().reset_index()[['station','ldate','runmax','tmpf']].rename(columns={'ldate':'target','runmax':'RM','tmpf':'T'})
    m=c[c.lhour==H].merge(s,on=['station','target'])
    m['contains_RM']=(m.RM>m.lo)&(m.RM<m.hi); m['above']=m.lo>=m.RM; m['below']=m.hi<=m.RM
    for lab,sel in [('bracket containing RM',m.contains_RM),('brackets above RM',m.above),('brackets below RM',m.below)]:
        g=m[sel]; py=g.y-g.ask-fee(g.ask); pn=(1-g.y)-(1-g.bid)-fee(g.bid)
        print(f'local {H}:00 {lab:24s} N={len(g):4d} mid={g.mid.mean():.3f} hit={g.y.mean():.3f} buyYES@ask pnl={py.mean():+.4f} buyNO@bid pnl={pn.mean():+.4f}')
    # the single bracket immediately above the RM bracket
    g=m[m.above].sort_values('lo').groupby('event_ticker').first().reset_index(); py=g.y-g.ask-fee(g.ask)
    print(f'   -> first bracket above RM: N={len(g)} mid={g.mid.mean():.3f} ask={g.ask.mean():.3f} hit={g.y.mean():.3f} pnl/ct={py.mean():+.4f}')
