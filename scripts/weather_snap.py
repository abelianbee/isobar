import pandas as pd, numpy as np, glob
R='/Users/abebraham/Desktop/resume/edge/data/'
ev=pd.read_parquet(R+'weather/events.parquet'); b=pd.read_parquet(R+'kalshi/brackets.parquet'); c=pd.read_parquet(R+'kalshi/candles_long.parquet')
ev['truth']=ev.cli_high
x=b.merge(ev[['event_ticker','truth']], on='event_ticker').dropna(subset=['truth']); x['py']=(x.truth>x.lo)&(x.truth<x.hi)
bad=set(x[(x.result=='yes')!=x.py].event_ticker); ev=ev[~ev.event_ticker.isin(bad)&ev.truth.notna()]
c=c[c.event_ticker.isin(ev.event_ticker)].merge(b[['ticker','lo','hi']], on='ticker')
c['open_ts']=(pd.to_datetime(c.open_time, format='ISO8601', utc=True)-pd.Timestamp(0, tz='UTC')).dt.total_seconds().astype(int)
c['h']=(c.ts-c.open_ts)//3600
c=c.merge(ev[['event_ticker','truth','station','tz','target','kind']].drop_duplicates('event_ticker'), on='event_ticker')
c['y']=((c.truth>c.lo)&(c.truth<c.hi)).astype(float); c['mid']=(c.bid+c.ask)/2
c['center']=np.where(np.isinf(c.lo), c.hi-1.5, np.where(np.isinf(c.hi), c.lo+1.5, (c.lo+c.hi)/2))
c=c[(c.ask>0)&(c.ask<=1)]
# event-hour aggregates
g=c.groupby(['event_ticker','h'])
agg=g.agg(n_br=('ticker','size'), sum_mid=('mid','sum'), sum_ask=('ask','sum'), sum_bid=('bid','sum'), vol=('vol','sum'), station=('station','first'), tz=('tz','first'), truth=('truth','first'), target=('target','first'), ts=('ts','max')).reset_index()
w=c.assign(wm=c.mid*c.center).groupby(['event_ticker','h']).agg(wm=('wm','sum'), sm=('mid','sum')).reset_index(); agg['imp_mean']=w.wm/w.sm
fav=c.sort_values('mid').groupby(['event_ticker','h']).last().reset_index()[['event_ticker','h','ticker','mid','ask','bid','y','center']].rename(columns={'ticker':'fav_ticker','mid':'fav_mid','ask':'fav_ask','bid':'fav_bid','y':'fav_y','center':'fav_center'})
agg=agg.merge(fav, on=['event_ticker','h'])
agg=agg[agg.n_br==6]
agg['local_hour']=[pd.Timestamp(t, unit='s', tz='UTC').tz_convert(tz).hour for t,tz in zip(agg.ts, agg.tz)]
agg['day_rel']=[(pd.Timestamp(t, unit='s', tz='UTC').tz_convert(tz).date()-tg).days for t,tz,tg in zip(agg.ts, agg.tz, agg.target)]
agg.to_parquet(R+'weather/snap_market.parquet'); c.to_parquet(R+'weather/candles_ev.parquet')
print('event-hours', len(agg), 'events', agg.event_ticker.nunique()); print(agg.groupby('station').event_ticker.nunique().to_dict())
print('sum of mids by hour-from-open:'); print(agg.groupby('h').agg(sum_mid=('sum_mid','mean'), sum_ask=('sum_ask','mean'), sum_bid=('sum_bid','mean'), fav_mid=('fav_mid','mean'), fav_hit=('fav_y','mean'), n=('h','size')).round(3).to_string())
