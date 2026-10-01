import pandas as pd, numpy as np, json, glob, os
R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
mk=pd.concat([pd.read_pickle(f) for f in glob.glob(R+'markets_*.pkl')]).drop_duplicates('ticker').set_index('ticker')
rows=[]
def g(d,k):
    if not d: return np.nan
    v=d.get(k); 
    if v is None: v=d.get(k+'_dollars')
    return float(v) if v is not None else np.nan
for f in glob.glob(R+'candles/*.json'):
    t=f.split('/')[-1][:-5]
    try: c=json.load(open(f)).get('candlesticks')
    except Exception: continue
    if not c: continue
    for x in c:
        rows.append((t, x['end_period_ts'], g(x.get('price'),'close'), g(x.get('price'),'mean'), g(x.get('yes_bid'),'close'), g(x.get('yes_ask'),'close'), g(x.get('yes_bid'),'high'), g(x.get('yes_ask'),'low'), float(x.get('volume') or x.get('volume_fp') or 0), float(x.get('open_interest') or x.get('open_interest_fp') or 0)))
c=pd.DataFrame(rows, columns=['ticker','ts','close','mean','bid','ask','bid_high','ask_low','vol','oi'])
c['time']=pd.to_datetime(c.ts, unit='s', utc=True)
c=c.join(mk[['event_ticker','series','floor_strike','cap_strike','strike_type','result','expiration_value','open_time','close_time']], on='ticker')
c.to_parquet(R+'candles_long.parquet'); print('rows', len(c), 'tickers', c.ticker.nunique(), 'events', c.event_ticker.nunique())
print(c.groupby('series').agg(tickers=('ticker','nunique'), rows=('ts','size')).to_string())
print(c[['bid','ask','close','vol']].describe().round(3).to_string())
