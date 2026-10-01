import requests, json, time, os, sys, datetime as dt, pandas as pd
B="https://api.elections.kalshi.com/trade-api/v2"; R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
S=requests.Session()
def get(path, params, tries=6):
    for i in range(tries):
        try:
            r=S.get(B+path, params=params, timeout=60)
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(2+2*i); continue
            if r.status_code==404: return None
            time.sleep(1+i)
        except Exception as e: time.sleep(2+2*i)
    return None
def all_markets(series):
    out=[]; cur=None
    for path in ['/historical/markets','/markets']:
        cur=None
        while True:
            p={'series_ticker':series,'limit':1000}
            if path=='/markets': p['status']='settled'
            if cur: p['cursor']=cur
            j=get(path,p)
            if not j: break
            ms=j.get('markets',[]); out+=ms; cur=j.get('cursor')
            if not cur or not ms: break
    df=pd.DataFrame(out).drop_duplicates('ticker'); df['series']=series; return df
series=sys.argv[1].split(',')
frames=[]
for s in series:
    df=all_markets(s); print(s, len(df), df.close_time.min() if len(df) else None, flush=True); frames.append(df)
mk=pd.concat(frames, ignore_index=True); mk.to_pickle(R+'markets_'+'_'.join(series)+'.pkl')
print('total markets', len(mk), flush=True)
# candlesticks
n=0; t0=time.time()
for _,m in mk.iterrows():
    t=m.ticker; f=R+f'candles/{t}.json'
    if os.path.exists(f): continue
    ts=int(dt.datetime.fromisoformat(m.open_time.replace('Z','+00:00')).timestamp()); te=int(dt.datetime.fromisoformat(m.close_time.replace('Z','+00:00')).timestamp())+3600
    j=get(f'/historical/markets/{t}/candlesticks',{'start_ts':ts,'end_ts':te,'period_interval':60})
    if j is None:
        j=get(f'/series/{m.series}/markets/{t}/candlesticks',{'start_ts':ts,'end_ts':te,'period_interval':60})
    json.dump(j if j is not None else {'candlesticks':None}, open(f,'w'))
    n+=1
    if n%200==0: print(n, 'candles fetched', round(time.time()-t0), 's', flush=True)
    time.sleep(0.12)
print('done', n)
