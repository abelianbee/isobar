import requests, json, time, sys, pandas as pd
B="https://api.elections.kalshi.com/trade-api/v2"; R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
def get(path, params):
    for i in range(6):
        r=requests.get(B+path, params=params, timeout=60)
        if r.status_code==200: return r.json()
        time.sleep(2+2*i)
    return None
def all_markets(series):
    out=[]
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
            time.sleep(0.3)
    df=pd.DataFrame(out).drop_duplicates('ticker') if out else pd.DataFrame(); df['series']=series; return df
series=sys.argv[1].split(','); frames=[]
for s in series:
    df=all_markets(s); print(s, len(df), df.close_time.min() if len(df) else None, flush=True); frames.append(df)
mk=pd.concat(frames, ignore_index=True); mk.to_pickle(R+'markets_'+sys.argv[2]+'.pkl'); print('total', len(mk))
