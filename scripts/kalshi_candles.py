import requests, json, time, os, sys, datetime as dt, pandas as pd, glob
from concurrent.futures import ThreadPoolExecutor
B="https://api.elections.kalshi.com/trade-api/v2"; R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
mk=pd.concat([pd.read_pickle(f) for f in glob.glob(R+'markets_*.pkl')]).drop_duplicates('ticker')
print('markets', len(mk), flush=True)
S=requests.Session(); R429=[0]
def get(path, params, tries=6):
    for i in range(tries):
        try:
            r=S.get(B+path, params=params, timeout=60)
            if r.status_code==200: return r.json()
            if r.status_code==429: R429[0]+=1; time.sleep(3+3*i); continue
            if r.status_code==404: return None
            time.sleep(1+i)
        except Exception: time.sleep(2+2*i)
    return 'ERR'
def work(m):
    t=m.ticker; f=R+f'candles/{t}.json'
    if os.path.exists(f): return 0
    ts=int(dt.datetime.fromisoformat(m.open_time.replace('Z','+00:00')).timestamp()); te=int(dt.datetime.fromisoformat(m.close_time.replace('Z','+00:00')).timestamp())+3600
    j=get(f'/historical/markets/{t}/candlesticks',{'start_ts':ts,'end_ts':te,'period_interval':60})
    if j is None: j=get(f'/series/{m.series}/markets/{t}/candlesticks',{'start_ts':ts,'end_ts':te,'period_interval':60})
    if j=='ERR' or j is None: return -1
    json.dump(j, open(f,'w')); return 1
rows=[m for _,m in mk.iterrows()]
t0=time.time(); done=0; err=0
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 else 4) as ex:
    for i,res in enumerate(ex.map(work, rows)):
        if res==1: done+=1
        if res==-1: err+=1
        if i%1000==0: print(i, "processed", done, "fetched", err, "errors", R429[0], "429s", round(time.time()-t0), "s", flush=True)
print('finished', done, err)
