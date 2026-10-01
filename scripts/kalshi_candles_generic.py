"""Generic candle downloader: python3 kalshi_candles_generic.py <markets_pkl_name> <threads> <interval>"""
import requests, json, time, os, sys, datetime as dt, pandas as pd, threading
from concurrent.futures import ThreadPoolExecutor
B="https://api.elections.kalshi.com/trade-api/v2"; R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
mk=pd.read_pickle(R+f'markets_{sys.argv[1]}.pkl').drop_duplicates('ticker')
have=set(x[:-5] for x in os.listdir(R+'candles')); mk=mk[~mk.ticker.isin(have)]
mk=mk.assign(ct=pd.to_datetime(mk.close_time,format='ISO8601',utc=True)).sort_values('ct',ascending=False)
print('pending',len(mk),flush=True)
lock=threading.Lock(); state={'next':time.time(),'interval':float(sys.argv[3]) if len(sys.argv)>3 else 0.25,'n429':0}
def throttle():
    with lock:
        now=time.time(); wait=max(0,state['next']-now); state['next']=max(now,state['next'])+state['interval']
    if wait>0: time.sleep(wait)
TL=threading.local()
def sess():
    if not hasattr(TL,'s'): TL.s=requests.Session()
    return TL.s
def get(path,params):
    for i in range(8):
        throttle()
        try: r=sess().get(B+path,params=params,timeout=30)
        except Exception: time.sleep(1); continue
        if r.status_code==200: return r.json()
        if r.status_code==404: return None
        if r.status_code==429:
            with lock: state['n429']+=1; state['interval']=min(1.0,state['interval']*1.1)
            time.sleep(2+2*i); continue
        time.sleep(1+i)
    return 'ERR'
def work(m):
    t=m.ticker; f=R+f'candles/{t}.json'
    if os.path.exists(f): return 0
    ts=int(dt.datetime.fromisoformat(m.open_time.replace('Z','+00:00')).timestamp()); te=int(dt.datetime.fromisoformat(m.close_time.replace('Z','+00:00')).timestamp())+3600
    j=get(f'/historical/markets/{t}/candlesticks',{'start_ts':ts,'end_ts':te,'period_interval':60})
    if j is None: j=get(f'/series/{m.series}/markets/{t}/candlesticks',{'start_ts':ts,'end_ts':te,'period_interval':60})
    if j=='ERR' or j is None: return -1
    json.dump(j,open(f,'w')); return 1
rows=[m for _,m in mk.iterrows()]; t0=time.time(); done=0; err=0
with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv)>2 else 3) as ex:
    for i,res in enumerate(ex.map(work,rows)):
        done+=(res==1); err+=(res==-1)
        if i%200==0:
            with lock: state['interval']=max(0.12,state['interval']*0.95)
            print(i,'processed',done,'fetched',err,'errors',state['n429'],'429s',round(state['interval'],3),'interval',round(time.time()-t0),'s',flush=True)
print('finished',done,err,flush=True)
