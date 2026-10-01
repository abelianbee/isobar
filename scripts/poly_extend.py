"""Extend Polymarket daily-high coverage by slug lookup (gamma /events?slug=...) for given cities and date range,
then download hourly CLOB price history for each bracket. Skips events already in poly_prices/."""
import requests, json, os, re, time, sys, datetime as dt, pandas as pd
R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
CITIES=sys.argv[1].split(',') if len(sys.argv)>1 else ['nyc','chicago','miami','atlanta','seattle','dallas']
START=dt.date.fromisoformat(sys.argv[2]) if len(sys.argv)>2 else dt.date(2026,3,20)
END=dt.date.fromisoformat(sys.argv[3]) if len(sys.argv)>3 else dt.date(2026,9,26)
S=requests.Session(); found=[]; n=0; t0=time.time()
def get(url, params, tries=5):
    for i in range(tries):
        try:
            r=S.get(url, params=params, timeout=60)
            if r.status_code==200: return r.json()
            if r.status_code==404: return None
        except Exception as e: pass
        time.sleep(2*(i+1))
    return None
d=START
while d<=END:
    for city in CITIES:
        slug=f"highest-temperature-in-{city}-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
        f=R+'poly_prices/'+slug+'.json'
        if os.path.exists(f): continue
        j=get('https://gamma-api.polymarket.com/events', {'slug':slug})
        if not j: continue
        e=j[0] if isinstance(j,list) else j
        if not e.get('closed'): continue
        found.append({k:e.get(k) for k in ['id','slug','title','endDate','startDate','volume','closed']})
        out={'slug':slug,'endDate':e.get('endDate'),'description':(e.get('description') or '')[:300],'markets':[]}
        for m in e.get('markets',[]):
            try: toks=json.loads(m.get('clobTokenIds') or '[]')
            except Exception: toks=[]
            rec={'title':m.get('groupItemTitle'),'question':m.get('question'),'outcomePrices':m.get('outcomePrices'),'volume':m.get('volume'),'volumeNum':m.get('volumeNum'),'token':toks[0] if toks else None,'history':None}
            if rec['token'] and e.get('endDate'):
                ed=pd.Timestamp(e['endDate'])
                h=get('https://clob.polymarket.com/prices-history', {'market':rec['token'],'startTs':int((ed-pd.Timedelta(days=6)).timestamp()),'endTs':int((ed+pd.Timedelta(days=2)).timestamp()),'fidelity':60})
                rec['history']=(h or {}).get('history')
                time.sleep(0.05)
            out['markets'].append(rec)
        json.dump(out, open(f,'w')); n+=1
        if n%20==0: print(n,'events',d,round(time.time()-t0),'s',flush=True)
    d+=dt.timedelta(days=1)
json.dump(found, open(R+f'poly_extend_{START}_{END}.json','w'))
print('DONE new events', n, 'found', len(found), flush=True)
