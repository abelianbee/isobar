import json, requests, time, os, re, pandas as pd
R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
T=json.load(open(R+'poly_temp_events.json'))
CITIES=['nyc','chicago','miami','dallas','atlanta','seattle','london','seoul','toronto']
ev=[e for e in T if '-in-' in e['slug'] and re.sub(r'-on-.*','',e['slug'].split('-in-')[1]) in CITIES]
ev.sort(key=lambda e: e.get('endDate',''), reverse=True)
print('events', len(ev), flush=True)
S=requests.Session(); n=0; t0=time.time()
for e in ev:
    f=R+'poly_prices/'+e['slug']+'.json'
    if os.path.exists(f): continue
    out={'slug':e['slug'],'endDate':e.get('endDate'),'description':e.get('description','')[:300],'markets':[]}
    for m in e['markets']:
        try: toks=json.loads(m.get('clobTokenIds') or '[]')
        except Exception: toks=[]
        rec={'title':m.get('groupItemTitle'),'question':m.get('question'),'outcomePrices':m.get('outcomePrices'),'volume':m.get('volume'),'volumeNum':m.get('volumeNum'),'token':toks[0] if toks else None,'history':None}
        if rec['token']:
            for i in range(4):
                try:
                    ed=pd.Timestamp(e.get('endDate')); r=S.get('https://clob.polymarket.com/prices-history',params={'market':rec['token'],'startTs':int((ed-pd.Timedelta(days=6)).timestamp()),'endTs':int((ed+pd.Timedelta(days=2)).timestamp()),'fidelity':60},timeout=60)
                    if r.status_code==200: rec['history']=r.json().get('history'); break
                    time.sleep(2*(i+1))
                except Exception: time.sleep(2*(i+1))
            time.sleep(0.05)
        out['markets'].append(rec)
    json.dump(out, open(f,'w')); n+=1
    if n%25==0: print(n, 'events', round(time.time()-t0), 's', flush=True)
print('done', n, flush=True)
