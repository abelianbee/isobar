import requests, json, collections, re, time
G="https://gamma-api.polymarket.com"; R='/Users/abebraham/Desktop/resume/edge/data/kalshi/'
allev=[]; cursor=None; pages=0
while True:
    p={'tag_slug':'weather','closed':'true','limit':100}
    if cursor: p['cursor']=cursor
    try: j=requests.get(G+'/events/keyset',params=p,timeout=60).json()
    except Exception as e: print('err',e,flush=True); time.sleep(5); continue
    ev=j.get('events',[]); allev+=ev; pages+=1; cursor=j.get('next_cursor')
    if pages%20==0: print('pages',pages,'events',len(allev),flush=True)
    if not ev or not cursor or pages>600: break
print('DONE pages', pages, 'events', len(allev), flush=True)
temp=[e for e in allev if 'temperature' in e.get('slug','')]
json.dump(temp, open(R+'poly_temp_events.json','w'))
cities=collections.Counter([re.sub(r'-on-.*','',e['slug'].split('-in-')[1]) if '-in-' in e['slug'] else 'other' for e in temp])
print('temperature events', len(temp)); print(cities.most_common(40))
vol=collections.defaultdict(float); n=collections.Counter()
for e in temp:
    c=re.sub(r'-on-.*','',e['slug'].split('-in-')[1]) if '-in-' in e['slug'] else 'other'; vol[c]+=float(e.get('volume') or 0); n[c]+=1
print({c:(round(vol[c]/max(1,n[c])),n[c]) for c in sorted(vol,key=lambda k:-vol[k])})
if temp: print('first', temp[0]['slug'], temp[0].get('startDate','')[:10], 'last', temp[-1]['slug'])
