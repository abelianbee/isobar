import pandas as pd, json, glob
R='/Users/abebraham/Desktop/resume/edge/data/weather/'
cl=[]
for f in sorted(glob.glob(R+'obs/cli_*.json')):
    st=f.split('/')[-1].split('_')[1]
    try: res=json.load(open(f))['results']
    except Exception: continue
    for r in res: cl.append({'station':st,'date':r['valid'],'high':r.get('high'),'high_time':str(r.get('high_time')),'low':r.get('low'),'high_normal':r.get('high_normal')})
cli=pd.DataFrame(cl); cli['date']=pd.to_datetime(cli.date).dt.date
for k in ['high','low','high_normal']: cli[k]=pd.to_numeric(cli[k], errors='coerce')
cli.to_parquet(R+'cli_daily.parquet'); print('cli rows', len(cli), cli.groupby('station').high.count().to_dict())
