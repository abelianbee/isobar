import pandas as pd, numpy as np, json, glob, re
R='/Users/abebraham/Desktop/resume/edge/data/weather/'
rows=[]
for f in sorted(glob.glob(R+'fcst/*.json')):
    st=f.split('/')[-1].split('_')[0]; j=json.load(open(f)); h=j['hourly']; t=pd.to_datetime(h['time'])
    df=pd.DataFrame({k:v for k,v in h.items() if k!='time'}); df['time']=t; df['date']=t.date; df['hour']=t.hour
    long=df.melt(id_vars=['time','date','hour'], var_name='var', value_name='temp').dropna()
    m=long['var'].str.extract(r'temperature_2m(?:_previous_day(\d))?_(.*)'); long['lead']=m[0].fillna('0').astype(int); long['model']=m[1]
    long['station']=st; rows.append(long)
fc=pd.concat(rows, ignore_index=True)
fc.to_parquet(R+'fcst_hourly.parquet')
daily=fc.groupby(['station','date','model','lead']).temp.max().reset_index().rename(columns={'temp':'tmax_fc'})
daily.to_parquet(R+'fcst_daily.parquet')
print('forecast daily rows', len(daily)); print(daily.groupby(['model','lead']).tmax_fc.count().unstack().to_string())
cl=[]
for f in sorted(glob.glob(R+'obs/cli_*.json')):
    st=f.split('/')[-1].split('_')[1]
    try: res=json.load(open(f))['results']
    except Exception: continue
    for r in res: cl.append({'station':st,'date':r['valid'],'high':r.get('high'),'high_time':r.get('high_time'),'low':r.get('low'),'high_normal':r.get('high_normal')})
cli=pd.DataFrame(cl); cli['date']=pd.to_datetime(cli.date).dt.date; cli['high']=pd.to_numeric(cli.high, errors='coerce'); cli['low']=pd.to_numeric(cli.low, errors='coerce'); cli['high_normal']=pd.to_numeric(cli.high_normal, errors='coerce'); cli['high_time']=cli.high_time.astype(str)
cli.to_parquet(R+'cli_daily.parquet'); print('cli rows', len(cli), cli.groupby('station').high.count().to_dict())
asos=[]
for f in glob.glob(R+'obs/asos_*.csv'):
    try: d=pd.read_csv(f)
    except Exception: continue
    if 'tmpf' not in d: continue
    d['valid']=pd.to_datetime(d.valid, utc=True); d['tmpf']=pd.to_numeric(d.tmpf, errors='coerce'); asos.append(d.dropna(subset=['tmpf']))
asos=pd.concat(asos); asos.to_parquet(R+'asos_hourly.parquet'); print('asos rows', len(asos), asos.groupby('station').size().to_dict())
map_={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL'}
daily['cli_st']=daily.station.map(map_)
x=daily.merge(cli[['station','date','high']].rename(columns={'station':'cli_st'}), on=['cli_st','date'])
x['err']=x.tmax_fc-x.high
print('\nMAE (F) by model x lead, all stations:'); print(x.groupby(['model','lead']).err.apply(lambda s: s.abs().mean()).unstack().round(2).to_string())
print('\nBias by model x lead:'); print(x.groupby(['model','lead']).err.mean().unstack().round(2).to_string())
print('\nMAE lead1 by station:'); print(x[x.lead==1].groupby(['station','model']).err.apply(lambda s: s.abs().mean()).unstack().round(2).to_string())
