import pandas as pd, numpy as np, json, glob
R='/Users/abebraham/Desktop/resume/edge/data/weather/'
rows=[]
for f in sorted(glob.glob(R+'fcst/*.json'))+sorted(glob.glob(R+'fcst2/*.json'))+sorted(glob.glob(R+'fcst3/*.json'))+sorted(glob.glob(R+'fcst4/*.json')):
    st=f.split('/')[-1].split('_')[0]; j=json.load(open(f)); h=j['hourly']; t=pd.to_datetime(h['time'])
    df=pd.DataFrame({k:v for k,v in h.items() if k!='time'}); df['time']=t; df['date']=t.date
    long=df.melt(id_vars=['time','date'], var_name='var', value_name='temp').dropna()
    m=long['var'].str.extract(r'temperature_2m(?:_previous_day(\d))?_(.*)'); long['lead']=m[0].fillna('0').astype(int); long['model']=m[1]; long['station']=st; rows.append(long)
fc=pd.concat(rows, ignore_index=True)
daily=fc.groupby(['station','date','model','lead']).temp.agg(tmax_fc='max', tmin_fc='min').reset_index()
daily.to_parquet(R+'fcst_daily.parquet'); print('daily rows', len(daily)); print(daily[daily.lead==1].groupby('model').tmax_fc.count().to_string())
