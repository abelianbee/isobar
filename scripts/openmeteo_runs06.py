import requests, json, time, os, re, datetime as dt, sys
R='/Users/abebraham/Desktop/resume/edge/data/weather/runs/'
ST={'NYC':(40.7789,-73.9692,'America/New_York'),'MDW':(41.7868,-87.7522,'America/Chicago'),'MIA':(25.7959,-80.2870,'America/New_York'),'AUS':(30.1975,-97.6664,'America/Chicago'),'LAX':(33.9425,-118.4081,'America/Los_Angeles'),'DEN':(39.8561,-104.6737,'America/Denver'),'PHL':(39.8721,-75.2411,'America/New_York')}
MODELS=['gfs_seamless','ecmwf_ifs','icon_seamless','ncep_nbm_conus','ncep_hrrr_conus','ukmo_seamless','meteofrance_seamless','knmi_seamless','dmi_seamless','ecmwf_aifs025_single']
start=dt.date(2026,4,15); end=dt.date(2026,9,26)
def fetch(lat,lon,tz,run,models):
    for i in range(6):
        r=requests.get('https://single-runs-api.open-meteo.com/v1/forecast',params={'latitude':lat,'longitude':lon,'run':run,'hourly':'temperature_2m','models':','.join(models),'temperature_unit':'fahrenheit','timezone':tz,'forecast_days':3},timeout=120)
        if r.status_code==429: time.sleep(30); continue
        j=r.json()
        if j.get('error'):
            m=re.search(r'Model: (\S+), run', j.get('reason',''))
            if m:
                bad=m.group(1); toks=[t for t in re.split(r'[_\-]', bad) if len(t)>2 and t not in ('ncep','cmc','dwd','seamless','global','conus','single','upper','level')]
                drop=[x for x in models if any(t in x for t in toks)]
                if not drop: drop=[models[0]]
                models=[x for x in models if x not in drop]
                if not models: return None
                continue
            return None
        return j
    return None
n=0; t0=time.time()
d=start
while d<=end:
    for st,(lat,lon,tz) in ST.items():
        for hh in ['06:00']:
            run=f'{d - dt.timedelta(days=1):%Y-%m-%d}T{hh}'   # D-1 run for target day d
            f=R+f'{st}_{d:%Y%m%d}_{hh[:2]}.json'
            if os.path.exists(f): continue
            j=fetch(lat,lon,tz,run,MODELS)
            json.dump(j if j else {'error':True}, open(f,'w')); n+=1
            time.sleep(0.15)
    if n and n%100<14: print(d, n, 'calls', round(time.time()-t0), 's', flush=True)
    d+=dt.timedelta(days=1)
print('done', n)
