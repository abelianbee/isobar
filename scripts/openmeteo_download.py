import requests, json, time, os, pandas as pd
R='/Users/abebraham/Desktop/resume/edge/data/weather/fcst/'
ST={'NYC':(40.7789,-73.9692,'America/New_York'),'MDW':(41.7868,-87.7522,'America/Chicago'),'MIA':(25.7959,-80.2870,'America/New_York'),'AUS':(30.1975,-97.6664,'America/Chicago'),'LAX':(33.9425,-118.4081,'America/Los_Angeles'),'DEN':(39.8561,-104.6737,'America/Denver'),'PHL':(39.8721,-75.2411,'America/New_York')}
MODELS='gfs_seamless,ecmwf_ifs025,icon_seamless,ncep_hrrr_conus,ncep_nbm_conus,gem_seamless,ukmo_seamless'
VARS='temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2,temperature_2m_previous_day3'
ranges=[('2021-08-01','2022-06-30'),('2022-07-01','2023-06-30'),('2023-07-01','2024-06-30'),('2024-07-01','2025-06-30'),('2025-07-01','2026-09-27')]
for st,(lat,lon,tz) in ST.items():
    for a,b in ranges:
        f=R+f'{st}_{a}.json'
        if os.path.exists(f): continue
        for i in range(5):
            r=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params={'latitude':lat,'longitude':lon,'start_date':a,'end_date':b,'hourly':VARS,'models':MODELS,'temperature_unit':'fahrenheit','timezone':tz},timeout=180)
            if r.status_code==200: json.dump(r.json(),open(f,'w')); print(st,a,'ok',len(r.content),flush=True); break
            print(st,a,r.status_code,r.text[:150],flush=True); time.sleep(10*(i+1))
        time.sleep(3)
