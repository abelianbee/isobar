import requests, json, time, os
R='/Users/abebraham/Desktop/resume/edge/data/weather/fcst2/'
ST={'NYC':(40.7789,-73.9692,'America/New_York'),'MDW':(41.7868,-87.7522,'America/Chicago'),'MIA':(25.7959,-80.2870,'America/New_York'),'AUS':(30.1975,-97.6664,'America/Chicago'),'LAX':(33.9425,-118.4081,'America/Los_Angeles'),'DEN':(39.8561,-104.6737,'America/Denver'),'PHL':(39.8721,-75.2411,'America/New_York')}
MODELS='ecmwf_ifs,ecmwf_aifs025_single,gfs_graphcast025,meteofrance_seamless,jma_seamless,knmi_seamless,dmi_seamless,metno_seamless,kma_seamless,bom_access_global,cma_grapes_global,italia_meteo_arpae_icon_2i'
VARS='temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2'
ranges=[('2023-07-01','2024-06-30'),('2024-07-01','2025-06-30'),('2025-07-01','2026-09-27')]
for st,(lat,lon,tz) in ST.items():
    for a,b in ranges:
        f=R+f'{st}_{a}.json'
        if os.path.exists(f): continue
        for i in range(6):
            r=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params={'latitude':lat,'longitude':lon,'start_date':a,'end_date':b,'hourly':VARS,'models':MODELS,'temperature_unit':'fahrenheit','timezone':tz},timeout=180)
            if r.status_code==200: json.dump(r.json(),open(f,'w')); print(st,a,'ok',len(r.content),flush=True); break
            print(st,a,r.status_code,r.text[:120],flush=True); time.sleep(20*(i+1))
        time.sleep(5)
