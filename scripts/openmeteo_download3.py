import requests, json, time, os
R='/Users/abebraham/Desktop/resume/edge/data/weather/fcst3/'
ST={'ATL':(33.6407,-84.4277,'America/New_York'),'BOS':(42.3656,-71.0096,'America/New_York'),'DFW':(32.8998,-97.0403,'America/Chicago'),'DCA':(38.8512,-77.0402,'America/New_York'),'HOU':(29.6454,-95.2789,'America/Chicago'),'LAS':(36.0840,-115.1537,'America/Los_Angeles'),'MSP':(44.8848,-93.2223,'America/Chicago'),'OKC':(35.3931,-97.6007,'America/Chicago'),'PHX':(33.4373,-112.0078,'America/Phoenix'),'SEA':(47.4502,-122.3088,'America/Los_Angeles'),'SFO':(37.6213,-122.3790,'America/Los_Angeles')}
SETS={'a':'gfs_seamless,ecmwf_ifs025,icon_seamless,ncep_hrrr_conus,ncep_nbm_conus,gem_seamless,ukmo_seamless','b':'ecmwf_ifs,ecmwf_aifs025_single,gfs_graphcast025,meteofrance_seamless,jma_seamless,knmi_seamless,dmi_seamless,metno_seamless,kma_seamless,bom_access_global,cma_grapes_global'}
VARS='temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2'
ranges=[('2024-01-01','2024-12-31'),('2025-01-01','2025-12-31'),('2026-01-01','2026-09-27')]
for st,(lat,lon,tz) in ST.items():
    for k,models in SETS.items():
        for a,b in ranges:
            f=R+f'{st}_{k}_{a}.json'
            if os.path.exists(f): continue
            for i in range(6):
                r=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params={'latitude':lat,'longitude':lon,'start_date':a,'end_date':b,'hourly':VARS,'models':models,'temperature_unit':'fahrenheit','timezone':tz},timeout=180)
                if r.status_code==200: json.dump(r.json(),open(f,'w')); print(st,k,a,'ok',flush=True); break
                print(st,k,a,r.status_code,r.text[:100],flush=True); time.sleep(30*(i+1))
            time.sleep(4)
print('done')
