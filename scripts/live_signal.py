"""Paper-trading signal: blend forecast vs Kalshi daily-high prices for tomorrow (and today).
Usage: python3 live_signal.py [thr]. Prints edges; does NOT place orders."""
import pandas as pd, numpy as np, requests, json, sys, re, datetime as dt, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/'; B="https://api.elections.kalshi.com/trade-api/v2"
THR=float(sys.argv[1]) if len(sys.argv)>1 else 0.05
ST={'NYC':(40.7789,-73.9692,'America/New_York','KXHIGHNY'),'MDW':(41.7868,-87.7522,'America/Chicago','KXHIGHCHI'),'MIA':(25.7959,-80.2870,'America/New_York','KXHIGHMIA'),'AUS':(30.1975,-97.6664,'America/Chicago','KXHIGHAUS'),'LAX':(33.9425,-118.4081,'America/Los_Angeles','KXHIGHLAX'),'DEN':(39.8561,-104.6737,'America/Denver','KXHIGHDEN'),'PHL':(39.8721,-75.2411,'America/New_York','KXHIGHPHIL'),
 'ATL':(33.6407,-84.4277,'America/New_York','KXHIGHTATL'),'BOS':(42.3656,-71.0096,'America/New_York','KXHIGHTBOS'),'DFW':(32.8998,-97.0403,'America/Chicago','KXHIGHTDAL'),'DCA':(38.8512,-77.0402,'America/New_York','KXHIGHTDC'),'HOU':(29.6454,-95.2789,'America/Chicago','KXHIGHTHOU'),'LAS':(36.0840,-115.1537,'America/Los_Angeles','KXHIGHTLV'),'MSP':(44.8848,-93.2223,'America/Chicago','KXHIGHTMIN'),'OKC':(35.3931,-97.6007,'America/Chicago','KXHIGHTOKC'),'PHX':(33.4373,-112.0078,'America/Phoenix','KXHIGHTPHX'),'SEA':(47.4502,-122.3088,'America/Los_Angeles','KXHIGHTSEA'),'SFO':(37.6213,-122.3790,'America/Los_Angeles','KXHIGHTSFO')}
CLI={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL','ATL':'KATL','BOS':'KBOS','DFW':'KDFW','DCA':'KDCA','HOU':'KHOU','LAS':'KLAS','MSP':'KMSP','OKC':'KOKC','PHX':'KPHX','SEA':'KSEA','SFO':'KSFO'}
MODELS=['gfs_seamless','ecmwf_ifs025','icon_seamless','ncep_hrrr_conus','ncep_nbm_conus','gem_seamless','ukmo_seamless','ecmwf_ifs','ecmwf_aifs025_single','gfs_graphcast025','meteofrance_seamless','jma_seamless','knmi_seamless','dmi_seamless','metno_seamless','kma_seamless','bom_access_global','cma_grapes_global']
_cf=json.load(open(R+'weather/comb_coefs.json'))['8']; B_CONST,B_LMID,B_LPM=_cf['const'],_cf['lmid'],_cf['lpm']   # logistic combination fitted on 2025-26 settled events at D-1 22Z
# ---- fit blend weights per station on archive (lead-1 forecasts vs CLI high), 2024+
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); fd['cli']=fd.station.map(CLI)
fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}), on=['cli','date'])
P=fd[fd.lead==1].pivot_table(index=['station','date','high'],columns='model',values='tmax_fc').reset_index()
P['doy']=pd.to_datetime(P.date).dt.dayofyear; P['s1']=np.sin(2*np.pi*P.doy/365); P['c1']=np.cos(2*np.pi*P.doy/365)
def design(d, cols):
    X=d[cols].copy(); X=X.apply(lambda r: r.fillna(r.mean()), axis=1); X['s1']=d.s1; X['c1']=d.c1; return sm.add_constant(X, has_constant='add')
FIT={}
for st,g in P.groupby('station'):
    g=g.dropna(subset=['high']); tr=g[g[[m for m in MODELS if m in g]].notna().sum(axis=1)>=4]; cols=[m for m in MODELS if m in tr and tr[m].notna().mean()>0.6]; tr=tr.dropna(subset=cols, thresh=int(0.7*len(cols)))
    if len(tr)<200 or len(cols)<3: print(st,'insufficient archive'); continue
    m=sm.OLS(tr.high, design(tr,cols)).fit(); res=(tr.high-m.predict(design(tr,cols))).values
    FIT[st]=(m,cols,res); print(f'{st}: blend fitted on {len(tr)} days, {len(cols)} models, resid sd {res.std():.2f}F')
# ---- live forecasts (Open-Meteo forecast API), daily max for today/tomorrow per model
def live_fc(st):
    lat,lon,tz,_=ST[st]
    r=requests.get('https://api.open-meteo.com/v1/forecast',params={'latitude':lat,'longitude':lon,'hourly':'temperature_2m','models':','.join(MODELS),'temperature_unit':'fahrenheit','timezone':tz,'forecast_days':3},timeout=60).json()
    h=r['hourly']; t=pd.to_datetime(h['time']); df=pd.DataFrame({k:v for k,v in h.items() if k!='time'}); df['date']=t.date
    d=df.groupby('date').max(); d.columns=[c.replace('temperature_2m_','') for c in d.columns]; return d
def fee(p): return np.ceil(100*0.07*p*(1-p))/100
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
rows=[]
for st,(lat,lon,tz,series) in ST.items():
    try: fc=live_fc(st)
    except Exception as e: print(st,'forecast error',e); continue
    if st not in FIT: continue
    m,cols,res=FIT[st]
    mk=requests.get(B+'/markets',params={'series_ticker':series,'status':'open','limit':200},timeout=60).json().get('markets',[])
    for target,row in fc.iterrows():
        doy=pd.Timestamp(target).dayofyear; d=pd.DataFrame([{**{c:row.get(c,np.nan) for c in cols},'s1':np.sin(2*np.pi*doy/365),'c1':np.cos(2*np.pi*doy/365)}])
        if d[cols].notna().sum(axis=1).iloc[0]<3: continue
        mu=float(m.predict(design(d,cols))[0])
        ev=[x for x in mk if pd.Timestamp('20'+re.search(r'-(\d{2})([A-Z]{3})(\d{2})',x['ticker']).group(1)+'-'+re.search(r'-(\d{2})([A-Z]{3})(\d{2})',x['ticker']).group(2)+'-'+re.search(r'-(\d{2})([A-Z]{3})(\d{2})',x['ticker']).group(3)).date()==target]
        for x in ev:
            lo=-np.inf; hi=np.inf; stt=x.get('strike_type'); fs=x.get('floor_strike'); cs=x.get('cap_strike')
            if stt=='between': lo=fs-0.5; hi=cs+0.5
            elif stt=='less': hi=cs-0.5
            elif stt=='greater': lo=fs+0.5
            v=mu+res; pm=((v>lo)&(v<hi)).mean()
            bid=float(x.get('yes_bid_dollars') or 0); ask=float(x.get('yes_ask_dollars') or 1); mid=(bid+ask)/2 if ask<1 else np.nan
            if not (0<ask<1): continue
            ph=1/(1+np.exp(-(B_CONST+B_LMID*lg(mid)+B_LPM*lg(pm))))
            sig='BUY YES' if ph-ask>THR else ('BUY NO' if bid-ph>THR else '')
            rows.append({'station':st,'target':target,'ticker':x['ticker'],'bracket':x.get('yes_sub_title'),'bid':bid,'ask':ask,'p_model':round(pm,3),'p_comb':round(ph,3),'edge_yes':round(ph-ask-fee(ask),3),'edge_no':round(bid-ph-fee(bid),3),'signal':sig,'blend_mu':round(mu,1)})
out=pd.DataFrame(rows); print(f'\nUTC now {dt.datetime.utcnow():%Y-%m-%d %H:%M}'); pd.set_option('display.width',250)
print(out.sort_values(['station','target','ticker']).to_string(index=False))
print('\nSIGNALS:'); print(out[out.signal!=''].to_string(index=False))
out.to_csv(R+f'kalshi/live_signals_{dt.datetime.utcnow():%Y%m%d_%H%M}.csv', index=False)
