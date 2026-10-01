"""Cross-venue panel: Kalshi daily-high brackets x hourly timestamps, joined with the Polymarket-implied probability
of the SAME Kalshi bracket at the same hour. Polymarket bracket prices -> normal fit (mu, sigma) per event-hour ->
shifted/convolved with the historical station-offset distribution (Kalshi station minus Polymarket station, prior data only)
-> probability of each Kalshi bracket. Also the reverse (Kalshi-implied probability of each Polymarket bracket)."""
import pandas as pd, numpy as np, json, glob, re, sys
from scipy.stats import norm
from scipy.optimize import minimize
R='/Users/abebraham/Desktop/resume/edge/data/'
# city -> (Kalshi series, Kalshi CLI station, Polymarket station)
PAIR={'nyc':('KXHIGHNY','KNYC','KLGA'),'chicago':('KXHIGHCHI','KMDW','KORD'),'miami':('KXHIGHMIA','KMIA','KMIA'),
      'atlanta':('KXHIGHTATL','KATL','KATL'),'seattle':('KXHIGHTSEA','KSEA','KSEA'),'dallas':('KXHIGHTDAL','KDFW','KDAL')}
MON={m:i for i,m in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december'],1)}
def parse_title(t):
    t=t.replace('°F','').replace('°C','').replace('–','-').replace('—','-').strip()
    if 'or below' in t: return -np.inf, float(t.split()[0])+0.5
    if 'or higher' in t or 'or above' in t: return float(t.split()[0])-0.5, np.inf
    if '-' in t and not t.startswith('-'): a,b=t.split('-'); return float(a)-0.5, float(b)+0.5
    return float(t)-0.5, float(t)+0.5
# ---------- Polymarket long table (bracket x hour)
rows=[]
for f in glob.glob(R+'kalshi/poly_prices/*.json'):
    j=json.load(open(f)); slug=j['slug']
    if '-in-' not in slug: continue
    city=re.sub(r'-on-.*','',slug.split('-in-')[1])
    if city not in PAIR or not j.get('endDate'): continue
    target=pd.Timestamp(j['endDate']).date()
    m=re.search(r'-on-([a-z]+)-(\d+)', slug)
    if m and (MON.get(m.group(1))!=target.month or int(m.group(2))!=target.day): continue
    D0=pd.Timestamp(target,tz='UTC')
    hours=pd.date_range(D0-pd.Timedelta(hours=14), D0+pd.Timedelta(hours=6), freq='h')   # D-1 10Z .. D 06Z
    for mk in j['markets']:
        if not mk.get('history') or not mk.get('title') or '°C' in mk['title']: continue
        lo,hi=parse_title(mk['title']); op=json.loads(mk['outcomePrices']) if mk['outcomePrices'] else None
        y=1.0 if op and op[0]=='1' else (0.0 if op and op[0]=='0' else np.nan)
        h=pd.DataFrame(mk['history']); h['t']=pd.to_datetime(h.t,unit='s',utc=True); h=h.sort_values('t')
        # price as of each grid hour: last point with t <= H+2min, not older than 3h
        idx=np.searchsorted(h.t.values, (hours+pd.Timedelta(minutes=2)).values, side='right')-1
        ok=idx>=0
        tt=h.t.values[np.clip(idx,0,None)]; pp=h.p.values[np.clip(idx,0,None)]
        stale=(hours.values-tt)>np.timedelta64(3,'h')
        age=(hours.values-tt)/np.timedelta64(1,'m')
        for H,o,s,p,a in zip(hours,ok,stale,pp,age):
            if o and not s: rows.append((slug,city,target,lo,hi,y,H,float(p),float(mk.get('volumeNum') or 0),float(a)))
P=pd.DataFrame(rows,columns=['slug','city','target','lo','hi','y','ts','p','bvol','p_age_min'])
P=P.dropna(subset=['y'])
w=P.drop_duplicates(['slug','lo']).groupby('slug').y.sum(); P=P[P.slug.isin(w[w==1].index)]
s=P.groupby(['slug','ts']).p.transform('sum'); P['psum']=s; P=P[(s>0.85)&(s<1.15)]; P['pn']=P.p/P.psum
print('poly rows',len(P),'events',P.slug.nunique(),P.groupby('city').slug.nunique().to_dict())
# ---------- normal fit per (event, hour)
def fit_normal(lo,hi,p):
    c=np.where(np.isinf(lo),hi-1,np.where(np.isinf(hi),lo+1,(lo+hi)/2)); mu0=float((c*p).sum()/p.sum())
    def loss(th):
        mu,ls=th; sg=np.exp(ls); q=norm.cdf((hi-mu)/sg)-norm.cdf((lo-mu)/sg); return ((q-p)**2).sum()
    r=minimize(loss,[mu0,np.log(2.5)],method='Nelder-Mead',options={'xatol':1e-3,'fatol':1e-7,'maxiter':400})
    mu,ls=r.x; return mu,float(np.exp(ls)),r.fun
def fit_table(df,keys):
    out=[]
    for k,g in df.groupby(keys,sort=False):
        if len(g)<4: continue
        mu,sg,l=fit_normal(g.lo.values,g.hi.values,g.pn.values); out.append((*k,mu,sg,l,len(g)))
    return pd.DataFrame(out,columns=keys+['mu','sig','loss','nb'])
FP=fit_table(P,['slug','ts']); FP=FP.merge(P.drop_duplicates('slug')[['slug','city','target']],on='slug')
FP=FP.merge(P.groupby(['slug','ts']).p_age_min.max().rename('p_age_max').reset_index(),on=['slug','ts'])
print('poly fits',len(FP),'median sigma',FP.sig.median().round(2),'median loss',FP.loss.median())
# ---------- Kalshi panel
K=pd.read_parquet(R+'weather/candles_ev.parquet'); K=K[K.series.isin([v[0] for v in PAIR.values()])&(K.kind=='high')].copy()
K['city']=K.series.map({v[0]:k for k,v in PAIR.items()}); K['ts']=K.time
K=K[(K.bid>=0)&(K.ask<=1)&(K.ask>K.bid-1e-9)]; K['mid']=(K.bid+K.ask)/2
K=K[K.target.isin(set(FP.target))]
s=K.groupby(['event_ticker','ts']).mid.transform('sum'); K['ksum']=s; K['pn']=K.mid/s
FK=fit_table(K[(s>0.85)&(s<1.15)],['event_ticker','ts']); print('kalshi fits',len(FK))
# ---------- station offset distribution D = T_Kalshi - T_Poly, from CLI, prior data only (expanding, min 90 days)
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); cli['date']=pd.to_datetime(cli.date).dt.date
OFF={}
for city,(ser,ks,ps) in PAIR.items():
    if ks==ps: continue
    a=cli[cli.station==ks][['date','high']].merge(cli[cli.station==ps][['date','high']],on='date',suffixes=('_k','_p')).sort_values('date')
    a['d']=a.high_k-a.high_p; a=a.dropna(subset=['d'])
    a['m']=a.d.shift(1).expanding(90).mean(); a['s']=a.d.shift(1).expanding(90).std()
    OFF[city]=a.set_index('date')[['m','s']]
    print(f'offset {city}: {ks}-{ps} mean {a.d.mean():+.2f} sd {a.d.std():.2f} (n={len(a)})')
def off(city,target):
    if city not in OFF: return 0.0,0.0
    o=OFF[city]; 
    if target in o.index and not np.isnan(o.loc[target,'m']): return float(o.loc[target,'m']),float(o.loc[target,'s'])
    prior=o[o.index<target].dropna()
    return (float(prior.m.iloc[-1]),float(prior.s.iloc[-1])) if len(prior) else (np.nan,np.nan)
# ---------- join: Kalshi bracket-hour with Poly fit at same hour
X=K.merge(FP[['city','target','ts','mu','sig','loss','nb','p_age_max']].rename(columns={'mu':'mu_p','sig':'sig_p','loss':'loss_p','nb':'nb_p'}),on=['city','target','ts'])
X=X.merge(FK[['event_ticker','ts','mu','sig','loss']].rename(columns={'mu':'mu_k','sig':'sig_k','loss':'loss_k'}),on=['event_ticker','ts'],how='left')
od=pd.DataFrame([(c,t,*off(c,t)) for c,t in X[['city','target']].drop_duplicates().itertuples(index=False)],columns=['city','target','dm','ds'])
X=X.merge(od,on=['city','target']); X=X.dropna(subset=['dm'])
sx=np.sqrt(X.sig_p**2+X.ds**2); X['px']=norm.cdf((X.hi-X.mu_p-X.dm)/sx)-norm.cdf((X.lo-X.mu_p-X.dm)/sx)
# Kalshi's own normal-fit prob of its bracket (control: is the normal smoothing itself informative?)
X['pk_fit']=norm.cdf((X.hi-X.mu_k)/X.sig_k)-norm.cdf((X.lo-X.mu_k)/X.sig_k)
X['mu_p_adj']=X.mu_p+X.dm
X.to_parquet(R+'kalshi/xvenue_panel.parquet')
print('panel rows',len(X),'events',X.event_ticker.nunique(),X.groupby('city').event_ticker.nunique().to_dict(),'hours/event',round(len(X)/X.event_ticker.nunique()/6,1))
print('target range',X.target.min(),X.target.max())
# ---------- reverse panel: Poly brackets with Kalshi-implied prob
Y=P.merge(FK.merge(K.drop_duplicates('event_ticker')[['event_ticker','city','target']],on='event_ticker')[['city','target','ts','mu','sig']].rename(columns={'mu':'mu_k','sig':'sig_k'}),on=['city','target','ts'])
Y=Y.merge(od,on=['city','target']).dropna(subset=['dm'])
sy=np.sqrt(Y.sig_k**2+Y.ds**2); Y['pxk']=norm.cdf((Y.hi-(Y.mu_k-Y.dm))/sy)-norm.cdf((Y.lo-(Y.mu_k-Y.dm))/sy)
Y.to_parquet(R+'kalshi/xvenue_panel_poly.parquet'); print('reverse panel rows',len(Y),'events',Y.slug.nunique())
