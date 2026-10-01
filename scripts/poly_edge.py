"""Polymarket replication: same blend, same combination, on Polymarket daily-high markets.
Snapshot = D-1 22Z price from CLOB hourly history (midpoint-ish). No exchange fee; assume half-spread cost SPREAD/2 per contract."""
import pandas as pd, numpy as np, statsmodels.api as sm, json, glob, re, sys, datetime as dt
R='/Users/abebraham/Desktop/resume/edge/data/'
SPREAD=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
CITY={'nyc':('LGA','KLGA','America/New_York','F'),'chicago':('ORD','KORD','America/Chicago','F'),'miami':('MIA','KMIA','America/New_York','F'),'dallas':('DAL','KDAL','America/Chicago','F'),'atlanta':('ATL','KATL','America/New_York','F'),'seattle':('SEA','KSEA','America/Los_Angeles','F'),'london':('EGLC','EGLC','Europe/London','C'),'seoul':('RKSI','RKSI','Asia/Seoul','C'),'toronto':('CYYZ','CYYZ','America/Toronto','C')}
MON={m:i for i,m in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december'],1)}
def parse_title(t):
    t=t.replace('°F','').replace('°C','').replace('–','-').replace('—','-').strip()
    if 'or below' in t: return -np.inf, float(t.split()[0])+0.5
    if 'or higher' in t or 'or above' in t: return float(t.split()[0])-0.5, np.inf
    if '-' in t and not t.startswith('-'): a,b=t.split('-'); return float(a)-0.5, float(b)+0.5
    return float(t)-0.5, float(t)+0.5
rows=[]
for f in glob.glob(R+'kalshi/poly_prices/*.json'):
    j=json.load(open(f)); slug=j['slug']; city=re.sub(r'-on-.*','',slug.split('-in-')[1])
    if city not in CITY: continue
    if not j.get('endDate'): continue
    target=pd.Timestamp(j['endDate']).date()
    m=re.search(r'-on-([a-z]+)-(\d+)', slug)
    if m and (MON.get(m.group(1))!=target.month or int(m.group(2))!=target.day): continue
    st,cli_st,tz,unit=CITY[city]
    # snapshot time: D-1 22:00Z
    snap=pd.Timestamp(target,tz='UTC')-pd.Timedelta(hours=2)   # 22Z on D-1
    for mk in j['markets']:
        if not mk['history'] or mk['title'] is None: continue
        lo,hi=parse_title(mk['title']); unit_row='C' if '°C' in mk['title'] else 'F'; op=json.loads(mk['outcomePrices']) if mk['outcomePrices'] else None
        y=1.0 if op and op[0]=='1' else (0.0 if op and op[0]=='0' else np.nan)
        h=pd.DataFrame(mk['history']); h['t']=pd.to_datetime(h.t,unit='s',utc=True); h=h[h.t<=snap]
        if h.empty: continue
        last=h.iloc[-1]
        if (snap-last.t)>pd.Timedelta(hours=3): continue
        rows.append({'slug':slug,'city':city,'station':st,'target':target,'lo':lo,'hi':hi,'y':y,'p':float(last.p),'vol':float(mk.get('volumeNum') or mk.get('volume') or 0),'unit':unit_row})
c=pd.DataFrame(rows).dropna(subset=['y']); print('bracket rows', len(c), 'events', c.slug.nunique(), c.groupby('city').slug.nunique().to_dict())
# sanity: exactly one winning bracket per event
w=c.groupby('slug').y.sum(); print('events with exactly one winner:', (w==1).mean().round(3)); c=c[c.slug.isin(w[w==1].index)]
# ---- blend per station (previous-runs archive lead-1), weights from years < target year
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); CLI={v[0]:v[1] for v in CITY.values()}
fd=fd[fd.station.isin(CLI)]; fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}),on=['cli','date']); fd['year']=pd.to_datetime(fd.date).dt.year
P=fd[fd.lead==1].pivot_table(index=['station','date','year','high'],columns='model',values='tmax_fc').reset_index()
MODELS=[m for m in P.columns if m not in ('station','date','year','high')]
P['doy']=pd.to_datetime(P.date).dt.dayofyear; P['s1']=np.sin(2*np.pi*P.doy/365); P['c1']=np.cos(2*np.pi*P.doy/365)
def design(d, cols):
    X=d[cols].copy(); X=X.apply(lambda r: r.fillna(r.mean()), axis=1); X['s1']=d.s1; X['c1']=d.c1; return sm.add_constant(X, has_constant='add')
RES={}; MU=[]
for st,g in P.groupby('station'):
    g=g[g[MODELS].notna().sum(axis=1)>=4].dropna(subset=['high'])
    for y in [2025,2026]:
        tr=g[g.year<y]; te=g[g.year==y]
        cols=[m for m in MODELS if tr[m].notna().mean()>0.6]; tr=tr.dropna(subset=cols, thresh=int(0.7*len(cols)))
        if len(tr)<200 or len(te)==0: continue
        X=design(tr,cols); ok=X.notna().all(axis=1); m=sm.OLS(tr.high[ok],X[ok]).fit(); res=(tr.high[ok]-m.predict(X[ok])).values
        Xt=design(te,cols); okt=Xt.notna().all(axis=1)
        MU.append(pd.DataFrame({'station':st,'target':te.date.values[okt.values],'mu':m.predict(Xt[okt]).values,'key':st+str(y)})); RES[st+str(y)]=res
        print(f'  blend {st} {y}: {len(cols)} models, {ok.sum()} train days, OOS MAE {np.abs(te.high.values[okt.values]-m.predict(Xt[okt]).values).mean():.2f}')
MU=pd.concat(MU); c=c.merge(MU,on=['station','target'])
def fee0(p): return 0.0
def pbr(mu,k,lo,hi,u):
    v=mu+RES[k]
    if u=='C': v=(v-32)*5/9
    return ((v>lo)&(v<hi)).mean()
c['pm']=[pbr(mu,k,lo,hi,u) for mu,k,lo,hi,u in zip(c.mu,c.key,c.lo,c.hi,c.unit)]
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
c['lmid']=lg(c.p); c['lpm']=lg(c.pm); c['tgt']=pd.to_datetime(c.target); c['month']=c.tgt.dt.to_period('M')
print('events with forecasts', c.slug.nunique(), c.groupby('city').slug.nunique().to_dict(), 'date range', c.target.min(), c.target.max())
# calibration + combination (rolling monthly refit, warm-up 60 events) ; also fixed Kalshi coefficients as a pure transfer test
K=json.load(open(R+'weather/comb_coefs.json'))['8']
preds=[]
for m in sorted(c.month.unique()):
    tr=c[c.month<m]; te=c[c.month==m]
    if tr.slug.nunique()<60: continue
    mod=sm.Logit(tr.y, sm.add_constant(tr[['lmid','lpm']])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[['lmid','lpm']])), b=mod.params.lpm))
c['ph_k']=1/(1+np.exp(-(K['const']+K['lmid']*c.lmid+K['lpm']*c.lpm)))
t=pd.concat(preds) if preds else c.assign(ph=np.nan,b=np.nan)
tk=c.copy()
print(f'\nALL events={tk.slug.nunique()} Brier market={((tk.p-tk.y)**2).mean():.4f} blend={((tk.pm-tk.y)**2).mean():.4f} comb(Kalshi coefs, no fitting)={((tk.ph_k-tk.y)**2).mean():.4f} | rolling OOS events={t.slug.nunique() if preds else 0} comb(rolling)={((t.ph-t.y)**2).mean() if preds else float('nan'):.4f} lpm coef={t.b.mean() if preds else float('nan'):.2f}')
mfull=sm.Logit(c.y, sm.add_constant(c[['lmid','lpm']])).fit(disp=0); print('full-sample logit:', mfull.params.round(3).to_dict(), 'z(lpm)=', round(mfull.tvalues.lpm,1))
for lab,col,t in [('rolling refit','ph',t),('Kalshi coefficients (pure transfer)','ph_k',tk)]:
    if col=='ph' and ('ph' not in t or t.ph.isna().all()): continue
    for thr in [0.05,0.08]:
        ask=t.p+SPREAD/2; bid=t.p-SPREAD/2
        by=t[t[col]-ask>thr]; bn=t[bid-t[col]>thr]
        pnl=pd.concat([by.y-ask[by.index], (1-bn.y)-(1-bid[bn.index])]); sig=pd.concat([by,bn]); sig['pnl']=pnl
        daily=pd.Series(0.0,index=t.target.unique()); dd=sig.groupby('target').pnl.sum(); daily.loc[dd.index]=dd.values
        wint=sig[sig.tgt.dt.month.isin([11,12,1,2,3])]
        print(f'{lab:36s} thr={thr}: N={len(by)}Y+{len(bn)}N pnl/ct={pnl.mean():+.4f} Sharpe={daily.mean()/daily.std()*np.sqrt(365):.2f} win={(pnl>0).mean():.2f} | winter N={len(wint)} pnl/ct={wint.pnl.mean():+.4f} | by city:', {k:(round(v,3),n) for k,v,n in sig.groupby('city').pnl.agg(['mean','size']).itertuples()})
