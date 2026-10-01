"""Polymarket side: does the Kalshi-implied probability (lx) carry information beyond (a) the Polymarket price and (b) the NWP blend?
Walk-forward trades on Polymarket (no fee) with spread and delay stress. Speed of adjustment."""
import pandas as pd, numpy as np, statsmodels.api as sm, json, warnings, sys; warnings.filterwarnings('ignore')
from scipy.stats import norm
R='/Users/abebraham/Desktop/resume/edge/data/'
Y=pd.read_parquet(R+'kalshi/xvenue_panel_poly.parquet').dropna(subset=['pxk','mu_k']); Y=Y[(Y.pn>0.005)&(Y.pn<0.995)].copy()
CITY={'nyc':('LGA','KLGA'),'chicago':('ORD','KORD'),'miami':('MIA','KMIA'),'dallas':('DAL','KDAL'),'atlanta':('ATL','KATL'),'seattle':('SEA','KSEA')}
Y['station']=Y.city.map(lambda c: CITY[c][0])
# ---- blend (copied from poly_edge.py): per-station OLS of CLI high on lead-1 model maxes, weights from years < target year
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
cli=pd.read_parquet(R+'weather/cli_daily.parquet'); CLI={v[0]:v[1] for v in CITY.values()}
fd=fd[fd.station.isin(CLI)].copy(); fd['cli']=fd.station.map(CLI); fd=fd.merge(cli[['station','date','high']].rename(columns={'station':'cli'}),on=['cli','date']); fd['year']=pd.to_datetime(fd.date).dt.year
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
MU=pd.concat(MU); Y=Y.merge(MU,on=['station','target'],how='left')
pm=np.full(len(Y),np.nan)
for (k,mu),idx in Y.groupby(['key','mu']).indices.items():
    v=mu+RES[k]; sub=Y.iloc[idx]; pm[idx]=[((v>l)&(v<h)).mean() for l,h in zip(sub.lo,sub.hi)]
Y['pm']=pm
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
Y['lp']=lg(Y.pn); Y['lx']=lg(Y.pxk); Y['lpm']=lg(Y.pm); Y['tgt']=pd.to_datetime(Y.target); Y['month']=Y.tgt.dt.to_period('M')
Y['hour']=Y.ts.dt.hour; Y['dayrel']=(Y.ts.dt.normalize()-Y.tgt.dt.tz_localize('UTC')).dt.days
Y['season']=np.where(Y.tgt.dt.month.isin([11,12,1,2,3]),'winter',np.where(Y.tgt.dt.month.isin([6,7,8,9]),'summer','shoulder'))
# next-hour Poly price for delayed execution
Y=Y.sort_values(['slug','lo','ts']); g=Y.groupby(['slug','lo']); Y['pn1']=g.pn.shift(-1); Y['ok1']=(g.ts.shift(-1)-Y.ts)==pd.Timedelta(hours=1)
Yb=Y.dropna(subset=['pm']).copy()
print('rows',len(Y),'with blend',len(Yb),'events',Yb.slug.nunique(),Yb.groupby('city').slug.nunique().to_dict(),Yb.groupby('season').slug.nunique().to_dict())
def clus(y,X,g): return sm.Logit(y,sm.add_constant(X)).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(g)[0]})
print('\n=== information beyond the Polymarket price (event-clustered logits) ===')
for lab,sel in [('ALL',Yb.dayrel.isin([-1,0])),('D-1 14-21Z',(Yb.dayrel==-1)&Yb.hour.between(14,21)),('D-1 22Z-D 03Z',((Yb.dayrel==-1)&(Yb.hour>=22))|((Yb.dayrel==0)&(Yb.hour<=3))),('D 04-06Z',(Yb.dayrel==0)&Yb.hour.between(4,6))]:
    s=Yb[sel]
    for cols in [['lp','lx'],['lp','lpm'],['lp','lx','lpm']]:
        m=clus(s.y,s[cols],s.slug); print(f'{lab:14s} n={len(s):6d} {"+".join(cols):10s} ' + ' '.join(f'{c}={m.params[c]:.2f}(z={m.tvalues[c]:.1f})' for c in cols) + f' Brier {((m.predict(sm.add_constant(s[cols]))-s.y)**2).mean():.4f} (mkt {((s.pn-s.y)**2).mean():.4f})')
print('\n=== bracket-level error correction: d pn(t+1) ~ gap(t) = pxk - pn  (share of gap closed per hour) ===')
s=Yb[Yb.ok1]; s=s.assign(d1=s.pn1-s.pn, gap=s.pxk-s.pn)
for lab,ss in [('all',s),('D-1',s[s.dayrel==-1]),('D',s[s.dayrel==0])]:
    m=sm.OLS(ss.d1,sm.add_constant(ss[['gap']])).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(ss.slug)[0]}); print(f'{lab:5s} n={len(ss)} gap coef {m.params.gap:+.3f} (t={m.tvalues.gap:.1f})  -> half-life {np.log(0.5)/np.log(1-m.params.gap) if 0<m.params.gap<1 else np.nan:.1f} h')
# reverse: does the Poly-implied gap move Kalshi? (uses Kalshi panel)
print('\n=== walk-forward trading on Polymarket: monthly refit on prior months, warm-up 40 events, first signal per bracket-side, hold to settlement ===')
def wf(df,cols,warm=40):
    out=[]
    for m in sorted(df.month.unique()):
        tr=df[df.month<m]; te=df[df.month==m]
        if tr.slug.nunique()<warm: continue
        mod=sm.Logit(tr.y,sm.add_constant(tr[cols])).fit(disp=0); out.append(te.assign(ph=mod.predict(sm.add_constant(te[cols]))))
    return pd.concat(out)
def trade(t,label,thr,hs,delay=False,show_city=True):
    if delay: t=t[t.ok1]; px=t.pn1
    else: px=t.pn
    ask=px+hs; bid=px-hs
    by=t[t.ph-(t.pn+hs)>thr].assign(pnl=lambda d: d.y-(px[d.index]+hs)).sort_values('ts').drop_duplicates(['slug','lo'])
    bn=t[(t.pn-hs)-t.ph>thr].assign(pnl=lambda d: (1-d.y)-(1-(px[d.index]-hs))).sort_values('ts').drop_duplicates(['slug','lo'])
    a=pd.concat([by,bn])
    if len(a)<5: print(label,'N',len(a)); return a
    daily=pd.Series(0.0,index=t.target.unique()); dd=a.groupby('target').pnl.sum(); daily.loc[dd.index]=dd.values
    print(f'{label:48s} thr={thr:.2f} hs={hs:.2f}{" DELAY1h" if delay else "        "} N={len(a):5d} pnl/ct={a.pnl.mean():+.4f} t={a.pnl.mean()/a.pnl.std()*np.sqrt(len(a)):4.1f} win={(a.pnl>0).mean():.2f} dSharpe={daily.mean()/daily.std()*np.sqrt(365):.2f}' + ((' | '+' '.join(f'{c}:{v:+.3f}({n})' for c,v,n in a.groupby('city').pnl.agg(['mean','size']).itertuples())) if show_city else ''))
    return a
T={ '+'.join(c): wf(Yb,c) for c in [['lp','lx'],['lp','lpm'],['lp','lx','lpm']] }
for k,t in T.items():
    for thr in [0.05,0.08]:
        for hs in [0.02,0.03,0.05]:
            a=trade(t,k,thr,hs,show_city=(hs==0.02))
        trade(t,k,thr,0.02,delay=True,show_city=False)
a=trade(T['lp+lx'],'lp+lx by season',0.05,0.02,show_city=False)
for ssn,g in a.groupby('season'): print(f'   {ssn:8s} N={len(g):4d} pnl/ct={g.pnl.mean():+.4f} t={g.pnl.mean()/g.pnl.std()*np.sqrt(len(g)):.1f}')
for blk,sel in [('D-1 14-21Z',(a.dayrel==-1)&a.hour.between(14,21)),('D-1 22Z-D 03Z',((a.dayrel==-1)&(a.hour>=22))|((a.dayrel==0)&(a.hour<=3))),('D 04-06Z',(a.dayrel==0)&a.hour.between(4,6))]:
    g=a[sel]; print(f'   {blk:14s} N={len(g):4d} pnl/ct={g.pnl.mean():+.4f}')
# incremental: trades of lp+lx+lpm that lp+lpm alone would NOT have taken
t3=T['lp+lx+lpm']; t2=T['lp+lpm'].set_index(['slug','lo','ts']).ph
t3=t3.assign(ph2=t2.reindex(pd.MultiIndex.from_frame(t3[['slug','lo','ts']])).values)
hs=0.02; thr=0.05
new=t3[((t3.ph-(t3.pn+hs)>thr)&~(t3.ph2-(t3.pn+hs)>thr))|(((t3.pn-hs)-t3.ph>thr)&~((t3.pn-hs)-t3.ph2>thr))]
newY=new[new.ph-(new.pn+hs)>thr].assign(pnl=lambda d:d.y-(d.pn+hs)); newN=new[(new.pn-hs)-new.ph>thr].assign(pnl=lambda d:(1-d.y)-(1-(d.pn-hs)))
nn=pd.concat([newY,newN]).sort_values('ts').drop_duplicates(['slug','lo'])
print(f'\nINCREMENTAL trades (3-var signal fires, blend-only does not): N={len(nn)} pnl/ct={nn.pnl.mean():+.4f} t={nn.pnl.mean()/nn.pnl.std()*np.sqrt(len(nn)):.1f}')
Yb.to_parquet(R+'kalshi/xvenue_poly_scored.parquet')
