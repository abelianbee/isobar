"""FINAL pre-registered evaluation of the cross-venue edge (Kalshi -> Polymarket).
Development sample: events with target <= 2026-03-31 (everything analysed while designing the test).
Out-of-sample: target >= 2026-04-01 (Polymarket + Kalshi data pulled AFTER the design was fixed).
Specs fixed before the OOS data were seen:
  S1 model-based: logit(y) ~ logit(poly mid) + logit(Kalshi-implied bracket prob), monthly refit on prior months, warm-up 40 events,
     buy Polymarket YES/NO when |p_hat - price| > 0.05 with 2c half-spread, first signal per bracket-side, hold to settlement.
  S2 model-free (identical contracts, same station, same 2F bins): buy Polymarket toward Kalshi mid when |Kalshi mid - Poly mid| > 0.08, 2c half-spread.
Also reported: thr 0.08 for S1; 3-var (adds NWP blend); season/city/month splits; sizing by participation."""
import pandas as pd, numpy as np, statsmodels.api as sm, json, warnings, sys; warnings.filterwarnings('ignore')
R='/Users/abebraham/Desktop/resume/edge/data/'; OOS=pd.Timestamp('2026-04-01')
Y=pd.read_parquet(R+'kalshi/xvenue_poly_scored.parquet'); Y['tgt']=pd.to_datetime(Y.target)
Kh=pd.read_parquet(R+'kalshi/xvenue_panel.parquet').drop_duplicates(['city','target','ts'])[['city','target','ts','h']]; Y=Y.merge(Kh,on=['city','target','ts'],how='left')
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
def wf(df,cols,warm=40):
    out=[]
    for m in sorted(df.month.unique()):
        tr=df[df.month<m]; te=df[df.month==m]
        if tr.slug.nunique()<warm: continue
        mod=sm.Logit(tr.y,sm.add_constant(tr[cols])).fit(disp=0); out.append(te.assign(ph=mod.predict(sm.add_constant(te[cols])),b_lx=mod.params.get('lx',np.nan)))
    return pd.concat(out)
def signals(t,thr,hs):
    by=t[t.ph-(t.p+hs)>thr].copy(); by['side']='Y'; by['cost']=by.p+hs; by['pnl']=by.y-by.cost; by['edge']=by.ph-by.cost
    bn=t[(t.p-hs)-t.ph>thr].copy(); bn['side']='N'; bn['cost']=1-(bn.p-hs); bn['pnl']=(1-bn.y)-bn.cost; bn['edge']=(1-bn.ph)-bn.cost
    return pd.concat([by,bn]).sort_values('ts').drop_duplicates(['slug','lo'])
def stats(a,alldays):
    if len(a)==0: return dict(N=0)
    d=pd.Series(0.0,index=alldays); dd=a.groupby('target').pnl.sum(); d.loc[dd.index]=dd.values
    return dict(N=len(a),pnl_ct=round(a.pnl.mean(),4),t=round(a.pnl.mean()/a.pnl.std()*np.sqrt(len(a)),1),win=round((a.pnl>0).mean(),3),roi=round(a.pnl.sum()/a.cost.sum(),3),sharpe_1lot=round(d.mean()/d.std()*np.sqrt(365),2),days=len(alldays),events=a.slug.nunique())
def show(lab,a,alldays): s=stats(a,alldays); print(f'{lab:52s} '+' '.join(f'{k}={v}' for k,v in s.items())); return s
T2=wf(Y,['lp','lx']); T3=wf(Y,['lp','lx','lpm']); Tb=wf(Y,['lp','lpm'])
out={}
for per,sel in [('DEV (<=2026-03)',lambda t: t.tgt<OOS),('OOS (2026-04..09)',lambda t: t.tgt>=OOS),('ALL',lambda t: t.tgt.notna())]:
    print(f'\n=== {per} ===')
    for lab,T in [('S1 lp+lx',T2),('S1b lp+lx+blend',T3),('blend only lp+lpm',Tb)]:
        t=T[sel(T)]; alld=sorted(t.target.unique())
        if len(t)==0: continue
        for thr in [0.05,0.08]:
            a=signals(t,thr,0.02); s=show(f'{lab} thr {thr} hs 2c',a,alld); out[f'{per}|{lab}|{thr}']=s
            if lab=='S1 lp+lx' and thr==0.05:
                print('   by season:', {k:(round(v,4),n) for k,v,n in a.groupby('season').pnl.agg(['mean','size']).itertuples()})
                print('   by city  :', {k:(round(v,4),n) for k,v,n in a.groupby('city').pnl.agg(['mean','size']).itertuples()})
                print('   by month :', {str(k):(round(v,3),n) for k,v,n in a.groupby('month').pnl.agg(['mean','size']).itertuples()})
                print('   Kalshi h<=8:', {str(k):(round(v,4),n) for k,v,n in a.groupby(a.h<=8).pnl.agg(['mean','size']).itertuples()})
                if per=='ALL': A_all=a
                if per.startswith('OOS'): A_oos=a
# ---- S2 model-free identical contracts
J=pd.read_parquet(R+'kalshi/xvenue_same.parquet'); J['tgt']=pd.to_datetime(J.target)
def mf(J,thr,hs):
    t=J.sort_values('ts'); by=t[t.mid-t.p>thr].drop_duplicates('ticker'); bn=t[t.p-t.mid>thr].drop_duplicates('ticker')
    a=pd.concat([by.assign(pnl=by.y_p-(by.p+hs),cost=by.p+hs,side='Y'),bn.assign(pnl=(1-bn.y_p)-(1-(bn.p-hs)),cost=1-(bn.p-hs),side='N')]); a['slug']=a.slug; return a
print('\n=== S2 model-free identical contracts (MIA/ATL/SEA), thr .08 hs 2c ===')
for per,sel in [('DEV',J.tgt<OOS),('OOS',J.tgt>=OOS),('ALL',J.tgt.notna())]:
    s=J[sel]
    if len(s)==0: continue
    for thr in [0.05,0.08,0.12]:
        a=mf(s,thr,0.02); st=show(f'{per} S2 thr {thr}',a,sorted(s.target.unique())); out[f'{per}|S2|{thr}']=st
        if thr==0.08: print('   by city:', {k:(round(v,4),n) for k,v,n in a.groupby('city').pnl.agg(['mean','size']).itertuples()})
# ---- sizing by participation: size = f x total $ volume of the bracket (whole market life), capped at $2,000/signal; P&L in $
print('\n=== $ sizing (S1 lp+lx thr .05): contracts = f x bracket total volume ($) / price, cap $2,000 notional per signal ===')
for f_ in [0.02,0.05,0.10]:
    a=A_all.copy(); a['notional']=np.minimum(f_*a.bvol,2000.0); a['contracts']=a.notional/a.cost; a['usd']=a.contracts*a.pnl
    d=a.groupby('target').usd.sum(); alld=pd.Series(0.0,index=sorted(Y.target.unique())); alld.loc[d.index]=d.values
    dev=a[a.tgt<OOS]; oos=a[a.tgt>=OOS]
    print(f'f={f_:.2f}: total ${a.usd.sum():,.0f} over {len(alld)} days (${a.usd.sum()/len(alld)*365:,.0f}/yr), avg notional/day ${a.groupby("target").notional.sum().mean():,.0f}, $Sharpe {alld.mean()/alld.std()*np.sqrt(365):.2f}, worst day ${alld.min():,.0f} | DEV ${dev.usd.sum():,.0f} OOS ${oos.usd.sum():,.0f}')
# ---- export the panel for review
A=A_all; daily=A.groupby('target').pnl.sum().reindex(sorted(Y.target.unique()),fill_value=0.0)
exp={'daily':[{'d':str(k),'pnl':round(v,4)} for k,v in daily.items()],
     'city':[{'city':k,'n':int(n),'pnl_ct':round(v,4),'win':round(w,3)} for k,v,n,w in A.groupby('city').pnl.agg(['mean','size',lambda s:(s>0).mean()]).itertuples()],
     'month':[{'m':str(k),'n':int(n),'pnl_ct':round(v,4)} for k,v,n in A.groupby('month').pnl.agg(['mean','size']).itertuples()],
     'edge_bins':[{'bin':str(k),'n':int(n),'pnl_ct':round(v,4)} for k,v,n in A.groupby(pd.cut(A.edge,[0.05,0.07,0.10,0.15,0.25,1]),observed=True).pnl.agg(['mean','size']).itertuples()],
     'hblock':[{'h':str(k),'n':int(n),'pnl_ct':round(v,4)} for k,v,n in A.groupby(pd.cut(A.h,[0,2,4,8,12,20,40]),observed=True).pnl.agg(['mean','size']).itertuples()],
     'stats':out,'oos_start':str(OOS.date()),'dev_stats':stats(A[A.tgt<OOS],sorted(Y[Y.tgt<OOS].target.unique())),'oos_stats':stats(A[A.tgt>=OOS],sorted(Y[Y.tgt>=OOS].target.unique())) if (A.tgt>=OOS).any() else {}}
json.dump(exp,open('/Users/abebraham/Desktop/resume/edge/results/xvenue_export.json','w'),default=str)
A.drop(columns=[c for c in A.columns if A[c].dtype.name=='category'],errors='ignore').to_parquet(R+'kalshi/xvenue_trades_final.parquet')
print('\nexported results/xvenue_export.json')
