"""Release build, strategy 2 (cross-venue: Kalshi-implied probability -> Polymarket brackets).
Reads the panels built by xvenue_build.py / xvenue_poly2.py. Writes results/release/xvenue.json and xvenue_trades.parquet.
Development sample: target <= 2026-03-31. Holdout: target >= 2026-04-01 (post-specification holdout; chronology self-reported, see REPORT)."""
import sys, json, glob, re, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent)); import edge_lib as E
OUT=E.RESULTS/'release'; OUT.mkdir(exist_ok=True,parents=True); HOLD=pd.Timestamp('2026-04-01'); THR=0.05; HS=0.02
CITIES=['nyc','chicago','miami','atlanta','seattle','dallas']; SAME=['miami','atlanta','seattle']
Y=pd.read_parquet(E.DATA/'kalshi/xvenue_poly_scored.parquet'); Y['tgt']=pd.to_datetime(Y.target)
X=pd.read_parquet(E.DATA/'kalshi/xvenue_panel.parquet')
Kh=X.drop_duplicates(['city','target','ts'])[['city','target','ts','h']+(['p_age_max'] if 'p_age_max' in X else [])]; Y=Y.merge(Kh,on=['city','target','ts'],how='left')
Y['event']=Y.slug
# ---------------- reconciliation from discovery to inclusion
files=[f for f in glob.glob(str(E.DATA/'kalshi/poly_prices/*.json')) if any(f'-in-{c}-on-' in f for c in CITIES)]
rec={'poly_files_6_cities':len(files)}
def _endok(f):
    j=json.load(open(f)); return bool(j.get('endDate'))
nend=sum(_endok(f) for f in files); rec['with_endDate']=nend
P0=pd.read_parquet(E.DATA/'kalshi/xvenue_panel_poly.parquet')
rec['events_with_hourly_prices_and_one_winner']=int(P0.slug.nunique()); rec['events_matched_to_kalshi_hours']=int(X.groupby(['city','target']).ngroups); rec['events_scored_with_blend']=int(Y.slug.nunique())
rec['note']='events without endDate, with no hourly history in the window, whose strip does not sum to 0.85-1.15, or with != 1 winning bracket are excluded; Kalshi match requires a candle in the same hour.'
# ---------------- helpers
def wf(df, cols, warm=40, weighted=False, frozen_until=None):
    """Monthly walk-forward. frozen_until: fit once on months <= frozen_until and apply to all later months (frozen-parameter variant)."""
    out=[]
    if frozen_until is not None:
        tr=df[df.tgt<=frozen_until]; te=df[df.tgt>frozen_until]
        mod=sm.Logit(tr.y,sm.add_constant(tr[cols])).fit(disp=0); return te.assign(ph=mod.predict(sm.add_constant(te[cols])),train_cutoff=str(tr.tgt.max().date()))
    for m in sorted(df.month.unique()):
        tr=df[df.month<m]; te=df[df.month==m]
        if tr.slug.nunique()<warm: continue
        assert tr.tgt.max()<te.tgt.min()
        Xtr=sm.add_constant(tr[cols])
        if weighted:
            w=1/tr.groupby(['slug','lo']).lo.transform('size'); mod=sm.GLM(tr.y,Xtr,family=sm.families.Binomial(),var_weights=w).fit()
        else: mod=sm.Logit(tr.y,Xtr).fit(disp=0)
        out.append(te.assign(ph=mod.predict(sm.add_constant(te[cols])),train_cutoff=str(tr.tgt.max().date())))
    return pd.concat(out)
def signals(t,thr=THR,hs=HS):
    by=t[t.ph-(t.p+hs)>thr].copy(); by['side']='Y'; by['cost']=by.p+hs; by['pnl']=by.y-by.cost; by['edge']=by.ph-by.cost
    bn=t[(t.p-hs)-t.ph>thr].copy(); bn['side']='N'; bn['cost']=1-(bn.p-hs); bn['pnl']=(1-bn.y)-bn.cost; bn['edge']=(1-bn.ph)-bn.cost
    return pd.concat([by,bn]).sort_values('ts').drop_duplicates(['slug','lo'])
def per(t,lab): return {'DEV':t[t.tgt<HOLD],'HOLDOUT':t[t.tgt>=HOLD],'ALL':t}[lab]
res={'spec':dict(signal='logit(y) ~ logit(Polymarket mid) + logit(Kalshi-implied prob of the same bracket); monthly refit on all prior months, 40-event warm-up',
                 execution='raw Polymarket midpoint (verified = CLOB /midpoint) plus 2c half-spread; first signal per bracket-side; hold to settlement',
                 threshold=THR, half_spread=HS, holdout_start=str(HOLD.date()), holdout_label='post-specification holdout: specification fixed on data through 2026-03; Apr-Sep 2026 Polymarket data downloaded afterwards (chronology self-reported; no external timestamp)',
                 cities=CITIES, same_station_cities=SAME),'reconciliation':rec}
all_days={lab:sorted(per(Y,lab).tgt.unique()) for lab in ['DEV','HOLDOUT','ALL']}
def info_logit(s,cols):
    m=sm.Logit(s.y,sm.add_constant(s[cols])).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(s.slug)[0]})
    return {c:dict(coef=round(float(m.params[c]),3),z=round(float(m.tvalues[c]),2)) for c in cols}|{'n':int(len(s)),'brier_model':round(float(((m.predict(sm.add_constant(s[cols]))-s.y)**2).mean()),4),'brier_market':round(float(((s.pn-s.y)**2).mean()),4)}
res['information']={lab:{'+'.join(c):info_logit(per(Y,lab),c) for c in [['lp','lx'],['lp','lpm'],['lp','lx','lpm']]} for lab in ['DEV','HOLDOUT','ALL']}
# price-age control (audit 4.2)
if 'p_age_min' in Y:
    fresh=Y[Y.p_age_min<=15]; res['information']['ALL_poly_price_age_le_15min']=info_logit(fresh,['lp','lx'])|{'share_rows_kept':round(float(len(fresh)/len(Y)),3)}
    res['information']['ALL_poly_price_age_gt_60min']=info_logit(Y[Y.p_age_min>60],['lp','lx'])
    res['price_age_minutes_quantiles']={str(q):round(float(Y.p_age_min.quantile(q)),1) for q in [0.5,0.75,0.9,0.95]}
# ---------------- trading: adaptive walk-forward (primary), frozen coefficients, equal-weight, spec curve
T={'lp+lx':wf(Y,['lp','lx']),'lp+lx+lpm':wf(Y,['lp','lx','lpm']),'lp+lpm':wf(Y,['lp','lpm'])}
Tfro=wf(Y,['lp','lx'],frozen_until=HOLD-pd.Timedelta(days=1)); Tw=wf(Y,['lp','lx'],weighted=True)
res['trading']={}
for k,t in T.items():
    for lab in ['DEV','HOLDOUT','ALL']:
        a=signals(per(t,lab)); res['trading'][f'{k}|{lab}']=E.summarize_trades(a,all_days[lab],event_col='event',label=f'{k} {lab}')
res['trading']['lp+lx|HOLDOUT|frozen_coefficients_fit_through_2026-03']=E.summarize_trades(signals(Tfro),all_days['HOLDOUT'],label='frozen')
res['trading']['lp+lx|ALL|equal_weight_per_bracket_GLM']=E.summarize_trades(signals(Tw),all_days['ALL'],label='equal weight')
if 'p_age_min' in Y:
    ta=T['lp+lx']; res['trading']['lp+lx|ALL|poly_price_age_le_15min']=E.summarize_trades(signals(ta[ta.p_age_min<=15]),all_days['ALL'],label='fresh quotes only')
grid={}
for thr in [0.03,0.05,0.08,0.12]:
    for hs in [0.01,0.02,0.03,0.05]:
        for lab in ['DEV','HOLDOUT','ALL']:
            s=E.summarize_trades(signals(per(T['lp+lx'],lab),thr,hs),all_days[lab],label=''); grid[f'thr{thr}_hs{hs}_{lab}']={k:s[k] for k in ['N','pnl_ct','t_event','sharpe_capital'] if k in s}
res['spec_curve']=grid
A=signals(T['lp+lx']); A['dev_or_holdout']=np.where(A.tgt<HOLD,'DEV','HOLDOUT')
res['breakdowns']={'by_city':{c:dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4),t_event=round(E.clustered_t(g.pnl,g.event)[2],2),share_of_total_pnl=round(float(g.pnl.sum()/A.pnl.sum()),3)) for c,g in A.groupby('city')},
    'by_city_holdout':{c:dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for c,g in A[A.tgt>=HOLD].groupby('city')},
    'by_season':{c:dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for c,g in A.groupby('season')},
    'by_month':{str(c):dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for c,g in A.groupby('month')},
    'by_kalshi_hours_since_open':{str(c):dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for c,g in A.groupby(pd.cut(A.h,[0,2,4,8,12,20,40]),observed=True)},
    'by_edge_bin':{str(c):dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for c,g in A.groupby(pd.cut(A.edge,[0.05,0.07,0.10,0.15,0.25,1]),observed=True)},
    'loco_t_event':{c:E.summarize_trades(A[A.city!=c],all_days['ALL'],label='')['t_event'] for c in CITIES},
    'loco_holdout_pnl_ct':{c:E.summarize_trades(A[(A.city!=c)&(A.tgt>=HOLD)],all_days['HOLDOUT'],label='')['pnl_ct'] for c in CITIES}}
# markouts (audit 4.12): Polymarket price change after entry, before and after paying the spread again
Ys=Y.set_index(['slug','lo','ts']).p; mk={}
for k in [3,6,12]:
    idx=pd.MultiIndex.from_arrays([A.slug,A.lo,A.ts+pd.Timedelta(hours=k)]); pk=Ys.reindex(idx).values; ok=~np.isnan(pk); a=A[ok]; pk=pk[ok]
    sgn=np.where(a.side=='Y',1,-1); mo=sgn*(pk-a.p.values)   # signed price move in the trade's favour, before spread
    mk[f'{k}h']=dict(n=int(len(a)),markout_before_spread=round(float(mo.mean()),4),t_event=round(E.clustered_t(mo,a.event)[2],2),exit_pnl_after_two_spreads=round(float((mo-2*HS).mean()),4),hold_to_settlement_same_trades=round(float(a.pnl.mean()),4))
res['markouts']=mk
# ---------------- identical contracts: settlement basis, VECM information shares, model-free trade, alternative strip mapping
K=X[X.city.isin(SAME)][['city','target','ts','h','ticker','lo','hi','y','bid','ask','mid','mu_k','sig_k']].rename(columns={'y':'y_k'})
Pp=P0[P0.city.isin(SAME)][['city','target','ts','slug','lo','hi','y','p','pn']].rename(columns={'y':'y_p'})
J=K.merge(Pp,on=['city','target','ts','lo','hi']); J=J[(J.ask>0)&(J.ask<1)&(J.p>0)&(J.p<1)].copy(); J['tgt']=pd.to_datetime(J.target)
u=J.drop_duplicates('ticker'); res['identical_contracts']={'rows':int(len(J)),'events':int(J.groupby(['city','target']).ngroups),
    'settlement_disagreement_share_brackets':round(float((u.y_k!=u.y_p).mean()),4),'settlement_disagreement_by_city':{c:round(float((g.y_k!=g.y_p).mean()),4) for c,g in u.groupby('city')},
    'events_with_any_disagreement':round(float(u.assign(d=u.y_k!=u.y_p).groupby(['city','target']).d.any().mean()),4)}
# Information shares. Identical-contract series are short (median 16 hourly observations), so a per-contract VECM with lags is not
# estimable; we (a) fit a lag-free VECM (k_ar_diff=0, rank 1) per contract with >=12 contiguous obs and aggregate medians with an event
# bootstrap, and (b) validate the pooled error-correction estimator of the Hasbrouck share on simulated cointegrated pairs with known shares.
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
J=J.sort_values(['ticker','ts']); isk=[]; jo=[]; nfail=0
def is_from(alpha,Om):
    psi=np.array([alpha[1],-alpha[0]]);
    if abs(psi).sum()<1e-12 or psi.sum()==0: return None
    psi=psi/psi.sum()
    def IS(order):
        idx=list(order); F=np.linalg.cholesky(Om[np.ix_(idx,idx)]); c_=psi[idx]@F; sh=c_**2/(c_**2).sum(); out=np.empty(2); out[idx]=sh; return out
    i1,i2=IS([0,1]),IS([1,0]); return min(i1[0],i2[0]),max(i1[0],i2[0])
for tk,g in J.groupby('ticker'):
    if len(g)<12 or not (g.ts.diff().dropna()==pd.Timedelta(hours=1)).all(): continue
    Z=np.column_stack([g.mid.values,g.p.values])
    if Z.std(axis=0).min()<1e-3: continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); r=VECM(Z,k_ar_diff=0,coint_rank=1,deterministic='n').fit()
            try: jt=coint_johansen(Z,-1,0); jo.append(bool(jt.lr1[0]>jt.cvt[0,1]))
            except Exception: pass
        a=r.alpha.flatten(); b=r.beta.flatten(); Om=r.sigma_u; out=is_from(a,Om)
        if out is None: continue
        isk.append(dict(ticker=tk,city=g.city.iloc[0],event=f'{g.city.iloc[0]}|{g.target.iloc[0]}',is_k_lo=out[0],is_k_hi=out[1],alpha_k=a[0],alpha_p=a[1],beta_ratio=-b[1]/b[0] if b[0]!=0 else np.nan,n=len(g)))
    except Exception: nfail+=1
IS=pd.DataFrame(isk)
if len(IS):
    IS['is_k_mid']=(IS.is_k_lo+IS.is_k_hi)/2; rng=np.random.default_rng(0); ev=IS.event.unique(); boots=[]
    for _ in range(1000):
        pick=rng.choice(ev,len(ev)); boots.append(IS.set_index('event').loc[pick].is_k_mid.median())
    res['vecm_information_share']=dict(method='statsmodels VECM(k_ar_diff=0, rank=1, no deterministic) per identical contract with >=12 contiguous hourly obs; Hasbrouck bounds over both Cholesky orderings, midpoint; median across contracts; 95% CI from a bootstrap over events. Series are short (median 16 obs) so per-contract estimates are noisy; the pooled estimator below is validated by simulation.',
        contracts=int(len(IS)),events=int(IS.event.nunique()),fit_failures=nfail,share_johansen_reject_no_coint_5pct=round(float(np.mean(jo)),3) if jo else None,
        kalshi_is_median=round(float(IS.is_k_mid.median()),3),kalshi_is_iqr=[round(float(IS.is_k_mid.quantile(.25)),3),round(float(IS.is_k_mid.quantile(.75)),3)],kalshi_is_median_ci95_events=[round(float(np.quantile(boots,.025)),3),round(float(np.quantile(boots,.975)),3)],
        by_city={c:dict(n=int(len(g)),is_k_median=round(float(g.is_k_mid.median()),3)) for c,g in IS.groupby('city')},share_contracts_kalshi_is_above_half=round(float((IS.is_k_mid>0.5).mean()),3),median_beta_ratio=round(float(IS.beta_ratio.median()),3))
# (b) pooled estimator + simulation validation
def pooled_is(df):
    cl={'groups':pd.factorize(df.ticker)[0]}
    mK=sm.OLS(df.dK1,sm.add_constant(df[['gap','dK','dP']])).fit(); mP=sm.OLS(df.dP1,sm.add_constant(df[['gap','dP','dK']])).fit()
    a=np.array([mK.params.gap,mP.params.gap]); Om=np.cov(np.vstack([mK.resid,mP.resid])); out=is_from(a,Om); return out
def simulate(is_k_true=0.8, n_series=1000, T=16, sd_eff=0.03, seed=0):
    """Two venues observe one efficient probability m_t (random walk). Venue innovations: Kalshi carries share s of efficient innovations
    immediately, Poly carries 1-s; each venue also has transient noise that mean-reverts to m. Returns pooled-estimator IS for Kalshi."""
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n_series):
        m=0.5+np.cumsum(rng.normal(0,sd_eff,T)); eK=rng.normal(0,sd_eff*np.sqrt(is_k_true),T); eP=rng.normal(0,sd_eff*np.sqrt(1-is_k_true),T)
        # partial-adjustment form: venue price = efficient price + venue-specific transient error with different speeds
        uK=np.zeros(T); uP=np.zeros(T)
        for t in range(1,T): uK[t]=0.3*uK[t-1]+eK[t]-np.sqrt(is_k_true)*(m[t]-m[t-1])*0; uP[t]=0.9*uP[t-1]+eP[t]
        pk=m+uK; pp=m+uP
        d=pd.DataFrame({'ticker':i,'mid':pk,'p':pp}); rows.append(d)
    S=pd.concat(rows); g=S.groupby('ticker'); S['dK1']=g.mid.shift(-1)-S.mid; S['dP1']=g.p.shift(-1)-S.p; S['dK']=g.mid.diff(); S['dP']=g.p.diff(); S['gap']=S.mid-S.p
    return pooled_is(S.dropna())
gg=J.groupby('ticker'); J['dK1']=gg.mid.shift(-1)-J.mid; J['dP1']=gg.p.shift(-1)-J.p; J['dK']=gg.mid.diff(); J['dP']=gg.p.diff(); J['gap']=J.mid-J.p
Ecm=J[(gg.ts.shift(-1)-J.ts==pd.Timedelta(hours=1))&(J.ts-gg.ts.shift(1)==pd.Timedelta(hours=1))].dropna(subset=['dK','dP','dK1','dP1'])
pi=pooled_is(Ecm); rng=np.random.default_rng(1); evs=Ecm.assign(event=Ecm.city+'|'+Ecm.target.astype(str)); ev_ids=evs.event.unique(); pb=[]
for _ in range(300):
    pick=rng.choice(ev_ids,len(ev_ids)); sub=evs.set_index('event').loc[pick].reset_index(); o=pooled_is(sub); pb.append((o[0]+o[1])/2 if o else np.nan)
res['pooled_information_share']=dict(method='pooled error-correction regressions with contract series stacked (2-equation ECM on hourly changes), Hasbrouck bounds from the residual covariance; 95% CI from 300 event bootstraps',
    kalshi_is_bounds=[round(pi[0],3),round(pi[1],3)] if pi else None,kalshi_is_mid_ci95_events=[round(float(np.nanquantile(pb,.025)),3),round(float(np.nanquantile(pb,.975)),3)],
    by_period={lab:[round(x,3) for x in (pooled_is(Ecm[sel]) or (np.nan,np.nan))] for lab,sel in [('DEV',Ecm.tgt<HOLD),('HOLDOUT',Ecm.tgt>=HOLD)]},
    by_city={c:[round(x,3) for x in (pooled_is(Ecm[Ecm.city==c]) or (np.nan,np.nan))] for c in SAME},
    simulation_check={f'true_kalshi_share_{s_}':[round(x,3) for x in simulate(s_)] for s_ in [0.2,0.5,0.8]},
    caveat='The pairs settle differently 16% of the time and prices are bounded probabilities on short panels; treat shares as descriptive, not as a decomposition of one common efficient price.')
# pooled error-correction (as before) for comparison, with event-clustered t
def ecm(s):
    cl={'groups':pd.factorize(s.ticker)[0]}; mP=sm.OLS(s.dP1,sm.add_constant(s[['gap','dP','dK']])).fit(cov_type='cluster',cov_kwds=cl); mK=sm.OLS(s.dK1,sm.add_constant(s[['gap','dK','dP']])).fit(cov_type='cluster',cov_kwds=cl)
    return dict(n=int(len(s)),poly_toward_kalshi_per_h=round(float(mP.params.gap),4),t=round(float(mP.tvalues.gap),2),kalshi_toward_poly_per_h=round(float(-mK.params.gap),4),t_k=round(float(-mK.tvalues.gap),2))
res['pooled_error_correction']={'ALL':ecm(Ecm),'DEV':ecm(Ecm[Ecm.tgt<HOLD]),'HOLDOUT':ecm(Ecm[Ecm.tgt>=HOLD])}|{c:ecm(Ecm[Ecm.city==c]) for c in SAME}
# model-free identical-contract trade, and the same trade scored on Kalshi's settlement (basis check)
def mf(Jx,thr=0.08,hs=HS,truth='y_p'):
    t=Jx.sort_values('ts'); by=t[t.mid-t.p>thr].drop_duplicates('ticker'); bn=t[t.p-t.mid>thr].drop_duplicates('ticker')
    a=pd.concat([by.assign(pnl=by[truth]-(by.p+hs),cost=by.p+hs),bn.assign(pnl=(1-bn[truth])-(1-(bn.p-hs)),cost=1-(bn.p-hs))]); a['event']=a.city+'|'+a.target.astype(str); return a
for lab,sel in [('DEV',J.tgt<HOLD),('HOLDOUT',J.tgt>=HOLD),('ALL',J.tgt.notna())]:
    d=sorted(J[sel].tgt.unique()); res[f'model_free_identical|{lab}']=E.summarize_trades(mf(J[sel]),d,label=lab)
    res[f'model_free_identical|{lab}|scored_on_kalshi_settlement']=E.summarize_trades(mf(J[sel],truth='y_k'),d,label=lab)['pnl_ct']
# alternative strip mapping (audit 4.4): piecewise-linear CDF from Kalshi bracket mids (no normal fit inside the strip), normal tails outside
Kf=X[['city','target','ts','lo','hi','pn','mu_k','sig_k','dm']].dropna(subset=['mu_k']).sort_values(['city','target','ts','lo'])
def pl_cdf_factory(g):
    edges=[e for e in g.hi.values if np.isfinite(e)]; cum=np.cumsum(g.pn.values); F=dict(zip(g.hi.values,cum)); mu,sg=g.mu_k.iloc[0],g.sig_k.iloc[0]
    e=np.array(edges); Fe=np.array([F[x] for x in edges]); Fe=np.clip(Fe/max(Fe.max(),1e-9)*Fe.max(),0,1)
    def Fx(x):
        x=np.asarray(x,dtype=float); out=np.empty_like(x)
        lo_m=x<=e[0]; hi_m=x>=e[-1]; mid=~lo_m&~hi_m
        out[mid]=np.interp(x[mid],e,Fe)
        out[lo_m]=Fe[0]*norm.cdf((x[lo_m]-mu)/sg)/max(norm.cdf((e[0]-mu)/sg),1e-9)
        out[hi_m]=1-(1-Fe[-1])*(1-norm.cdf((x[hi_m]-mu)/sg))/max(1-norm.cdf((e[-1]-mu)/sg),1e-9)
        return np.clip(out,0,1)
    return Fx
alt=[]
for (c_,t_,ts_),g in Kf.groupby(['city','target','ts']):
    if len(g)<4 or g.pn.sum()<0.8: continue
    Fx=pl_cdf_factory(g); dm=g.dm.iloc[0]; sub=Y[(Y.city==c_)&(Y.target==t_)&(Y.ts==ts_)]
    if len(sub)==0: continue
    alt.append(pd.Series(Fx(sub.hi.values-dm)-Fx(sub.lo.values-dm),index=sub.index))
if alt:
    ALT=pd.concat(alt); Ya=Y.loc[ALT.index].copy(); Ya['pxk_pl']=np.clip(ALT.values,0.001,0.999); Ya['lx_pl']=E.lg(Ya.pxk_pl)
    res['alt_strip_mapping_piecewise_linear']={'information_ALL':info_logit(Ya,['lp','lx_pl']),'information_ALL_normal_same_rows':info_logit(Ya,['lp','lx']),
        'trade_ALL':E.summarize_trades(signals(wf(Ya,['lp','lx_pl'])),all_days['ALL'],label='pl'),'trade_HOLDOUT':E.summarize_trades(signals(per(wf(Ya,['lp','lx_pl']),'HOLDOUT')),all_days['HOLDOUT'],label='pl'),
        'trade_identical_cities_only_ALL':E.summarize_trades(signals(wf(Ya[Ya.city.isin(SAME)],['lp','lx_pl'])),all_days['ALL'],label='pl same')}
res['env']=E.env_info()
A[['slug','city','target','ts','h','lo','hi','side','p','pn','pxk','ph','cost','pnl','edge','train_cutoff','dev_or_holdout']].to_parquet(OUT/'xvenue_trades.parquet')
json.dump(res,open(OUT/'xvenue.json','w'),indent=1,default=str)
print('XVENUE primary ALL:',res['trading']['lp+lx|ALL']); print('HOLDOUT:',res['trading']['lp+lx|HOLDOUT']); print('frozen:',res['trading']['lp+lx|HOLDOUT|frozen_coefficients_fit_through_2026-03']); print('VECM:',res.get('vecm_information_share'))
