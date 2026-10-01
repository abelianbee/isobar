"""Shared library for the forecasting-edge project. Paths are relative to the project root (env EDGE_ROOT overrides).
No exec-based composition; every function is importable and testable."""
from __future__ import annotations
import os, re, json, hashlib, datetime as dt, warnings
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(os.environ.get('EDGE_ROOT', Path(__file__).resolve().parent.parent))
DATA=ROOT/'data'; RESULTS=ROOT/'results'

# ---------------- core conversions
def fee(p):
    """Kalshi taker fee per contract: ceil(0.07*P*(1-P)*100)/100 (fee schedule Jul 2026)."""
    p=np.asarray(p,dtype=float); return np.ceil(100*0.07*p*(1-p))/100
def lg(p):
    p=np.clip(np.asarray(p,dtype=float),0.005,0.995); return np.log(p/(1-p))
def f_to_c(f): return (np.asarray(f,dtype=float)-32)*5/9
_TD=re.compile(r'-(\d{2})([A-Z]{3})(\d{2})(?:-|$)')
def ticker_date(ticker:str)->dt.date:
    """Target date from a Kalshi ticker like KXHIGHNY-26JUL29-B87.5 (never from close_time: it ignores DST)."""
    m=_TD.search(ticker)
    if not m: raise ValueError(f'no date code in {ticker}')
    return pd.Timestamp(f'20{m.group(1)}-{m.group(2)}-{m.group(3)}').date()
def bracket_bounds(strike_type:str, floor_strike, cap_strike):
    """Half-open temperature interval (lo,hi) in whole-degree space with .5 edges.
    between 77-78 -> (76.5,78.5); less than 76 (cap 76) -> (-inf,75.5); greater than 83 (floor 83) -> (83.5,inf)."""
    if strike_type=='between': return float(floor_strike)-0.5, float(cap_strike)+0.5
    if strike_type=='less': return -np.inf, float(cap_strike)-0.5
    if strike_type=='greater': return float(floor_strike)+0.5, np.inf
    raise ValueError(strike_type)
def poly_title_bounds(title:str):
    t=title.replace('°F','').replace('°C','').replace('–','-').replace('—','-').strip()
    if 'or below' in t: return -np.inf, float(t.split()[0])+0.5
    if 'or higher' in t or 'or above' in t: return float(t.split()[0])-0.5, np.inf
    if '-' in t and not t.startswith('-'): a,b=t.split('-'); return float(a)-0.5, float(b)+0.5
    return float(t)-0.5, float(t)+0.5

# ---------------- statistics
def clustered_t(x, groups):
    """Mean, SE and t of x with one-way cluster-robust SE (Liang-Zeger) by groups."""
    x=np.asarray(x,dtype=float); g=pd.factorize(pd.Series(groups))[0]; m=x.mean(); r=x-m
    s=pd.Series(r).groupby(g).sum().values; se=np.sqrt((s**2).sum())/len(x)
    G=len(s); se*=np.sqrt(G/(G-1)) if G>1 else 1
    return float(m), float(se), float(m/se) if se>0 else np.nan
def block_bootstrap_mean(daily:pd.Series, block=7, B=2000, seed=0):
    """Moving-block bootstrap of the mean of a daily series (index = consecutive calendar days). Returns (lo95, hi95)."""
    rng=np.random.default_rng(seed); v=daily.values; n=len(v)
    if n<block*3: return (np.nan,np.nan)
    nb=int(np.ceil(n/block)); starts=rng.integers(0,n-block+1,size=(B,nb))
    idx=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(B,-1)[:,:n]
    means=v[idx].mean(axis=1); return (float(np.quantile(means,0.025)), float(np.quantile(means,0.975)))
def capital_sharpe(trades:pd.DataFrame, all_days, capital=None, q=1.0):
    """Return-on-capital Sharpe: daily P&L / fixed capital, over ALL calendar days in all_days (zeros when inactive).
    capital defaults to the MAXIMUM daily gross cost (collateral) over the window, so every day's positions fit inside it. With constant
    per-signal sizing the Sharpe equals the daily-P&L Sharpe over all calendar days; the return on capital is what changes."""
    all_days=pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(list(all_days))).unique()))
    d=pd.to_datetime(trades['target']); pnl=trades.groupby(d).pnl.sum().reindex(all_days,fill_value=0.0)
    cost=trades.groupby(d).cost.sum().reindex(all_days,fill_value=0.0)
    if capital is None: capital=float(np.quantile(cost[cost>0],q)) if (cost>0).any() else np.nan
    r=pnl/capital
    return dict(capital=capital, ann_return=float(r.mean()*365), ann_vol=float(r.std()*np.sqrt(365)), sharpe=float(r.mean()/r.std()*np.sqrt(365)) if r.std()>0 else np.nan,
                max_drawdown=float((pnl.cumsum()-pnl.cumsum().cummax()).min()), days=len(all_days), active_days=int((cost>0).sum()),
                sharpe_1lot_pnl=float(pnl.mean()/pnl.std()*np.sqrt(365)) if pnl.std()>0 else np.nan, boot95=block_bootstrap_mean(pnl))
def summarize_trades(a:pd.DataFrame, all_days, event_col='event', label=''):
    """Standard trade-table summary with clustered inference. a needs: pnl, cost, target, event_col."""
    if len(a)==0: return dict(label=label,N=0)
    m,se_e,t_e=clustered_t(a.pnl,a[event_col]); _,se_d,t_d=clustered_t(a.pnl,a.target)
    cs=capital_sharpe(a,all_days)
    return dict(label=label,N=int(len(a)),events=int(a[event_col].nunique()),days_active=int(a.target.nunique()),pnl_ct=round(m,4),
                se_event=round(se_e,4),t_event=round(t_e,2),se_day=round(se_d,4),t_day=round(t_d,2),t_iid=round(float(a.pnl.mean()/a.pnl.std()*np.sqrt(len(a))),2),
                win=round(float((a.pnl>0).mean()),3),roi=round(float(a.pnl.sum()/a.cost.sum()),4),
                ci95_pnl_ct_event=[round(m-1.96*se_e,4),round(m+1.96*se_e,4)],
                capital=round(cs['capital'],2),sharpe_capital=round(cs['sharpe'],2),ann_return_on_capital=round(cs['ann_return'],3),
                sharpe_1lot_pnl=round(cs['sharpe_1lot_pnl'],2),max_dd_1lot_usd=round(cs['max_drawdown'],2),boot95_daily_pnl=[round(x,4) for x in cs['boot95']])

# ---------------- reproducibility
def sha256(path, chunk=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()
def file_manifest(paths):
    out=[]
    for p in paths:
        p=Path(p)
        if p.is_dir():
            files=sorted(p.glob('*')); out.append(dict(path=str(p.relative_to(ROOT)),kind='dir',n_files=len(files),bytes=int(sum(f.stat().st_size for f in files)),
                sha256_of_names_and_sizes=hashlib.sha256('\n'.join(f'{f.name}:{f.stat().st_size}' for f in files).encode()).hexdigest()))
        else:
            out.append(dict(path=str(p.relative_to(ROOT)),kind='file',bytes=p.stat().st_size,sha256=sha256(p),mtime=dt.datetime.fromtimestamp(p.stat().st_mtime,dt.timezone.utc).isoformat()))
    return out
def env_info():
    import sys, platform, pandas, numpy, statsmodels, scipy
    try: import sklearn; skv=sklearn.__version__
    except Exception: skv=None
    return dict(python=sys.version.split()[0],platform=platform.platform(),pandas=pandas.__version__,numpy=numpy.__version__,statsmodels=statsmodels.__version__,scipy=scipy.__version__,sklearn=skv,built_utc=dt.datetime.now(dt.timezone.utc).isoformat())

# ---------------- weather blend (importable version of weather_edge2's preamble)
CLI_MAP={'NYC':'KNYC','MDW':'KMDW','MIA':'KMIA','AUS':'KAUS','LAX':'KLAX','DEN':'KDEN','PHL':'KPHL','ATL':'KATL','BOS':'KBOS','DFW':'KDFW','DCA':'KDCA','HOU':'KHOU','LAS':'KLAS','MSP':'KMSP','OKC':'KOKC','PHX':'KPHX','SEA':'KSEA','SFO':'KSFO',
         'LGA':'KLGA','ORD':'KORD','DAL':'KDAL'}
def _design(d, cols):
    X=d[cols].copy(); X=X.apply(lambda r: r.fillna(r.mean()), axis=1); X['s1']=d.s1; X['c1']=d.c1; X.insert(0,'const',1.0); return X
def build_blend(stations, years=range(2023,2027), mode='ols', lead=1, kind='high', min_train=250, ridge_alphas=(0.1,1,3,10,30,100)):
    """Per-station forecast blend trained on years < y only. mode: 'ols' (original) or 'ridge' (alpha by 5-fold CV inside the training window).
    Returns (MU dataframe [station,target,mu,res_key,ncols,cond], RES dict res_key->training residual array, diagnostics list)."""
    import statsmodels.api as sm
    fd=pd.read_parquet(DATA/'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
    cli=pd.read_parquet(DATA/'weather/cli_daily.parquet')
    col='tmax_fc' if kind=='high' else 'tmin_fc'; truth='high' if kind=='high' else 'low'
    fd=fd[fd.station.isin(stations)].copy(); fd['cli']=fd.station.map(CLI_MAP)
    fd=fd.merge(cli[['station','date',truth]].rename(columns={'station':'cli',truth:'high'}), on=['cli','date']); fd['year']=pd.to_datetime(fd.date).dt.year
    P=fd[fd.lead==lead].pivot_table(index=['station','date','year','high'],columns='model',values=col).reset_index()
    MODELS=[m for m in P.columns if m not in ('station','date','year','high')]
    P['doy']=pd.to_datetime(P.date).dt.dayofyear; P['s1']=np.sin(2*np.pi*P.doy/365); P['c1']=np.cos(2*np.pi*P.doy/365)
    rows=[]; RES={}; diag=[]
    for st,g in P.groupby('station'):
        for y in years:
            tr0=g[g.year<y]; te=g[g.year==y]
            tr=tr0[tr0[MODELS].notna().sum(axis=1)>=4]
            cols=[m for m in MODELS if tr[m].notna().mean()>0.6] if len(tr)>=min_train else []
            if len(tr)<min_train or len(cols)<3:
                cols=[m for m in ['gfs_seamless','ncep_hrrr_conus'] if m in tr0 and tr0[m].notna().mean()>0.6]; tr=tr0.dropna(subset=cols)
            else: tr=tr.dropna(subset=cols, thresh=max(3,int(0.7*len(cols))))
            if len(tr)<min_train or len(te)==0 or not cols: continue
            X=_design(tr,cols); Xt=_design(te,cols); yv=tr.high.values
            cond=float(np.linalg.cond(X.values))
            if mode=='ols':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')   # rank deficiency is reported via cond below, not silently
                    m=sm.OLS(yv,X).fit()
                mu=m.predict(Xt); fit=m.predict(X); alpha=None
            else:
                from sklearn.linear_model import RidgeCV
                from sklearn.model_selection import KFold
                Xs=X.drop(columns='const').values.astype(float); mean=np.nanmean(Xs,0); sd=np.nanstd(Xs,0)+1e-9
                Zs=np.nan_to_num((Xs-mean)/sd); Zt=np.nan_to_num((Xt.drop(columns='const').values.astype(float)-mean)/sd)   # missing -> training mean
                rc=RidgeCV(alphas=ridge_alphas, cv=KFold(5,shuffle=True,random_state=0)).fit(Zs, yv)
                mu=rc.predict(Zt); fit=rc.predict(Zs); alpha=float(rc.alpha_)
            res=yv-fit; key=f'{st}{y}{mode}'
            rows.append(pd.DataFrame({'station':st,'target':te.date.values,'mu':np.asarray(mu),'res_key':key,'ncols':len(cols)}))
            RES[key]=res; diag.append(dict(station=st,year=y,mode=mode,n_train=int(len(tr)),n_models=len(cols),cond=round(cond,1),alpha=alpha,resid_sd=round(float(res.std()),3),
                                           oos_mae=round(float(np.abs(te.high.values-np.asarray(mu)).mean()),3)))
    return pd.concat(rows), RES, diag
def bracket_prob(mu, res, lo, hi):
    v=mu+res; return float(((v>lo)&(v<hi)).mean())
