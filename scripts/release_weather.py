"""Release build, strategy 1 (forecast blend vs Kalshi daily-high brackets). Reproducible from data on disk; no exec.
Writes results/release/weather.json, weather_trades.parquet, weather_blend_diag.json and regenerates data/weather/comb_coefs.json with provenance.
Walk-forward: blend weights from years < target year; logistic combination refit monthly on ALL prior months (expanding window) after a 120-event warm-up."""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import edge_lib as E
OUT=E.RESULTS/'release'; OUT.mkdir(exist_ok=True, parents=True)
STATIONS=['NYC','MDW','MIA','AUS','LAX','DEN','PHL','ATL','BOS','DFW','DCA','HOU','LAS','MSP','OKC','PHX','SEA','SFO']
SNAPS={8:-2,10:0,12:2}   # label -> hours relative to D 00Z: 8 = D-1 22Z, 10 = D 00Z, 12 = D 02Z (clock-based, not hours-since-open)
HOURS=[8,10,12]; THRS=[0.03,0.05,0.08,0.12]; PRIMARY_H=8; PRIMARY_THR=0.05; START=pd.Timestamp('2025-01-01')
c0=pd.read_parquet(E.DATA/'weather/candles_ev.parquet')
c0=c0[(c0.kind=='high')&(c0.h<=37)&(c0.ask>0)&(c0.ask<1)&(c0.bid<1)&(c0.bid>=0)].copy()
c0['tgt']=pd.to_datetime(c0.target); c0=c0[c0.tgt>=START]
c0['decision_utc']=c0.time; c0['month']=c0.tgt.dt.to_period('M'); c0['lmid']=E.lg(c0.mid)
# clock-based snapshot label: hours from D 00Z (UTC) to the candle end time
c0['rel_h']=((c0.time-c0.tgt.dt.tz_localize('UTC'))/pd.Timedelta(hours=1)).round().astype(int)
c0['snap']=c0.rel_h.map({v:k for k,v in SNAPS.items()})
recon0={'rows_h8_hours_since_open':int((c0.h==8).sum()),'rows_D-1_22Z_clock':int((c0.snap==8).sum()),'share_h8_rows_at_22Z':round(float((c0[c0.h==8].time.dt.hour==22).mean()),3)}
c0['season']=np.where(c0.tgt.dt.month.isin([11,12,1,2,3]),'winter',np.where(c0.tgt.dt.month.isin([6,7,8,9]),'summer','shoulder'))
recon={'kalshi_bracket_hours_2025plus':int(len(c0)),'events_with_candles':int(c0.event_ticker.nunique()),**recon0}
excl=pd.read_parquet(E.DATA/'kalshi/exclusions.parquet') if (E.DATA/'kalshi/exclusions.parquet').exists() else pd.DataFrame(columns=['event_ticker','reason','n'])
recon['excluded_events_by_reason']=excl.groupby('reason').event_ticker.nunique().to_dict()
def score(mode):
    global RES
    MU,RES,diag=E.build_blend(STATIONS, years=range(2023,2027), mode=mode)
    c=c0.merge(MU[['station','target','mu','res_key']],on=['station','target'])
    pm=np.empty(len(c))
    for (k,mu),idx in c.groupby(['res_key','mu']).indices.items():
        v=mu+RES[k]; sub=c.iloc[idx]; pm[idx]=[((v>l)&(v<h)).mean() for l,h in zip(sub.lo,sub.hi)]
    c['pm']=pm; c['lpm']=E.lg(c.pm); return c,diag
def walk_forward(s, cols=('lmid','lpm'), warm=120, weighted=False):
    cols=list(cols); out=[]
    for m in sorted(s.month.unique()):
        tr=s[s.month<m]; te=s[s.month==m]
        if tr.event_ticker.nunique()<warm: continue
        assert tr.tgt.max()<te.tgt.min(), 'leakage: training row after test start'
        X=sm.add_constant(tr[cols]); 
        if weighted:
            w=1/tr.groupby('ticker').ticker.transform('size'); mod=sm.GLM(tr.y,X,family=sm.families.Binomial(),var_weights=w).fit()
        else: mod=sm.Logit(tr.y,X).fit(disp=0)
        out.append(te.assign(ph=mod.predict(sm.add_constant(te[cols])),b_lpm=mod.params['lpm'],train_cutoff=str(tr.tgt.max().date())))
    return pd.concat(out)
def trades(t, thr, slip=0.0, fill='close'):
    t=t.copy()
    if fill=='next':   # execute at next hour's close quote
        nxt=cH[(cH.rel_h==t.rel_h.iloc[0]+1)][['ticker','ask','bid','vol']].rename(columns={'ask':'a1','bid':'b1','vol':'v1'}); t=t.merge(nxt,on='ticker'); pa,pb=t.a1,t.b1
    elif fill=='worst':
        nxt=cH[(cH.rel_h==t.rel_h.iloc[0]+1)][['ticker','ask','bid','vol']].rename(columns={'ask':'a1','bid':'b1','vol':'v1'}); t=t.merge(nxt,on='ticker'); t=t[t.v1>0]; pa,pb=np.minimum(0.99,t.a1+0.01),np.maximum(0.01,t.b1-0.01)
    else: pa,pb=t.ask,t.bid
    by=t[t.ph-t.ask>thr].copy(); bn=t[t.bid-t.ph>thr].copy()
    by['side']='Y'; by['cost']=pa[by.index]+E.fee(pa[by.index])+slip; by['pnl']=by.y-by.cost; by['edge']=by.ph-by.ask
    bn['side']='N'; bn['cost']=(1-pb[bn.index])+E.fee(pb[bn.index])+slip; bn['pnl']=(1-bn.y)-bn.cost; bn['edge']=bn.bid-bn.ph
    a=pd.concat([by,bn]); a['event']=a.event_ticker; return a
res={'spec':dict(primary_hour=PRIMARY_H,primary_threshold=PRIMARY_THR,fill='candle close ask/bid at the decision time; decision time is clock-based (D-1 22:00Z, D 00:00Z, D 02:00Z), not hours-since-open',fee='ceil(0.07*P*(1-P)*100)/100',blend='per-station linear blend of lead-1 Open-Meteo previous_day1 daily maxima, weights from years < target year',
                 combination='logit(y) ~ logit(mid) + logit(p_blend), expanding-window monthly refit, 120-event warm-up',start=str(START.date()),stations=STATIONS,
                 timing_caveat='previous_day1 archive fields are assumed public by D-1 22Z (h=8); this is verified only for Apr-Sep 2026 via single-run archives, NOT for winter. Winter results are conditional on this assumption.'),
     'modes':{}}
for mode in ['ols','ridge']:
    c,diag=score(mode); cH=c
    json.dump(diag,open(OUT/f'weather_blend_diag_{mode}.json','w'),indent=1)
    recon[f'scored_bracket_hours_{mode}']=int(len(c)); recon[f'scored_events_{mode}']=int(c.event_ticker.nunique())
    M={'blend_diag_summary':dict(max_cond=max(d['cond'] for d in diag), median_cond=float(np.median([d['cond'] for d in diag])), rank_deficient_fits=int(sum(d['cond']>1e8 for d in diag)), n_fits=len(diag),
                                 oos_mae_by_year={str(y):round(float(np.mean([d['oos_mae'] for d in diag if d['year']==y])),3) for y in range(2023,2027) if any(d['year']==y for d in diag)})}
    for h in HOURS:
        s=c[c.snap==h]; t=walk_forward(s); all_days=sorted(t.tgt.unique())
        H={'oos_events':int(t.event_ticker.nunique()),'days':len(all_days),'brier_market':round(float(((t.mid-t.y)**2).mean()),4),'brier_blend':round(float(((t.pm-t.y)**2).mean()),4),'brier_combined':round(float(((t.ph-t.y)**2).mean()),4),'mean_lpm_coef':round(float(t.b_lpm.mean()),3),'decision_utc_hour_share':t.decision_utc.dt.hour.value_counts(normalize=True).round(3).to_dict(),'decision_on_D_minus_1_share':round(float((t.decision_utc.dt.date<t.tgt.dt.date).mean()),4)}
        grid={}
        for thr in THRS:
            for slip in [0.0,0.01,0.02]:
                a=trades(t,thr,slip); grid[f'thr{thr}_slip{slip}']=E.summarize_trades(a,all_days,label=f'h{h} thr{thr} slip{slip}')
        H['spec_curve']=grid
        a=trades(t,PRIMARY_THR); H['primary']=E.summarize_trades(a,all_days,label='primary')
        for ssn in ['winter','summer','shoulder']:
            sub=a[a.season==ssn]; H[f'season_{ssn}_trades_only_active_days']=E.summarize_trades(sub,[d for d in all_days if pd.Timestamp(d).month in {'winter':[11,12,1,2,3],'summer':[6,7,8,9],'shoulder':[4,5,10]}[ssn]],label=ssn)
            H[f'season_{ssn}_as_strategy_all_days']=E.summarize_trades(sub,all_days,label=ssn+' over all calendar days')
        H['by_station']={k:dict(n=int(n),pnl_ct=round(v,4),t_event=round(E.clustered_t(g.pnl,g.event)[2],2)) for k,g in a.groupby('station') for v,n in [(g.pnl.mean(),len(g))]}
        H['by_month']={str(k):dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for k,g in a.groupby('month')}
        H['by_edge_bin']={str(k):dict(n=int(len(g)),pnl_ct=round(float(g.pnl.mean()),4)) for k,g in a.groupby(pd.cut(a.edge,[0.05,0.07,0.10,0.15,0.25,1]),observed=True)}
        H['stress_next_hour_fill']=E.summarize_trades(trades(t,PRIMARY_THR,fill='next'),all_days,label='next-hour close')
        H['stress_worst_fill']=E.summarize_trades(trades(t,PRIMARY_THR,fill='worst'),all_days,label='next-hour +1c, volume>0')
        # leave-one-station-out and city contribution
        H['loco']={st:E.summarize_trades(a[a.station!=st],all_days,label=f'without {st}')['t_event'] for st in a.station.unique()}
        # cross-city dependence: mean pairwise correlation of daily station P&L
        piv=a.groupby(['target','station']).pnl.sum().unstack().reindex(all_days).fillna(0.0); cc=piv.corr().values; H['mean_pairwise_daily_corr_between_stations']=round(float(np.nanmean(cc[np.triu_indices_from(cc,1)])),3)
        H['events_with_multiple_positions_share']=round(float((a.groupby('event').size()>1).mean()),3)
        # capacity: (a) ORIGINAL future-volume sizing (labelled infeasible), (b) trailing-volume sizing (known at decision time)
        rh=SNAPS[h]; vol_next=c[(c.rel_h>rh)&(c.rel_h<=rh+6)].groupby('ticker').vol.sum().rename('vol_next6'); vol_prev=c[(c.rel_h<rh)&(c.rel_h>=rh-6)].groupby('ticker').vol.sum().rename('vol_prev6')
        aa=a.join(vol_next,on='ticker').join(vol_prev,on='ticker').fillna({'vol_next6':0,'vol_prev6':0})
        cap={}
        for part in [0.10,0.20]:
            for lab,col in [('future_volume_INFEASIBLE','vol_next6'),('trailing_volume','vol_prev6')]:
                n=np.minimum(100,np.floor(part*aa[col])); usd=(n*aa.pnl); d=usd.groupby(aa.target).sum().reindex(all_days,fill_value=0.0)
                cap[f'{lab}_part{part}']=dict(contracts_per_signal=round(float(n.mean()),1),usd_per_year=round(float(d.mean()*365),0),usd_sharpe=round(float(d.mean()/d.std()*np.sqrt(365)),2) if d.std()>0 else None,share_signals_zero_size=round(float((n==0).mean()),3))
        H['capacity']=cap
        if h==PRIMARY_H and mode=='ols':
            a.assign(mode=mode)[['ticker','event','station','target','decision_utc','h','side','ask','bid','mid','pm','ph','cost','pnl','edge','train_cutoff']].to_parquet(OUT/'weather_trades.parquet')
            # equal-weight-per-bracket logistic (audit 4.10 analogue)
            tw=walk_forward(s,weighted=True); H['primary_equal_event_weight']=E.summarize_trades(trades(tw,PRIMARY_THR),all_days,label='GLM equal weight per bracket')
            # placebo: blend shuffled across events within station
            rng=np.random.default_rng(0); sp=s.copy()
            mu_map=c.drop_duplicates('event_ticker').set_index('event_ticker')[['mu','res_key']]
            parts=[]
            for st,g in sp.groupby('station'):
                ev=g.event_ticker.unique(); perm=dict(zip(ev,rng.permutation(ev))); g=g.copy(); g['mu']=g.event_ticker.map(lambda e: mu_map.mu[perm[e]]); parts.append(g)
            sp=pd.concat(parts); pmv=np.empty(len(sp))
            for (k,mu),idx in sp.groupby(['res_key','mu']).indices.items():
                v=mu+RES[k]; sub=sp.iloc[idx]; pmv[idx]=[((v>l)&(v<hh)).mean() for l,hh in zip(sub.lo,sub.hi)]
            sp['pm']=pmv; sp['lpm']=E.lg(sp.pm); tp=walk_forward(sp); H['placebo_shuffled_forecast']=dict(mean_lpm_coef=round(float(tp.b_lpm.mean()),3),**E.summarize_trades(trades(tp,PRIMARY_THR),all_days,label='placebo'))
        M[f'h{h}']=H
    res['modes'][mode]=M
    if mode=='ols':
        # regenerate comb_coefs.json with provenance (fit on all scored rows, as live_signal.py expects the same keys)
        cc={}
        for h in HOURS:
            s=c[c.snap==h]
            for lab,sub in [(str(h),s),(f'{h}_winter',s[s.season=='winter'])]:
                m=sm.Logit(sub.y,sm.add_constant(sub[['lmid','lpm']])).fit(disp=0); cc[lab]=dict(const=float(m.params['const']),lmid=float(m.params['lmid']),lpm=float(m.params['lpm']),n=int(len(sub)))
        cc['_provenance']=dict(training_cutoff=str(c.tgt.max().date()),training_start=str(c.tgt.min().date()),rows=int(len(c)),events=int(c.event_ticker.nunique()),estimation='expanding window over all scored rows; monthly refit in backtest',
                              inputs_sha256={p.name:E.sha256(p) for p in [E.DATA/'weather/candles_ev.parquet',E.DATA/'weather/fcst_daily.parquet',E.DATA/'weather/cli_daily.parquet']},built=E.env_info()['built_utc'],
                              note='Polymarket rows are never used in this fit (Kalshi candles only).')
        json.dump(cc,open(E.DATA/'weather/comb_coefs.json','w'),indent=1)
res['reconciliation']=recon; res['env']=E.env_info()
json.dump(res,open(OUT/'weather.json','w'),indent=1,default=str)
p=res['modes']['ols'][f'h{PRIMARY_H}']['primary']; r=res['modes']['ridge'][f'h{PRIMARY_H}']['primary']
print('WEATHER primary (ols):',p); print('WEATHER primary (ridge):',r); print('winter as strategy:',res['modes']['ols'][f'h{PRIMARY_H}']['season_winter_as_strategy_all_days'])
