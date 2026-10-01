import pandas as pd, numpy as np, statsmodels.api as sm
pm=pd.read_pickle('data/tennis/atp_playermatch.pkl')
pm['year']=pm.Date.dt.year
pm['d_surp1']=pm.surp_l1-pm.o_surp_l1
pm['d_p_l1']=pm.p_l1-pm.o_p_l1   # my prev-match closing prob minus opp's
pm['lrank_d']=np.log(pm['rank'].clip(1,2000))-np.log(pm.orank.clip(1,2000))
pm['res']=pm.won-pm.p
rng=np.random.default_rng(1); pm['r']=rng.random(len(pm)); one=pm.sort_values('r').drop_duplicates('mid')
wt=one[(one.days_since<=7)&(one.o_days<=7)&(one.won_l1==1)&(one.o_won_l1==1)].dropna(subset=['d_p_l1','lrank_d']).copy()
print('within-tournament matches, both won prev:', len(wt))
def logit(y,X,name):
    m=sm.Logit(y, sm.add_constant(X)).fit(disp=0); print(f'--- {name} (N={len(y)}) ---'); print(m.summary().tables[1])
logit(wt.won, wt[['lp','d_p_l1']], 'won ~ lp + prev-round prob diff')
logit(wt.won, wt[['lp','d_p_l1','lrank_d']], '+ log rank diff')
for lo,hi in [(2004,2009),(2010,2014),(2015,2019),(2020,2026)]:
    s=wt[(wt.year>=lo)&(wt.year<=hi)]; logit(s.won, s[['lp','d_p_l1']], f'{lo}-{hi}')
sp=wt[wt.ps_avail]; logit(sp.won, sp[['lp','d_p_l1']], 'Pinnacle closing only')
# residual buckets
wt['b']=pd.cut(wt.d_p_l1,[-1,-0.5,-0.3,-0.15,-0.05,0.05,0.15,0.3,0.5,1])
print(wt.groupby('b',observed=True).agg(n=('res','size'),mean_res=('res','mean'),se=('res',lambda s: s.std()/np.sqrt(len(s)))).round(4))
# by round
print(wt.groupby('Round').apply(lambda s: pd.Series({'n':len(s),'corr':np.corrcoef(s.d_p_l1,s.res)[0,1]})).round(3))
# --- betting sim: bet on player whose previous-round win was MORE expected (d_p_l1>thr), at closing odds
pm_wt=pm[(pm.days_since<=7)&(pm.o_days<=7)&(pm.won_l1==1)&(pm.o_won_l1==1)].dropna(subset=['d_p_l1']).copy()
for thr in [0.15,0.25,0.35]:
    b=pm_wt[pm_wt.d_p_l1>thr].copy()
    for oc in ['PS','Max','odds']:
        bb=b.dropna(subset=[oc]); pnl=np.where(bb.won==1, bb[oc]-1, -1.0)
        print(f'thr={thr} odds={oc}: N={len(bb)} ROI={pnl.mean()*100:.2f}% winrate={bb.won.mean():.3f} avg_p={bb.p.mean():.3f}  by-era ROI:', {f'{lo}': round(pnl[(bb.year>=lo)&(bb.year<=hi)].mean()*100,1) for lo,hi in [(2004,2009),(2010,2014),(2015,2019),(2020,2026)]})
