import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/'
df=pd.read_pickle(R+'data/tennis/atp_odds_tml.pkl')
df=df[(df.Date.dt.year>=2004)].copy()
df['oW']=df.PSW.where(df.PSW.notna(), df.AvgW.where(df.AvgW.notna(), df.B365W)); df['oL']=df.PSL.where(df.PSL.notna(), df.AvgL.where(df.AvgL.notna(), df.B365L))
df=df[(df.oW>1)&(df.oL>1)].copy(); iw=1/df.oW; il=1/df.oL; df['pW']=iw/(iw+il)
df['mid']=np.arange(len(df)); df['sets']=df.Wsets+df.Lsets; df['Winner']=df.Winner.str.strip(); df['Loser']=df.Loser.str.strip()
W=pd.DataFrame({'mid':df.mid,'Date':df.Date,'player':df.Winner,'opp':df.Loser,'won':1,'p':df.pW,'odds':df.oW,'minutes':df.minutes,'sets':df.sets,'bo':df['Best of'],'Round':df.Round,'Series':df.Series,'year':df.Date.dt.year,'ps':df.PSW.notna()})
L=pd.DataFrame({'mid':df.mid,'Date':df.Date,'player':df.Loser,'opp':df.Winner,'won':0,'p':1-df.pW,'odds':df.oL,'minutes':df.minutes,'sets':df.sets,'bo':df['Best of'],'Round':df.Round,'Series':df.Series,'year':df.Date.dt.year,'ps':df.PSW.notna()})
pm=pd.concat([W,L]).sort_values(['player','Date','mid']).reset_index(drop=True)
g=pm.groupby('player')
pm['prev_min']=g.minutes.shift(1); pm['prev_sets']=g.sets.shift(1); pm['days']=(pm.Date-g.Date.shift(1)).dt.days; pm['prev_won']=g.won.shift(1)
# cumulative minutes in last 7 days (excluding today)
pm['cum7']=0.0
for name,grp in g:
    pass
pm=pm.sort_values(['player','Date']).reset_index(drop=True)
# rolling minutes over previous 7 days per player
def roll(grp):
    s=grp.set_index('Date').minutes.fillna(grp.minutes.median()); r=s.rolling('7D', closed='left').sum(); return pd.Series(r.values, index=grp.index)
pm['min7']=pm.groupby('player', group_keys=False).apply(roll)
opp=pm[['mid','player','prev_min','prev_sets','days','min7','prev_won']].rename(columns={'player':'opp','prev_min':'o_prev_min','prev_sets':'o_prev_sets','days':'o_days','min7':'o_min7','prev_won':'o_prev_won'})
pm=pm.merge(opp,on=['mid','opp'],how='left')
pm['lp']=np.log(pm.p/(1-pm.p))
rng=np.random.default_rng(2); pm['r']=rng.random(len(pm)); one=pm.sort_values('r').drop_duplicates('mid')
wt=one[(one.days<=7)&(one.o_days<=7)&(one.prev_won==1)&(one.o_prev_won==1)].dropna(subset=['prev_min','o_prev_min','min7','o_min7','prev_sets','o_prev_sets']).copy()
wt['d_min']=(wt.prev_min-wt.o_prev_min)/60; wt['d_min7']=(wt.min7-wt.o_min7)/60; wt['d_days']=wt.days-wt.o_days; wt['d_sets']=wt.prev_sets-wt.o_prev_sets
print('N within-tournament both won prev, minutes known:', len(wt))
def logit(y,X,name):
    m=sm.Logit(y, sm.add_constant(X)).fit(disp=0); print(f'--- {name} (N={len(y)}) ---'); print(m.summary().tables[1])
logit(wt.won, wt[['lp','d_min']], 'H8: prev-match minutes diff (hours)')
logit(wt.won, wt[['lp','d_min7']], 'H8: 7-day cumulative minutes diff (hours)')
logit(wt.won, wt[['lp','d_min','d_days']], 'H8: minutes + rest-days diff')
logit(wt.won, wt[['lp','d_sets']], 'H8: prev sets diff')
gs=wt[wt.bo==5]; logit(gs.won, gs[['lp','d_min','d_days']], 'Grand Slams only (Bo5)')
for lo,hi in [(2004,2011),(2012,2019),(2020,2026)]:
    s=wt[(wt.year>=lo)&(wt.year<=hi)]; logit(s.won, s[['lp','d_min7','d_days']], f'{lo}-{hi}')
wt['res']=wt.won-wt.p; wt['b']=pd.cut(wt.d_min7,[-10,-3,-1.5,-0.5,0.5,1.5,3,10])
print(wt.groupby('b',observed=True).agg(n=('res','size'),res=('res','mean'),se=('res',lambda s:s.std()/np.sqrt(len(s)))).round(4))
