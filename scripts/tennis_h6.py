import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_pickle('data/tennis/atp_odds.pkl')
df=df[df.Comment.astype(str).str.contains('Completed', na=False) | df.Comment.isna()].copy()
# closing odds preference: Pinnacle, then Avg, then B365
def pick(w,l):
    return w,l
df['oW']=df.PSW.where(df.PSW.notna(), df.AvgW.where(df.AvgW.notna(), df.B365W))
df['oL']=df.PSL.where(df.PSL.notna(), df.AvgL.where(df.AvgL.notna(), df.B365L))
df['ps_avail']=df.PSW.notna()&df.PSL.notna()
df=df[(df.oW>1)&(df.oL>1)].copy()
iw=1/df.oW; il=1/df.oL; df['pW']=iw/(iw+il); df['pL']=1-df.pW
df['mid']=np.arange(len(df))
# long format
W=pd.DataFrame({'mid':df.mid,'Date':df.Date,'player':df.Winner.str.strip(),'opp':df.Loser.str.strip(),'won':1,'p':df.pW,'odds':df.oW,'B365':df.B365W,'Max':df.MaxW,'PS':df.PSW,'ps_avail':df.ps_avail,'Tournament':df.Tournament,'Round':df.Round,'Surface':df.Surface,'Series':df.Series,'sets':df.Wsets+df.Lsets, 'rank':df.WRank,'orank':df.LRank})
L=pd.DataFrame({'mid':df.mid,'Date':df.Date,'player':df.Loser.str.strip(),'opp':df.Winner.str.strip(),'won':0,'p':df.pL,'odds':df.oL,'B365':df.B365L,'Max':df.MaxL,'PS':df.PSL,'ps_avail':df.ps_avail,'Tournament':df.Tournament,'Round':df.Round,'Surface':df.Surface,'Series':df.Series,'sets':df.Wsets+df.Lsets,'rank':df.LRank,'orank':df.WRank})
pm=pd.concat([W,L]).sort_values(['player','Date','mid']).reset_index(drop=True)
pm['surp']=pm.won-pm.p
g=pm.groupby('player')
for k in [1,2,3]:
    pm[f'surp_l{k}']=g.surp.shift(k)
pm['won_l1']=g.won.shift(1); pm['p_l1']=g.p.shift(1)
pm['days_since']=(pm.Date-g.Date.shift(1)).dt.days
pm['prev_sets']=g.sets.shift(1)
pm['surp_l123']=pm[['surp_l1','surp_l2','surp_l3']].sum(axis=1, min_count=3)
# attach opponent's lags
opp=pm[['mid','player','surp_l1','surp_l123','days_since','prev_sets','won_l1','p_l1']].rename(columns={'player':'opp','surp_l1':'o_surp_l1','surp_l123':'o_surp_l123','days_since':'o_days','prev_sets':'o_prev_sets','won_l1':'o_won_l1','p_l1':'o_p_l1'})
pm=pm.merge(opp, on=['mid','opp'], how='left')
pm['lp']=np.log(pm.p/(1-pm.p))
pm.to_pickle('data/tennis/atp_playermatch.pkl')
# one random row per match for clean SEs
rng=np.random.default_rng(0)
pm['r']=rng.random(len(pm)); one=pm.sort_values('r').drop_duplicates('mid')
one=one.dropna(subset=['surp_l1','o_surp_l1']).copy()
one['d_surp1']=one.surp_l1-one.o_surp_l1
one['d_surp123']=one.surp_l123-one.o_surp_l123
one['year']=one.Date.dt.year
print('N matches', len(one), one.year.min(), one.year.max())
def logit(y,X,name):
    m=sm.Logit(y, sm.add_constant(X)).fit(disp=0); print(f'--- {name} ---'); print(m.summary().tables[1])
logit(one.won, one[['lp']], 'calibration: won ~ logit(p)')
logit(one.won, one[['lp','d_surp1']], 'H6: + last-match surprise diff')
o3=one.dropna(subset=['d_surp123']); logit(o3.won, o3[['lp','d_surp123']], 'H6: + last-3 surprise diff')
# same tournament only (prev match within 7 days)
st=one[(one.days_since<=7)&(one.o_days<=7)]; logit(st.won, st[['lp','d_surp1']], 'H6: prev match within 7 days both')
# split by era
for lo,hi in [(2004,2012),(2013,2019),(2020,2026)]:
    s=one[(one.year>=lo)&(one.year<=hi)]; logit(s.won, s[['lp','d_surp1']], f'H6 {lo}-{hi} N={len(s)}')
# Pinnacle only
sp=one[one.ps_avail]; logit(sp.won, sp[['lp','d_surp1']], f'H6 Pinnacle-closing only N={len(sp)}')
# bucket residuals
one['res']=one.won-one.p
one['b']=pd.cut(one.d_surp1,[-2,-0.6,-0.3,-0.1,0.1,0.3,0.6,2])
print(one.groupby('b',observed=True).agg(n=('res','size'),mean_res=('res','mean'),se=('res',lambda s: s.std()/np.sqrt(len(s)))).round(4))
