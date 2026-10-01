import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/'
df=pd.read_pickle(R+'data/tennis/atp_odds_elo.pkl')
d=df[(df.Date.dt.year>=2006)&df.PSW.notna()&df.PSL.notna()&(df.nW>=10)&(df.nL>=10)].copy()
d=d[(d.PSW>1)&(d.PSL>1)]
iw=1/d.PSW; il=1/d.PSL; d['pW']=iw/(iw+il); d['vig']=iw+il-1
d['eloPW']=1/(1+10**(-(0.5*(d.eloW-d.eloL)+0.5*(d.seloW-d.seloL))/400))
d['year']=d.Date.dt.year
# favourite = market favourite. Is fav the elo underdog?
d['fav_is_W']=d.pW>=0.5
d['pf']=np.where(d.fav_is_W,d.pW,1-d.pW); d['elo_pf']=np.where(d.fav_is_W,d.eloPW,1-d.eloPW)
d['fav_odds']=np.where(d.fav_is_W,d.PSW,d.PSL); d['fav_won']=d.fav_is_W.astype(int)
d['fav_odds_max']=np.where(d.fav_is_W,d.MaxW,d.MaxL); d['fav_odds_b365']=np.where(d.fav_is_W,d.B365W,d.B365L)
d['gap']=d.pf-d.elo_pf     # market likes fav more than Elo does
print('vig mean', d.vig.mean().round(4), 'N', len(d))
# 1) favourite-longshot: ROI betting all favourites at Pinnacle closing by price bucket
d['pb']=pd.cut(d.pf,[0.5,0.6,0.7,0.8,0.9,1.0])
def roi(x, oc='fav_odds'):
    x=x.dropna(subset=[oc]); pnl=np.where(x.fav_won==1, x[oc]-1, -1.0); return len(x), round(pnl.mean()*100,2), round(pnl.std()/np.sqrt(len(x))*100,2)
print('--- all favourites by price bucket (N, ROI%, se%) ---')
for b,x in d.groupby('pb',observed=True): print(b, roi(x))
print('--- by elo-gap bucket (market likes fav more than Elo -> gap>0) ---')
d['gb']=pd.cut(d.gap,[-1,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,1])
for b,x in d.groupby('gb',observed=True): print(b, roi(x), 'winrate', round(x.fav_won.mean(),3), 'avg pf', round(x.pf.mean(),3))
print('--- OOS: strategy bet FAV when gap>thr; fit nothing, just threshold; by era ---')
for thr in [0.05,0.1,0.15,0.2]:
    for lo,hi in [(2006,2011),(2012,2017),(2018,2021),(2022,2026)]:
        x=d[(d.gap>thr)&(d.year>=lo)&(d.year<=hi)]
        n,r,se=roi(x); nm,rm,_=roi(x,'fav_odds_max'); nb,rb,_=roi(x,'fav_odds_b365')
        print(f'thr={thr} {lo}-{hi}: N={n} ROI_pinn={r}% (se {se}) ROI_max={rm}% (N={nm}) ROI_b365={rb}%')
# 2) proper OOS logistic: fit on 2006-2017 won~lp+blend_d, predict 2018+, bet when model prob*odds > 1+edge
rng=np.random.default_rng(0); flip=rng.random(len(d))<0.5
d['won']=np.where(flip,0,1); d['p']=np.where(flip,1-d.pW,d.pW); d['odds']=np.where(flip,d.PSL,d.PSW); d['oddsmax']=np.where(flip,d.MaxL,d.MaxW)
d['lp']=np.log(d.p/(1-d.p)); d['blend_d']=np.where(flip,-1,1)*(0.5*(d.eloW-d.eloL)+0.5*(d.seloW-d.seloL))/400*np.log(10)
tr=d[d.year<=2017]; te=d[d.year>=2018]
m=sm.Logit(tr.won, sm.add_constant(tr[['lp','blend_d']])).fit(disp=0); print(m.params.round(4).to_dict())
te=te.copy(); te['ph']=m.predict(sm.add_constant(te[['lp','blend_d']]))
# both sides: evaluate bet on side with edge
rows=[]
for side in [1,-1]:
    q=te.copy()
    if side==-1: q['ph']=1-q.ph; q['odds']=np.where(flip[d.year>=2018],d.PSW[d.year>=2018],d.PSL[d.year>=2018]); q['won']=1-q.won
    q['ev']=q.ph*q.odds-1; rows.append(q)
q=pd.concat(rows)
for e in [0.0,0.02,0.04,0.06]:
    x=q[q.ev>e]; pnl=np.where(x.won==1,x.odds-1,-1.0)
    print(f'OOS 2018-26 EV>{e}: N={len(x)} ROI={pnl.mean()*100:.2f}% se={pnl.std()/np.sqrt(len(x))*100:.2f}%  per-year:', x.assign(pnl=pnl).groupby('year').pnl.mean().mul(100).round(1).to_dict())
