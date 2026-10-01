import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/'
df=pd.read_pickle(R+'data/tennis/atp_odds_elo.pkl')
d=df[(df.Date.dt.year>=2006)&(df.nW>=10)&(df.nL>=10)].copy()
d['oW']=d.PSW.where(d.PSW.notna(), d.AvgW.where(d.AvgW.notna(), d.B365W)); d['oL']=d.PSL.where(d.PSL.notna(), d.AvgL.where(d.AvgL.notna(), d.B365L))
d=d[(d.oW>1)&(d.oL>1)].copy(); iw=1/d.oW; il=1/d.oL; d['pW']=iw/(iw+il); d['year']=d.Date.dt.year
# surface season start: first clay tournament date each year (after Feb, i.e., European clay ~April) and first grass (June)
d['Surface']=d.Surface.fillna('Hard'); d['doy']=d.Date.dt.dayofyear
first_clay=d[(d.Surface=='Clay')&(d.doy>80)].groupby('year').Date.min().rename('clay0')
first_grass=d[(d.Surface=='Grass')].groupby('year').Date.min().rename('grass0')
d=d.merge(first_clay,on='year').merge(first_grass,on='year')
d['days_clay']=(d.Date-d.clay0).dt.days; d['days_grass']=(d.Date-d.grass0).dt.days
d['trans']=np.where((d.Surface=='Clay')&(d.days_clay>=0)&(d.days_clay<14),'clay_early',np.where((d.Surface=='Grass')&(d.days_grass>=0)&(d.days_grass<14),'grass_early',
          np.where((d.Surface=='Clay')&(d.days_clay>=14)&(d.days_clay<70),'clay_late',np.where((d.Surface=='Grass')&(d.days_grass>=14),'grass_late','other'))))
rng=np.random.default_rng(0); flip=rng.random(len(d))<0.5
d['won']=np.where(flip,0,1); d['p']=np.where(flip,1-d.pW,d.pW); d['lp']=np.log(d.p/(1-d.p))
d['elo_d']=np.where(flip,-1,1)*(d.eloW-d.eloL)/400*np.log(10); d['selo_d']=np.where(flip,-1,1)*(d.seloW-d.seloL)/400*np.log(10)
d['surf_gap']=d.selo_d-d.elo_d   # surface-specific minus overall: surface specialist signal
def logit(y,X,name):
    m=sm.Logit(y, sm.add_constant(X)).fit(disp=0); print(f'--- {name} (N={len(y)}) ---'); print(m.summary().tables[1])
for t,g in d.groupby('trans'):
    logit(g.won, g[['lp','surf_gap']], f'H10 {t}: won ~ lp + (surfaceElo - overallElo)')
# trade: in early clay/grass, bet on player with surf_gap > thr at Pinnacle closing
ps=d[d.PSW.notna()&d.PSL.notna()].copy(); ps['odds']=np.where(flip[d.PSW.notna()&d.PSL.notna()],ps.PSL,ps.PSW)
for t in ['clay_early','grass_early','clay_late','other']:
    for thr in [0.3,0.5]:
        x=ps[(ps.trans==t)]; 
        a=x[x.surf_gap>thr]; b=x[x.surf_gap<-thr]
        pnl=np.concatenate([np.where(a.won==1,a.odds-1,-1.0), np.where(b.won==0, np.where(flip[(d.PSW.notna()&d.PSL.notna())][x.surf_gap<-thr], ps.loc[b.index,'PSW'], ps.loc[b.index,'PSL'])-1, -1.0)])
        print(f'{t} |gap|>{thr}: N={len(pnl)} ROI={pnl.mean()*100:+.2f}% se={pnl.std()/np.sqrt(len(pnl))*100:.2f}%')
