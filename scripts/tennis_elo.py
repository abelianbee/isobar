import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/'
df=pd.read_pickle(R+'data/tennis/atp_odds.pkl').sort_values(['Date']).reset_index(drop=True)
df['Winner']=df.Winner.str.strip(); df['Loser']=df.Loser.str.strip()
df['Surface']=df.Surface.fillna('Hard')
# Elo with 538-style K: K = 250/((n+5)^0.4)
elo={}; cnt={}; selo={}; scnt={}
pre_w=[];pre_l=[];spre_w=[];spre_l=[]; nw=[]; nl=[]
for w,l,s in zip(df.Winner, df.Loser, df.Surface):
    ew=elo.get(w,1500); el=elo.get(l,1500); pre_w.append(ew); pre_l.append(el); nw.append(cnt.get(w,0)); nl.append(cnt.get(l,0))
    kw=250/((cnt.get(w,0)+5)**0.4); kl=250/((cnt.get(l,0)+5)**0.4)
    pw=1/(1+10**((el-ew)/400))
    elo[w]=ew+kw*(1-pw); elo[l]=el-kl*(1-pw); cnt[w]=cnt.get(w,0)+1; cnt[l]=cnt.get(l,0)+1
    key=(s,); sew=selo.get((w,s),1500); sel=selo.get((l,s),1500); spre_w.append(sew); spre_l.append(sel)
    kw=250/((scnt.get((w,s),0)+5)**0.4); kl=250/((scnt.get((l,s),0)+5)**0.4)
    pw=1/(1+10**((sel-sew)/400))
    selo[(w,s)]=sew+kw*(1-pw); selo[(l,s)]=sel-kl*(1-pw); scnt[(w,s)]=scnt.get((w,s),0)+1; scnt[(l,s)]=scnt.get((l,s),0)+1
df['eloW']=pre_w; df['eloL']=pre_l; df['seloW']=spre_w; df['seloL']=spre_l; df['nW']=nw; df['nL']=nl
df.to_pickle(R+'data/tennis/atp_odds_elo.pkl')
# analysis on matches with closing odds, after 2005 burn-in
d=df[(df.Date.dt.year>=2006)].copy()
d['oW']=d.PSW.where(d.PSW.notna(), d.AvgW.where(d.AvgW.notna(), d.B365W)); d['oL']=d.PSL.where(d.PSL.notna(), d.AvgL.where(d.AvgL.notna(), d.B365L))
d=d[(d.oW>1)&(d.oL>1)&(d.nW>=10)&(d.nL>=10)].copy()
iw=1/d.oW; il=1/d.oL; d['pW']=iw/(iw+il)
rng=np.random.default_rng(0); flip=rng.random(len(d))<0.5
d['won']=np.where(flip,0,1); d['p']=np.where(flip,1-d.pW,d.pW)
d['elo_d']=np.where(flip, d.eloL-d.eloW, d.eloW-d.eloL)/400*np.log(10)   # in logit units
d['selo_d']=np.where(flip, d.seloL-d.seloW, d.seloW-d.seloL)/400*np.log(10)
d['blend_d']=0.5*d.elo_d+0.5*d.selo_d
d['rank_d']=np.where(flip, np.log(d.WRank.clip(1,2000))-np.log(d.LRank.clip(1,2000)), np.log(d.LRank.clip(1,2000))-np.log(d.WRank.clip(1,2000)))  # + means I am worse-ranked (higher number)
d['lp']=np.log(d.p/(1-d.p)); d['year']=d.Date.dt.year
d=d.dropna(subset=['rank_d'])
def logit(y,X,name):
    m=sm.Logit(y, sm.add_constant(X)).fit(disp=0); print(f'--- {name} (N={len(y)}) ---'); print(m.summary().tables[1]); return m
print('elo vs market: corr', np.corrcoef(d.elo_d, d.lp)[0,1].round(3))
logit(d.won, d[['elo_d']], 'won ~ elo only'); logit(d.won, d[['selo_d']], 'won ~ surface elo only'); logit(d.won, d[['lp']], 'won ~ market')
logit(d.won, d[['lp','elo_d']], 'H9: market + elo')
logit(d.won, d[['lp','selo_d']], 'market + surface elo')
logit(d.won, d[['lp','blend_d']], 'market + blend')
logit(d.won, d[['lp','elo_d','rank_d']], 'H9: market + elo + log-rank diff')
for lo,hi in [(2006,2010),(2011,2015),(2016,2019),(2020,2026)]:
    s=d[(d.year>=lo)&(d.year<=hi)]; logit(s.won, s[['lp','blend_d','rank_d']], f'{lo}-{hi}')
sp=d[d.PSW.notna()&d.PSL.notna()]; logit(sp.won, sp[['lp','blend_d','rank_d']], 'Pinnacle-only')
