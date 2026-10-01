import pandas as pd, numpy as np, statsmodels.api as sm
R='/Users/abebraham/Desktop/resume/edge/data/tennis/'
w=pd.read_csv(R+'mlt_wta.csv', encoding='latin1', low_memory=False).drop(columns=['Unnamed: 0'])
w['Date']=pd.to_datetime(w.Date, errors='coerce'); w=w.sort_values('Date').reset_index(drop=True)
for c in ['PSW','PSL','B365W','B365L','AvgW','AvgL','MaxW','MaxL','WRank','LRank']: w[c]=pd.to_numeric(w[c], errors='coerce')
w['Winner']=w.Winner.str.strip(); w['Loser']=w.Loser.str.strip(); w['Surface']=w.Surface.fillna('Hard')
print('WTA', len(w), w.Date.min(), w.Date.max(), 'PS avail', w.PSW.notna().mean().round(2))
# Elo
elo={}; cnt={}; selo={}; scnt={}; pw_=[]; pl_=[]; spw=[]; spl=[]; nw=[]; nl=[]
for a,b,s in zip(w.Winner,w.Loser,w.Surface):
    ea=elo.get(a,1500); eb=elo.get(b,1500); pw_.append(ea); pl_.append(eb); nw.append(cnt.get(a,0)); nl.append(cnt.get(b,0))
    ka=250/((cnt.get(a,0)+5)**0.4); kb=250/((cnt.get(b,0)+5)**0.4); p=1/(1+10**((eb-ea)/400)); elo[a]=ea+ka*(1-p); elo[b]=eb-kb*(1-p); cnt[a]=cnt.get(a,0)+1; cnt[b]=cnt.get(b,0)+1
    sa=selo.get((a,s),1500); sb=selo.get((b,s),1500); spw.append(sa); spl.append(sb)
    ka=250/((scnt.get((a,s),0)+5)**0.4); kb=250/((scnt.get((b,s),0)+5)**0.4); p=1/(1+10**((sb-sa)/400)); selo[(a,s)]=sa+ka*(1-p); selo[(b,s)]=sb-kb*(1-p); scnt[(a,s)]=scnt.get((a,s),0)+1; scnt[(b,s)]=scnt.get((b,s),0)+1
w['eloW']=pw_; w['eloL']=pl_; w['seloW']=spw; w['seloL']=spl; w['nW']=nw; w['nL']=nl
d=w[(w.Date.dt.year>=2009)&(w.nW>=10)&(w.nL>=10)].copy()
d['oW']=d.PSW.where(d.PSW.notna(), d.AvgW.where(d.AvgW.notna(), d.B365W)); d['oL']=d.PSL.where(d.PSL.notna(), d.AvgL.where(d.AvgL.notna(), d.B365L))
d=d[(d.oW>1)&(d.oL>1)].copy(); iw=1/d.oW; il=1/d.oL; d['pW']=iw/(iw+il); d['year']=d.Date.dt.year; d['mid']=np.arange(len(d))
# ---- H6-style: last-match surprise
W=pd.DataFrame({'mid':d.mid,'Date':d.Date,'player':d.Winner,'opp':d.Loser,'won':1,'p':d.pW}); L=pd.DataFrame({'mid':d.mid,'Date':d.Date,'player':d.Loser,'opp':d.Winner,'won':0,'p':1-d.pW})
pm=pd.concat([W,L]).sort_values(['player','Date','mid']).reset_index(drop=True); g=pm.groupby('player')
pm['surp_l1']=(pm.won-pm.p).groupby(pm.player).shift(1); pm['days']=(pm.Date-g.Date.shift(1)).dt.days; pm['won_l1']=g.won.shift(1); pm['p_l1']=g.p.shift(1)
opp=pm[['mid','player','surp_l1','days','won_l1','p_l1']].rename(columns={'player':'opp','surp_l1':'o_surp','days':'o_days','won_l1':'o_won_l1','p_l1':'o_p_l1'}); pm=pm.merge(opp,on=['mid','opp'])
pm['lp']=np.log(pm.p/(1-pm.p)); rng=np.random.default_rng(0); pm['r']=rng.random(len(pm)); one=pm.sort_values('r').drop_duplicates('mid').dropna(subset=['surp_l1','o_surp'])
one['d_surp']=one.surp_l1-one.o_surp; one['year']=one.Date.dt.year
def logit(y,X,name):
    m=sm.Logit(y, sm.add_constant(X)).fit(disp=0); print(f'--- {name} (N={len(y)}) ---'); print(m.summary().tables[1])
logit(one.won, one[['lp','d_surp']], 'WTA H6 all: won ~ lp + last-surprise diff')
wt=one[(one.days<=7)&(one.o_days<=7)&(one.won_l1==1)&(one.o_won_l1==1)].copy(); wt['d_p_l1']=wt.p_l1-wt.o_p_l1
logit(wt.won, wt[['lp','d_p_l1']], 'WTA within-tournament: won ~ lp + prev-round prob diff')
for lo,hi in [(2009,2013),(2014,2019)]:
    s=wt[(wt.year>=lo)&(wt.year<=hi)]; logit(s.won, s[['lp','d_p_l1']], f'  {lo}-{hi}')
# ---- H9: Elo vs market
flip=rng.random(len(d))<0.5
d['won']=np.where(flip,0,1); d['p']=np.where(flip,1-d.pW,d.pW); d['lp']=np.log(d.p/(1-d.p))
d['blend_d']=np.where(flip,-1,1)*(0.5*(d.eloW-d.eloL)+0.5*(d.seloW-d.seloL))/400*np.log(10)
d['rank_d']=np.where(flip,1,-1)*(np.log(d.LRank.clip(1,2000))-np.log(d.WRank.clip(1,2000)))
dd=d.dropna(subset=['rank_d'])
logit(dd.won, dd[['lp']], 'WTA market only'); logit(dd.won, dd[['lp','blend_d']], 'WTA market + Elo'); logit(dd.won, dd[['lp','blend_d','rank_d']], 'WTA market + Elo + rank')
for lo,hi in [(2009,2013),(2014,2019)]:
    s=dd[(dd.year>=lo)&(dd.year<=hi)]; logit(s.won, s[['lp','blend_d']], f'  WTA {lo}-{hi}')
# anti-Elo favourite trade at Pinnacle
ps=d[d.PSW.notna()&d.PSL.notna()].copy(); iw=1/ps.PSW; il=1/ps.PSL; ps['pf_']=iw/(iw+il); ps['favW']=ps.pf_>=0.5
ps['pf']=np.where(ps.favW,ps.pf_,1-ps.pf_); ps['eloP']=1/(1+10**(-(0.5*(ps.eloW-ps.eloL)+0.5*(ps.seloW-ps.seloL))/400)); ps['elo_pf']=np.where(ps.favW,ps.eloP,1-ps.eloP); ps['gap']=ps.pf-ps.elo_pf
ps['fav_odds']=np.where(ps.favW,ps.PSW,ps.PSL); ps['fav_won']=ps.favW.astype(int)
for thr in [0.05,0.1,0.15]:
    for lo,hi in [(2009,2013),(2014,2019)]:
        x=ps[(ps.gap>thr)&(ps.year>=lo)&(ps.year<=hi)]; pnl=np.where(x.fav_won==1,x.fav_odds-1,-1.0); print(f'WTA anti-Elo fav gap>{thr} {lo}-{hi}: N={len(x)} ROI={pnl.mean()*100:+.2f}% se={pnl.std()/np.sqrt(len(x))*100:.2f}')
    for lo,hi in [(2009,2013),(2014,2019)]:
        x=ps[(ps.gap<-thr)&(ps.year>=lo)&(ps.year<=hi)]; pnl=np.where(x.fav_won==0,np.where(x.favW,x.PSL,x.PSW)-1,-1.0); print(f'   pro-Elo underdog gap<-{thr} {lo}-{hi}: N={len(x)} ROI={pnl.mean()*100:+.2f}%')
