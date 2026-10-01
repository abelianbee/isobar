import pandas as pd, numpy as np, json, statsmodels.api as sm
d=pd.DataFrame(json.load(open('data/nba/nba_archive_10Y.json')))
d['date']=pd.to_datetime(d['date'].astype(int).astype(str), format='%Y%m%d')
for c in ['home_final','away_final','home_close_spread','home_open_spread','close_over_under','home_close_ml','away_close_ml']:
    d[c]=pd.to_numeric(d[c], errors='coerce')
d=d.dropna(subset=['home_final','away_final','home_close_spread']).sort_values('date').reset_index(drop=True)
print(len(d), d.date.min(), d.date.max()); print(d.season.value_counts().sort_index().to_dict())
d['margin']=d.home_final-d.away_final           # home margin
d['home_cover_margin']=d.margin + d.home_close_spread  # >0 home covers (spread negative for fav)
# sanity: spread sign convention
print('mean home spread', d.home_close_spread.mean().round(2), 'mean home margin', d.margin.mean().round(2), 'mean cover margin', d.home_cover_margin.mean().round(3))
print('home cover rate (ex push)', (d.home_cover_margin[d.home_cover_margin!=0]>0).mean().round(4))
# team-game long format
h=d[['date','season','home_team','home_cover_margin','home_close_spread']].rename(columns={'home_team':'team','home_cover_margin':'cm','home_close_spread':'spread'}); h['home']=1
a=d[['date','season','away_team','home_cover_margin','home_close_spread']].rename(columns={'away_team':'team','home_cover_margin':'cm','home_close_spread':'spread'}); a['cm']=-a.cm; a['spread']=-a.spread; a['home']=0
tg=pd.concat([h,a]).sort_values(['team','date']).reset_index(drop=True)
tg['prev_cm']=tg.groupby(['team','season']).cm.shift(1)
tg['prev_date']=tg.groupby(['team','season']).date.shift(1)
tg['rest']=(tg.date-tg.prev_date).dt.days
tg=tg.dropna(subset=['prev_cm'])
tg['cover']=(tg.cm>0).astype(float); tg.loc[tg.cm==0,'cover']=np.nan
x=tg.dropna(subset=['cover'])
print('N team-games', len(x))
X=sm.add_constant(x[['prev_cm']]); m=sm.OLS(x.cover, X).fit(cov_type='HC1'); print(m.summary().tables[1])
X=sm.add_constant(x[['prev_cm','home','spread']]); m=sm.OLS(x.cm, X).fit(cov_type='HC1'); print('cover margin on prev_cm'); print(m.summary().tables[1])
# bucket
x['b']=pd.cut(x.prev_cm,[-60,-15,-7,0,7,15,60])
print(x.groupby('b', observed=True).agg(n=('cover','size'), cover=('cover','mean'), cm=('cm','mean')).round(3))
# by season stability: fade teams that covered by >10 last game -> bet against
x['fade']=np.where(x.prev_cm>10,1,np.where(x.prev_cm<-10,-1,0))
sel=x[x.fade!=0].copy(); sel['win']=np.where(sel.fade==1, 1-sel.cover, sel.cover)
print(sel.groupby('season').agg(n=('win','size'), winrate=('win','mean')).round(3)); print('overall', sel.win.mean().round(4), len(sel), 'breakeven at -110 = 0.5238')
