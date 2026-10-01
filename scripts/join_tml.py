import pandas as pd, numpy as np, glob, re, unicodedata
R='/Users/abebraham/Desktop/resume/edge/'
def strip_acc(s): return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c))
tml=pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(R+'data/tennis/tml/*.csv'))], ignore_index=True)
tml['tdate']=pd.to_datetime(tml.tourney_date.astype(str), format='%Y%m%d', errors='coerce')
def key_full(name):
    n=strip_acc(name).replace('-',' ').strip(); parts=n.split()
    if len(parts)<2: return n.lower()
    return (parts[-1]+' '+parts[0][0]).lower()
def key_td(name):
    n=strip_acc(name).replace('-',' ').strip()
    m=re.match(r'^(.*?)\s+([A-Za-z])\.?', n)
    if not m: return n.lower()
    return (m.group(1).split()[-1]+' '+m.group(2)).lower()
tml['wk']=tml.winner_name.map(key_full); tml['lk']=tml.loser_name.map(key_full)
odds=pd.read_pickle(R+'data/tennis/atp_odds.pkl')
odds['wk']=odds.Winner.map(key_td); odds['lk']=odds.Loser.map(key_td)
odds['year']=odds.Date.dt.year; tml['year']=tml.tdate.dt.year
m=odds.reset_index().merge(tml[['year','wk','lk','tdate','tourney_name','minutes','score','round','w_svpt','l_svpt','best_of']], on=['year','wk','lk'], how='left')
m['dd']=(m.Date-m.tdate).dt.days
m=m[((m.dd>=-2)&(m.dd<=21))|m.tdate.isna()]
m=m.sort_values(['index','dd']).drop_duplicates('index')
print('odds rows', len(odds), 'matched', m.minutes.notna().sum(), 'match rate', round(m.minutes.notna().mean(),3))
print(m.groupby('year').minutes.apply(lambda s: s.notna().mean()).round(2).to_dict())
out=odds.join(m.set_index('index')[['minutes','score','w_svpt','l_svpt','tdate']])
out.to_pickle(R+'data/tennis/atp_odds_tml.pkl')
um=m[m.minutes.isna()&(m.year>=2010)&(m.year<=2025)]; print(um[['Date','Tournament','Winner','Loser','wk','lk']].sample(12, random_state=0).to_string())
