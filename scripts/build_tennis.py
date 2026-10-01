import pandas as pd, numpy as np, glob, re
D='data/tennis/'
def norm(df, src):
    df=df.copy()
    df.columns=[c.strip() for c in df.columns]
    df['Date']=pd.to_datetime(df['Date'], errors='coerce', dayfirst=('maxfar' in src))
    df['src']=src
    return df
frames=[]
a=pd.read_csv(D+'mlt_atp.csv', encoding='latin1', low_memory=False).drop(columns=['Unnamed: 0'])
a=norm(a,'mlt'); print('mlt', a.Date.min(), a.Date.max(), len(a))
frames.append(a)
for f in sorted(glob.glob(D+'edt_*.xlsx')):
    x=pd.read_excel(f); x=norm(x,'edt'); print(f, x.Date.min(), x.Date.max(), len(x)); frames.append(x)
for f in sorted(glob.glob(D+'gm_*.xlsx')):
    x=pd.read_excel(f); x=norm(x,'gm'); print(f, x.Date.min(), x.Date.max(), len(x)); frames.append(x)
df=pd.concat(frames, ignore_index=True, sort=False)
df['key']=df.Date.dt.strftime('%Y-%m-%d')+'|'+df.Winner.astype(str).str.strip()+'|'+df.Loser.astype(str).str.strip()
# prefer sources in order mlt > edt > gm for overlap
pri={'mlt':0,'edt':1,'gm':2}
df['pri']=df.src.map(pri)
df=df.sort_values(['key','pri']).drop_duplicates('key', keep='first')
df=df.sort_values('Date').reset_index(drop=True)
for c in ['WRank','LRank','WPts','LPts','B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL','Wsets','Lsets','Best of']:
    if c in df: df[c]=pd.to_numeric(df[c], errors='coerce')
print(df.groupby(df.Date.dt.year).agg(n=('key','size'), ps=('PSW', lambda s: s.notna().mean()), b365=('B365W', lambda s: s.notna().mean()), avg=('AvgW', lambda s: s.notna().mean())).round(2).to_string())
df.to_pickle(D+"atp_odds.pkl")
print('saved', df.shape)
