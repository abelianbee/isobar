import pandas as pd, numpy as np, glob
from scipy.stats import norm
R='/Users/abebraham/Desktop/resume/edge/data/'
SER={'KXHIGHNY':('NYC','KNYC','America/New_York','high'),'KXHIGHCHI':('MDW','KMDW','America/Chicago','high'),'KXHIGHMIA':('MIA','KMIA','America/New_York','high'),'KXHIGHAUS':('AUS','KAUS','America/Chicago','high'),'KXHIGHLAX':('LAX','KLAX','America/Los_Angeles','high'),'KXHIGHDEN':('DEN','KDEN','America/Denver','high'),'KXHIGHPHIL':('PHL','KPHL','America/New_York','high'),
 'KXLOWTNYC':('NYC','KNYC','America/New_York','low'),'KXLOWTCHI':('MDW','KMDW','America/Chicago','low'),'KXLOWTMIA':('MIA','KMIA','America/New_York','low'),'KXLOWTAUS':('AUS','KAUS','America/Chicago','low'),'KXLOWTLAX':('LAX','KLAX','America/Los_Angeles','low'),'KXLOWTDEN':('DEN','KDEN','America/Denver','low'),'KXLOWTPHIL':('PHL','KPHL','America/New_York','low'),
 'KXHIGHTATL':('ATL','KATL','America/New_York','high'),'KXHIGHTBOS':('BOS','KBOS','America/New_York','high'),'KXHIGHTDAL':('DFW','KDFW','America/Chicago','high'),'KXHIGHTDC':('DCA','KDCA','America/New_York','high'),'KXHIGHTHOU':('HOU','KHOU','America/Chicago','high'),'KXHIGHTLV':('LAS','KLAS','America/Los_Angeles','high'),'KXHIGHTMIN':('MSP','KMSP','America/Chicago','high'),'KXHIGHTOKC':('OKC','KOKC','America/Chicago','high'),'KXHIGHTPHX':('PHX','KPHX','America/Phoenix','high'),'KXHIGHTSEA':('SEA','KSEA','America/Los_Angeles','high'),'KXHIGHTSFO':('SFO','KSFO','America/Los_Angeles','high')}
c=pd.read_parquet(R+'kalshi/candles_long.parquet')
mk=pd.concat([pd.read_pickle(f) for f in glob.glob(R+'kalshi/markets_*.pkl')]).drop_duplicates('ticker'); mk=mk[mk.series.isin(SER)]
  # close is 23:59 local -> date of target day (ET approx; fix per tz below)
ev=mk.groupby('event_ticker').agg(series=('series','first'), close_time=('close_time','first'), open_time=('open_time','first'), truth=('expiration_value','first')).reset_index()
ev['truth']=pd.to_numeric(ev.truth, errors='coerce')
ev['tz']=ev.series.map(lambda s: SER[s][2]); ev['station']=ev.series.map(lambda s: SER[s][0]); ev['cli']=ev.series.map(lambda s: SER[s][1]); ev['kind']=ev.series.map(lambda s: SER[s][3])
ev['target']=[pd.Timestamp(ct).tz_convert(tz).date() - pd.Timedelta(minutes=1)*0 for ct,tz in zip(ev.close_time, ev.tz)]
import re as _re
def _tdate(t):
    m=_re.search(r'-(\d{2})([A-Z]{3})(\d{2})', t); return pd.Timestamp(f'20{m.group(1)}-{m.group(2)}-{m.group(3)}').date()
ev['target']=ev.event_ticker.map(_tdate)
cli=pd.read_parquet(R+'weather/cli_daily.parquet')
ev=ev.merge(cli[['station','date','high','low']].rename(columns={'station':'cli','date':'target','high':'cli_hi','low':'cli_lo'}), on=['cli','target'], how='left'); ev['cli_high']=np.where(ev.kind=='low', ev.cli_lo, ev.cli_hi)
print('events', len(ev), 'truth==cli_high agreement:', (ev.truth==ev.cli_high).mean().round(3), 'missing cli', ev.cli_high.isna().mean().round(3))
print(ev[(ev.truth!=ev.cli_high)&ev.cli_high.notna()].head(8).to_string())
# forecasts
fd=pd.read_parquet(R+'weather/fcst_daily.parquet'); fd['date']=pd.to_datetime(fd.date).dt.date
piv=fd.pivot_table(index=['station','date'], columns=['model','lead'], values='tmax_fc')
piv.columns=[f'{m}_L{l}' for m,l in piv.columns]; piv=piv.reset_index().rename(columns={'date':'target'})
ev=ev.merge(piv, on=['station','target'], how='left')
ev.to_parquet(R+'weather/events.parquet')
# bracket table per market
b=mk[['ticker','event_ticker','floor_strike','cap_strike','strike_type','result']+[c for c in ['subtitle','yes_sub_title'] if c in mk]].copy()
b['lo']=np.where(b.strike_type=='less', -np.inf, np.where(b.strike_type=='between', b.floor_strike+0.5, b.floor_strike+0.5))
b['hi']=np.where(b.strike_type=='greater', np.inf, np.where(b.strike_type=='between', b.cap_strike+0.5, b.cap_strike+0.5))
# between 77-78 means high in {77,78}: floor 77 cap 78 -> lo 76.5 hi 78.5 ; less than 76 -> hi=75.5 ; greater than 83 -> lo 83.5
b['lo']=np.where(b.strike_type=='between', b.floor_strike-0.5, b.lo); b['hi']=np.where(b.strike_type=='less', b.cap_strike-0.5, b.hi)
# --- fallback when the API omits strike fields (this happens for the SETTLED-YES market of many events, and for whole events before 2025):
#     parse the bracket from the market subtitle ("70° to 71°", "72° or above", "63° or below"), else from the ticker (-B70.5 = between 70-71;
#     -T83 = threshold 83: 'greater' if at/above the highest between-cap in the event, 'less' if at/below the lowest between-floor).
import re as _re2
b=b.reset_index(drop=True); miss=(b.strike_type.isna()|b.lo.isna()|b.hi.isna())
sub=(b['subtitle'] if 'subtitle' in b else pd.Series('',index=b.index)).fillna('').astype(str)
sub=sub.where(sub!='', (b['yes_sub_title'] if 'yes_sub_title' in b else pd.Series('',index=b.index)).fillna('').astype(str))
bet=b[b.strike_type=='between'].groupby('event_ticker').agg(mx=('cap_strike','max'),mn=('floor_strike','min')); b=b.join(bet,on='event_ticker'); fixed={'subtitle':0,'ticker':0}
for i in b.index[miss]:
    t=sub[i].replace('°','').replace('or higher','or above').replace('or lower','or below')
    t=_re2.sub(r'(-?\d+)\.0\b',r'\1',t)
    m1=_re2.match(r'\s*(-?\d+)\s*to\s*(-?\d+)',t); m2=_re2.match(r'\s*(-?\d+)\s*or above',t) or _re2.match(r'\s*>\s*(-?\d+)',t); m3=_re2.match(r'\s*(-?\d+)\s*or below',t) or _re2.match(r'\s*<\s*(-?\d+)',t)
    if _re2.match(r'\s*>\s*-?\d+',t): m2=_re2.match(r'\s*>\s*(-?\d+)',t)   # '>71' means 72 or higher -> lo=71.5
    if _re2.match(r'\s*<\s*-?\d+',t): m3=_re2.match(r'\s*<\s*(-?\d+)',t)   # '<68' means 67 or lower  -> hi=67.5
    if m1: b.at[i,'lo']=float(m1.group(1))-0.5; b.at[i,'hi']=float(m1.group(2))+0.5; b.at[i,'strike_type']='between'; b.at[i,'floor_strike']=float(m1.group(1)); b.at[i,'cap_strike']=float(m1.group(2)); fixed['subtitle']+=1; continue
    if m2: b.at[i,'lo']=(float(m2.group(1))+0.5 if t.strip().startswith('>') else float(m2.group(1))-0.5); b.at[i,'hi']=np.inf; b.at[i,'strike_type']='greater'; b.at[i,'floor_strike']=float(m2.group(1)); fixed['subtitle']+=1; continue
    if m3: b.at[i,'lo']=-np.inf; b.at[i,'hi']=(float(m3.group(1))-0.5 if t.strip().startswith('<') else float(m3.group(1))+0.5); b.at[i,'strike_type']='less'; b.at[i,'cap_strike']=float(m3.group(1)); fixed['subtitle']+=1; continue
    mb=_re2.search(r'-B(-?\d+(?:\.\d+)?)$',b.at[i,'ticker']); mt=_re2.search(r'-T(-?\d+(?:\.\d+)?)$',b.at[i,'ticker'])
    if mb: c_=float(mb.group(1)); b.at[i,'lo']=c_-1; b.at[i,'hi']=c_+1; b.at[i,'strike_type']='between'; fixed['ticker']+=1; continue
    if mt:
        v=float(mt.group(1)); mx,mn=b.at[i,'mx'],b.at[i,'mn']
        if pd.notna(mx) and v>=mx: b.at[i,'lo']=v+0.5; b.at[i,'hi']=np.inf; b.at[i,'strike_type']='greater'; fixed['ticker']+=1
        elif pd.notna(mn) and v<=mn: b.at[i,'lo']=-np.inf; b.at[i,'hi']=v-0.5; b.at[i,'strike_type']='less'; fixed['ticker']+=1
print('brackets reconstructed:', fixed, '| still missing bounds:', int((b.lo.isna()|b.hi.isna()).sum()))
b=b.drop(columns=['mx','mn']); print('tail brackets reconstructed from ticker:', fixed, '| still missing bounds:', int((b.lo.isna()|b.hi.isna()).sum()))
b.to_parquet(R+'kalshi/brackets.parquet')
# sanity: result vs truth
x=b.merge(ev[['event_ticker','truth']], on='event_ticker'); x['pred_yes']=(x.truth>x.lo)&(x.truth<x.hi)
print('bracket result consistency:', ((x.result=='yes')==x.pred_yes).mean().round(4), 'N', len(x))
_ex=[]
_nw=x.groupby('event_ticker').pred_yes.sum(); _ex+= [(e,'winners!=1',int(n)) for e,n in _nw[_nw!=1].items()]
_mm=x[(x.result=='yes')!=x.pred_yes].groupby('event_ticker').size(); _ex+= [(e,'kalshi_result_vs_truth_mismatch',int(n)) for e,n in _mm.items()]
_nt=ev[ev.truth.isna()].event_ticker; _ex+= [(e,'no_settlement_truth',0) for e in _nt]
_nb=b[b.lo.isna()|b.hi.isna()].event_ticker.unique(); _ex+= [(e,'bracket_bounds_missing',0) for e in _nb]
pd.DataFrame(_ex,columns=['event_ticker','reason','n']).to_parquet(R+'kalshi/exclusions.parquet'); print('exclusion log rows:', len(_ex), pd.DataFrame(_ex,columns=['e','reason','n']).reason.value_counts().to_dict())
print(x[((x.result=='yes')!=x.pred_yes)].head(5).to_string())
