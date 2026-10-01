"""Regression tests for the bugs that mattered: dates, brackets, units, fees, timestamps, leakage, frozen trades."""
import sys, json, numpy as np, pandas as pd, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'scripts')); import edge_lib as E

@pytest.mark.parametrize('tk,exp',[('KXHIGHNY-26JUL29-B87.5','2026-07-29'),('KXHIGHNY-26JUL29-T88','2026-07-29'),('HIGHAUS-24JUN11-B93.5','2024-06-11'),
                                   ('KXHIGHTATL-26FEB20-T73','2026-02-20'),('KXHIGHCHI-25NOV02-B41.5','2025-11-02'),('KXHIGHMIA-26MAR08-B82.5','2026-03-08')])
def test_ticker_date(tk,exp): assert str(E.ticker_date(tk))==exp
def test_ticker_date_ignores_close_time_dst():
    # close_time 2026-07-30T04:59Z is 00:59 EDT on Jul 30 but the market is for Jul 29 -> only the ticker is safe
    assert str(E.ticker_date('KXHIGHNY-26JUL29-B87.5'))=='2026-07-29'
def test_bracket_bounds():
    assert E.bracket_bounds('between',77,78)==(76.5,78.5)
    assert E.bracket_bounds('less',None,76)==(-np.inf,75.5)
    assert E.bracket_bounds('greater',83,None)==(83.5,np.inf)
    lo,hi=E.bracket_bounds('between',77,78); assert (77>lo and 77<hi) and (78>lo and 78<hi) and not (76>lo and 76<hi) and not (79>lo and 79<hi)
def test_poly_titles():
    assert E.poly_title_bounds('53°F or below')==(-np.inf,53.5); assert E.poly_title_bounds('56-57°F')==(55.5,57.5); assert E.poly_title_bounds('66°F or higher')==(65.5,np.inf)
def test_f_to_c(): assert abs(E.f_to_c(212)-100)<1e-9 and abs(E.f_to_c(32))<1e-9
def test_fee_rounding():
    assert E.fee(0.5)==0.02 and E.fee(0.1)==0.01 and E.fee(0.99)==0.01 and E.fee(0.35)==0.02   # 0.07*0.35*0.65=0.0159 -> ceil to 2c
def test_clustered_t_reduces_to_iid_when_groups_unique():
    x=np.array([1.,2.,3.,4.,5.]); m,se,t=E.clustered_t(x,np.arange(5)); assert abs(m-3)<1e-9 and abs(se-x.std(ddof=1)/np.sqrt(5))<1e-9
def test_block_bootstrap_shape():
    s=pd.Series(np.random.default_rng(0).normal(0.1,1,400)); lo,hi=E.block_bootstrap_mean(s,B=200); assert lo<0.1<hi

# ---- data-dependent tests (skip if data absent)
DATA=E.DATA
@pytest.mark.skipif(not (DATA/'kalshi/brackets.parquet').exists(), reason='data')
def test_one_winner_per_event_and_result_consistency():
    b=pd.read_parquet(DATA/'kalshi/brackets.parquet'); ev=pd.read_parquet(DATA/'weather/events.parquet')[['event_ticker','truth']].dropna()
    x=b.merge(ev,on='event_ticker'); x['py']=(x.truth>x.lo)&(x.truth<x.hi)
    x=x[x.event_ticker.str.extract(r'-(\d{2})[A-Z]{3}')[0].astype(int)>=23]   # 2021-22 markets used a different settlement/date convention and are excluded upstream (exclusions.parquet)
    ok=x.groupby('event_ticker').py.sum(); assert (ok==1).mean()>0.99, f'events without exactly one winning bracket: {(ok!=1).mean():.3%}'
    assert (((x.result=='yes')==x.py)).mean()>0.995, 'Kalshi result vs truth-derived winner mismatch'
    ex=pd.read_parquet(DATA/'kalshi/exclusions.parquet'); assert set(ex.reason.unique())<= {'winners!=1','kalshi_result_vs_truth_mismatch','no_settlement_truth','bracket_bounds_missing'}
@pytest.mark.skipif(not (DATA/'weather/candles_ev.parquet').exists(), reason='data')
def test_hours_since_open_is_not_a_clock():
    # documents why the release uses a clock-based decision time: h=8 (hours since open) is 22Z for only ~70-85% of rows
    c=pd.read_parquet(DATA/'weather/candles_ev.parquet',columns=['series','h','time','target'])
    s=c[c.h==8]; assert (s.time.dt.hour==22).mean()<0.95
@pytest.mark.skipif(not (E.RESULTS/'release/weather_trades.parquet').exists(), reason='release not built')
def test_release_decision_time_is_D_minus_1_22Z():
    t=pd.read_parquet(E.RESULTS/'release/weather_trades.parquet'); d=pd.to_datetime(t.decision_utc,utc=True)
    assert (d.dt.hour==22).all() and (d.dt.date<pd.to_datetime(t.target).dt.date).all()
@pytest.mark.skipif(not (E.RESULTS/'release/weather_trades.parquet').exists(), reason='release not built')
def test_no_training_after_cutoff_in_release_trades():
    t=pd.read_parquet(E.RESULTS/'release/weather_trades.parquet'); assert (pd.to_datetime(t.train_cutoff)<pd.to_datetime(t.target)).all()
@pytest.mark.skipif(not (E.RESULTS/'release/frozen_trades_sha256.txt').exists(), reason='no frozen trades')
def test_frozen_trades_reproduce():
    t=pd.read_parquet(E.RESULTS/'release/weather_trades.parquet'); h=pd.util.hash_pandas_object(t[['ticker','side','pnl']].round(6).reset_index(drop=True),index=False).sum()
    assert str(int(h))==open(E.RESULTS/'release/frozen_trades_sha256.txt').read().strip().split()[0]
