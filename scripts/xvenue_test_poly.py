import pandas as pd, numpy as np, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
R='/Users/abebraham/Desktop/resume/edge/data/'
Y=pd.read_parquet(R+'kalshi/xvenue_panel_poly.parquet').dropna(subset=['pxk','mu_k']); Y=Y[(Y.pn>0.005)&(Y.pn<0.995)]
def lg(p): p=np.clip(p,0.005,0.995); return np.log(p/(1-p))
Y['lp']=lg(Y.pn); Y['lx']=lg(Y.pxk); Y['tgt']=pd.to_datetime(Y.target); Y['month']=Y.tgt.dt.to_period('M')
# own-strip normal fit for Poly (control)
def fit_normal(lo,hi,p):
    from scipy.optimize import minimize
    c=np.where(np.isinf(lo),hi-1,np.where(np.isinf(hi),lo+1,(lo+hi)/2)); mu0=float((c*p).sum()/p.sum())
    def loss(th):
        mu,ls=th; sg=np.exp(ls); q=norm.cdf((hi-mu)/sg)-norm.cdf((lo-mu)/sg); return ((q-p)**2).sum()
    r=minimize(loss,[mu0,np.log(2.5)],method='Nelder-Mead',options={'xatol':1e-3,'fatol':1e-7,'maxiter':300}); return r.x[0],float(np.exp(r.x[1]))
F=[]
for (s,t),g in Y.groupby(['slug','ts']):
    if len(g)>=4: mu,sg=fit_normal(g.lo.values,g.hi.values,g.pn.values); F.append((s,t,mu,sg))
F=pd.DataFrame(F,columns=['slug','ts','mu_o','sig_o']); Y=Y.merge(F,on=['slug','ts'])
Y['po']=norm.cdf((Y.hi-Y.mu_o)/Y.sig_o)-norm.cdf((Y.lo-Y.mu_o)/Y.sig_o); Y['lo_']=lg(Y.po)
print('rows',len(Y),'events',Y.slug.nunique())
for cols in [['lp','lx'],['lp','lo_'],['lp','lx','lo_']]:
    m=sm.Logit(Y.y,sm.add_constant(Y[cols])).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(Y.slug)[0]})
    print(cols, m.params.round(3).to_dict(), 'z', m.tvalues.round(1).to_dict())
def wf(df,cols,warm=40):
    preds=[]
    for m in sorted(df.month.unique()):
        tr=df[df.month<m]; te=df[df.month==m]
        if tr.slug.nunique()<warm: continue
        mod=sm.Logit(tr.y,sm.add_constant(tr[cols])).fit(disp=0); preds.append(te.assign(ph=mod.predict(sm.add_constant(te[cols]))))
    return pd.concat(preds)
for cols,lab in [(['lp','lx'],'Kalshi-implied'),(['lp','lo_'],'own strip fit'),(['lp','lx','lo_'],'both')]:
    t=wf(Y,cols)
    for thr in [0.03,0.05,0.08]:
        for hs in [0.01,0.02]:
            ask=t.pn+hs; bid=t.pn-hs
            by=t[t.ph-ask>thr].sort_values('ts').drop_duplicates(['slug','lo']); bn=t[bid-t.ph>thr].sort_values('ts').drop_duplicates(['slug','lo'])
            pnl=pd.concat([by.y-(by.pn+hs),(1-bn.y)-(1-(bn.pn-hs))])
            if len(pnl)>5: print(f'POLY trade | {lab:15s} thr={thr} half-spread={hs}: N={len(pnl):4d} pnl/ct={pnl.mean():+.4f} t={pnl.mean()/pnl.std()*np.sqrt(len(pnl)):4.1f} win={(pnl>0).mean():.2f}')
