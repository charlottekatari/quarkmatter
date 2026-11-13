#!/usr/bin/env python3
"""Radial pulsation numerical audit for MAN hybrid-star model.
Gondek first-order system; rapid/slow interface matching, fixed-grid RK4+numba.
The stage-3 stellar model is imported and reused without modifying its EOS.
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import math
import numpy as np, pandas as pd
from scipy.optimize import brentq
from numba import njit
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage4_radial_modes import Background,Matter
from stage3_hybrid_tov import GEOM_PER_MEV_FM3 as CV, HBARC, C_SI

@njit(cache=True)
def ode_step(r,h,z0,z1,om2,A,C,D,K,AA,CC,DD,KK):
    xi,q=z0,z1
    def fo(x,y,a,c,d,k):return a*x+c*y,k*x+d*y
    k1x,k1q=fo(xi,q,A,C,D, K)
    k2x,k2q=fo(xi+h*k1x/2,q+h*k1q/2,(A+AA)/2,(C+CC)/2,(D+DD)/2, (K+KK)/2)
    k3x,k3q=fo(xi+h*k2x/2,q+h*k2q/2,(A+AA)/2,(C+CC)/2,(D+DD)/2, (K+KK)/2)
    k4x,k4q=fo(xi+h*k3x,q+h*k3q,AA,CC,DD,KK)
    return xi+h/6*(k1x+2*k2x+2*k3x+k4x),q+h/6*(k1q+2*k2q+2*k3q+k4q)

@njit(cache=True)
def propagate(rs,base,modeomega2, xi0,q0):
    x=xi0;q=q0;nodecount=0;prev=x
    for i in range(len(rs)-1):
        h=rs[i+1]-rs[i]
        a,c,d,k0,k1=base[i]
        aa,cc,dd,kk0,kk1=base[i+1]
        x,q=ode_step(rs[i],h,x,q,modeomega2,a,c,d,k0+modeomega2*k1,aa,cc,dd,kk0+modeomega2*kk1)
        if x*prev<0:nodecount+=1
        prev=x
        if abs(x)>1e150 or abs(q)>1e150:
            return np.nan,nodecount,x
    return q,nodecount,x

class Prepared:
    def __init__(self,bg,step=.015,surf_factor=10):
        self.bg=bg; self.parts=[];self.scale=bg.Pc*CV
        for j,(phase,r0,r1,sol) in enumerate(bg.parts):
            if j==len(bg.parts)-1 and surf_factor>1:
                lim=surf_factor*bg.mat.H.Pmin*CV
                r1=brentq(lambda r:sol.sol(r)[1]-lim,r0,r1)
            r=np.linspace(r0,r1,max(10,math.ceil((r1-r0)/step)+1))
            rows=np.empty((len(r),5))
            for i,rr in enumerate(r):
                m,p,nu,e,gp,fac,dp=bg.thermo(rr,j)
                rows[i,0]=-3/rr-dp/(e+p)
                rows[i,1]=-self.scale/(gp*rr)
                rows[i,2]=dp/(e+p)-4*np.pi*(e+p)*rr*fac
                rows[i,3]=(-4*dp-8*np.pi*fac*(e+p)*p*rr+rr*dp*dp/(e+p))/self.scale
                rows[i,4]=fac*np.exp(-2*nu)*(e+p)*rr/self.scale
            self.parts.append((r,rows))
        _,gpc=bg.mat.vals(bg.Pc*CV,bg.phase,bg.B)
        self.q0=-3*gpc/self.scale
        if bg.rq:
            _,_,_,_,_,_,dpq=bg.thermo(bg.rq,0)
            _,_,_,_,_,_,dph=bg.thermo(bg.rq,1)
            self.jumpcoeff=self.scale/bg.rq*(1/dph-1/dpq)
        else:self.jumpcoeff=0
    def f(self,omega2,mode='rapid',with_nodes=False):
        xi=1.;q=self.q0;nodes=0
        for j,(rs,a) in enumerate(self.parts):
            q,nn,xi=propagate(rs,a,omega2,xi,q)
            nodes+=nn
            if not np.isfinite(q):return (np.nan,nodes) if with_nodes else np.nan
            if j==0 and len(self.parts)==2 and mode=='rapid':xi+=q*self.jumpcoeff
        return (q,nodes) if with_nodes else q

def find_roots(obj, mode, scan=None):
    if scan is None:
        scan=np.unique(np.r_[np.linspace(-.012,-.001,19),np.linspace(-.001,.001,31),np.linspace(.001,.024,30)])
    vals=np.array([obj.f(float(x),mode) for x in scan])
    results=[]
    for i in range(len(scan)-1):
        if np.isfinite(vals[i]*vals[i+1]) and vals[i]*vals[i+1]<0:
            r=brentq(lambda x:obj.f(x,mode),scan[i],scan[i+1],xtol=1e-10)
            _,nodes=obj.f(r,mode,with_nodes=True)
            results.append((r,nodes))
    return results

def main():
    res=Path(str(BASE_DIR/'stage4_results'));res.mkdir(exist_ok=True)
    mat=Matter(str(BASE_DIR/'eos.zip'))
    tr=pd.read_csv(str(BASE_DIR/'stage2_results/MIT_Bag_Maxwell_candidates.csv'))
    sets={0:[55,120,350,650],175:[21,23,30,90,200,450,575,700],185:[80,81,100,140,200,370,425,600],190:[112,113,120,180,350]}
    rows=[]
    for b,pc_list in sets.items():
        if b:Pt=float(tr[tr.B_root_MeV==b]['P_transition_MeV_fm^-3'].iloc[0]);B=b**4/HBARC**3
        else:Pt=None;B=None
        for pc in pc_list:
            if b and pc<=Pt:continue
            try:
                bg=Background(float(pc),mat,Pt,B,max_step=.12)
                obj=Prepared(bg,step=.023,surf_factor=10)
                for mode in (('rapid','slow') if b else ('pure',)):
                    rt=find_roots(obj, mode)
                    data={'Broot':b,'Pc':pc,'M':bg.mass,'R':bg.R,'Rq':bg.rq,'scenario':mode,
                          'first_root':rt[0][0] if rt else np.nan,'first_nodes':rt[0][1] if rt else np.nan,
                          'root_count':len(rt),'roots':repr(rt)}
                    rows.append(data)
                    print(data,flush=True)
            except Exception as exc:print('ERROR',b,pc,exc,flush=True)
            pd.DataFrame(rows).to_csv(res/'radial_mode_trials.csv',index=False)
    return rows

if __name__=='__main__':main()
