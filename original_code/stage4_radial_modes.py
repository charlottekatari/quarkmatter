#!/usr/bin/env python3
"""Exploratory relativistic radial oscillations, Gondek form; includes phase conversion matching.
Main checks: eigenvalue residual, eigenfunction nodes, convergence and control stars.
Units: r,m km, p,eps km^-2; omega^2 km^-2. Frequencies kHz if omega^2>0.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_hybrid_tov import load_hadron, Hadron, GEOM_PER_MEV_FM3 as CV, KM_PER_M_SUN as MS, q_eps, HBARC, C_SI

class Matter:
    def __init__(self,zpath):
        h=load_hadron(zpath); self.H=Hadron(h)
        self.dlogeps_dlogP=self.H.f_logeps.derivative()
    def vals(self,p,phase,B):
        P=float(p)/CV
        if phase=='Q':
            ep=CV*q_eps(P,B)
            gamma_p=(ep+p)/3.
        else:
            P=np.clip(P,self.H.Pmin,self.H.Pmax)
            ep=CV*self.H.eps(P)
            dlog=float(self.dlogeps_dlogP(np.log(P)))
            cs2=P/(self.H.eps(P)*dlog)
            gamma_p=(ep+p)*cs2
        return ep,gamma_p

def bg_rhs(r,y,matter,phase,B):
    m,p,nu=y
    e,_=matter.vals(p,phase,B)
    fl=1-2*m/r
    if fl<=0:raise ValueError('horizon')
    dmdr=4*np.pi*r*r*e
    dpdr=-(e+p)*(m+4*np.pi*r**3*p)/(r*(r-2*m))
    dnudr=-dpdr/(e+p)
    return [dmdr,dpdr,dnudr]

class Background:
    def __init__(self,Pc,matter,Pt=None,B=None, max_step=.09,rtol=3e-9):
        self.Pc=Pc;self.mat=matter;self.Pt=Pt;self.B=B;self.phase='H' if Pt is None or Pc<=Pt else 'Q'
        r0=1e-4
        p=Pc*CV
        e,_=matter.vals(p,self.phase,B)
        m=4*np.pi*e*r0**3/3
        p-=2*np.pi/3*(e+p)*(e+3*p)*r0**2
        self.r0=r0; self.parts=[]
        def integrate(phase,rstart,y0,stop):
            target=stop*CV
            def event(r,y):return y[1]-target
            event.terminal=True;event.direction=-1
            sol=solve_ivp(lambda r,y:bg_rhs(r,y,matter,phase,B),(rstart,1500),y0,events=event,
                         rtol=rtol,atol=[1e-12,3e-18,1e-12],dense_output=True,max_step=max_step,method='DOP853')
            if not sol.success or len(sol.t_events[0])!=1: raise RuntimeError(f'Background failed: {sol.message}')
            rend=sol.t_events[0][0];yend=sol.y_events[0][0]
            self.parts.append((phase,rstart,rend,sol))
            return rend,yend
        if self.phase=='Q':
            self.rq,yy=integrate('Q',r0,[m,p,0],Pt)
            rend,yend=integrate('H',self.rq,yy,matter.H.Pmin)
        else:
            self.rq=0;rend,yend=integrate('H',r0,[m,p,0],matter.H.Pmin)
        self.R=rend;self.M=yend[0]; self.mass=self.M/MS
        self.nu_shift=.5*np.log(1-2*self.M/self.R)-yend[2]
    def y(self,r,idx):return self.parts[idx][3].sol(r)
    def thermo(self,r,idx):
        phase=self.parts[idx][0];m,p,nu=self.y(r,idx)
        ep,gp=self.mat.vals(p,phase,self.B)
        fac=1/(1-2*m/r)
        dp=-(ep+p)*(m+4*np.pi*r**3*p)/(r*(r-2*m))
        return m,p,nu+self.nu_shift,ep,gp,fac,dp

def integrate_pulse(bg,om2, conversion='rapid', surface_factor=1, rtol=8e-8, step=.08, track_nodes=False):
    # normalized Lagrangian pressure perturbation q=Delta p / Pc_geom
    scale=bg.Pc*CV
    e,gp=bg.mat.vals(bg.Pc*CV,bg.phase,bg.B)
    initial=[1.0,-3*gp/scale]
    stored=[]
    def pulse_rhs(r,y,idx):
        xi,q=y
        m,p,nu,e,gp,fac,dp=bg.thermo(r,idx)
        dxi=-(3*xi+q*scale/gp)/r -dp/(e+p)*xi
        dq=(xi*(om2*fac*np.exp(-2*nu)*(e+p)*r-4*dp -8*np.pi*fac*(e+p)*p*r +r*dp*dp/(e+p))
            + q*scale*(dp/(e+p)-4*np.pi*(e+p)*r*fac))/scale
        return [dxi,dq]
    for idx,(phase,rs,re,sol) in enumerate(bg.parts):
        rend=re
        if idx==len(bg.parts)-1 and surface_factor>1:
            # find r at surface pressure multiple of min, by brentq
            lim=bg.mat.H.Pmin*CV*surface_factor
            rend=brentq(lambda r:sol.sol(r)[1]-lim,rs,re)
        pulses=solve_ivp(lambda r,y:pulse_rhs(r,y,idx),(rs,rend),initial,
                rtol=rtol,atol=[2e-7,8e-9],method='DOP853',max_step=step,
                dense_output=track_nodes)
        if not pulses.success or not np.all(np.isfinite(pulses.y)):
            return np.nan, -1, None
        end=pulses.y[:,-1]
        if track_nodes:
            stored.append((pulses.t,pulses.y[0].copy(),pulses.y[1].copy()))
        if phase=='Q' and idx+1<len(bg.parts):
            if conversion=='rapid':
                r=re
                _,_,_,_,_,_,dpq=bg.thermo(r,idx)
                _,_,_,_,_,_,dph=bg.thermo(r,idx+1)
                # xi is delta_r/r: xi_out=xi_in + DeltaP/r (1/P'_out - 1/P'_in)
                end[0]+=end[1]*scale/r*(1/dph-1/dpq)
            elif conversion!='slow':raise ValueError('bad conversion')
        initial=end
    node_count=0
    if track_nodes:
        # Count spatial sign changes of xi, skipping interface jump to avoid false positive
        for tt,xx,qq in stored:
            mask=np.abs(xx)>1e-6
            if np.sum(mask)>1:
                node_count+=int(np.count_nonzero(np.diff(np.sign(xx[mask]))))
    return end[1],node_count,stored

def eigenroots(bg,mode, grid=None, low=-.01, high=.016, surface_factor=10, step=.12):
    if grid is None:
        # dense near 0: fundamental up to ~ few kHz
        grid=np.r_[np.linspace(low,-.0001,18),np.linspace(-.0001,.0001,14),np.linspace(.0001,high,26)]
    values=[]
    for x in grid:
        z,_,_=integrate_pulse(bg,float(x),mode,surface_factor,step=step)
        values.append(z)
    roots=[]
    for ia in range(len(grid)-1):
        if np.isfinite(values[ia]) and np.isfinite(values[ia+1]) and values[ia]*values[ia+1]<0:
            try:
                root=brentq(lambda x:integrate_pulse(bg,x,mode,surface_factor,step=step)[0],grid[ia],grid[ia+1],xtol=3e-9)
                resid,nodes,pts=integrate_pulse(bg,root,mode,surface_factor,step=step,track_nodes=True)
                roots.append(dict(omega2=root,nodes=nodes,residual=resid))
            except Exception:pass
    return roots

def classify(roots):
    if not roots:return (np.nan,'unresolved')
    # selection of true fundamental mode requires 0-node; rapid interface xi jumps can change sign
    roots=sorted(roots,key=lambda x:x['omega2'])
    low=roots[0]
    return (low['omega2'],'stable' if low['omega2']>0 else 'unstable')

if __name__=='__main__':
    mat=Matter(str(BASE_DIR/'eos.zip'))
    trans=pd.read_csv(str(BASE_DIR/'stage2_results/MIT_Bag_Maxwell_candidates.csv'))
    out=Path(str(BASE_DIR/'stage4_results'));out.mkdir(exist_ok=True)
    rows=[]
    # representative configurations: onset, left of extrema, middle declining, second branch, end
    sel={0:[60,130,250,450],175:[21,23,30,90,200,450,575,700],185:[80,81,100,140,200,370,425,600],190:[112,113,120,180,350]}
    for br,pcs in sel.items():
        if br:
            r=trans[trans.B_root_MeV==br].iloc[0];Pt=float(r['P_transition_MeV_fm^-3']);B=br**4/HBARC**3
        else:Pt=None;B=None
        for pc in pcs:
            if br and pc<=Pt:continue
            try:
                bg=Background(pc,mat,Pt,B)
                for conv in (['rapid','slow'] if br else ['pure']):
                    mode='rapid' if conv=='pure' else conv
                    roots=eigenroots(bg,mode,surface_factor=10)
                    omega2,status=classify(roots)
                    rec=dict(Broot=br,Pc=pc,M=bg.mass,R=bg.R,Rq=bg.rq,mode=conv,omega2=omega2,status=status,
                             roots=';'.join(f'{a["omega2"]:.7f} ({a["nodes"]} nodes)' for a in roots))
                    rows.append(rec)
                    print(rec,flush=True)
            except Exception as er:print('FAIL',br,pc,str(er),flush=True)
            pd.DataFrame(rows).to_csv(out/'radial_mode_trials.csv',index=False)
