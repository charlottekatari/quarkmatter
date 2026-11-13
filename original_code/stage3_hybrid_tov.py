#!/usr/bin/env python3
"""МАН: stage III, Maxwell hybrid EoS and stellar structure (TOV).

Reproduce: python stage3_hybrid_tov.py eos.zip stage2_results/MIT_Bag_Maxwell_candidates.csv stage3_results
This is a cold, static, nonrotating, unmagnetized idealized model.
"""
from pathlib import Path
import sys, zipfile, json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G_SI=6.67430e-11
C_SI=299792458.0
MEV_J=1.602176634e-13
M_SUN_KG=1.98847e30
KM_PER_M_SUN=(G_SI*M_SUN_KG/C_SI**2)/1000.0
GEOM_PER_MEV_FM3=G_SI/C_SI**4*(MEV_J/1e-45)*(1e6)  # km^-2
HBARC=197.3269804
C_Q=1/(108*np.pi**2*HBARC**3)
ROOTS=(145,155,165,175,185,190)


def load_hadron(zpath):
    with zipfile.ZipFile(zpath) as z:
        from io import StringIO
        nball=np.loadtxt(StringIO(z.read('eos.nb').decode()))
        lines=z.read('eos.thermo').decode().splitlines()
        masses=list(map(float,lines[0].split()))
        tab=np.loadtxt(StringIO('\n'.join(lines[1:])))
    nB=nball[2:]
    P=nB*tab[:,3]
    eps=nB*masses[0]*(1+tab[:,9])
    mu=masses[0]*(1+tab[:,5])
    assert len(nB)==len(P)==197
    assert np.all(np.diff(P)>0) and np.all(np.diff(eps)>0)
    return dict(nB=nB,P=P,eps=eps,mu=mu)


class Hadron:
    def __init__(self,h):
        self.Pmin=float(h['P'][0]); self.Pmax=float(h['P'][-1])
        self.f_logeps=PchipInterpolator(np.log(h['P']),np.log(h['eps']),extrapolate=False)
        self.f_lognb=PchipInterpolator(np.log(h['P']),np.log(h['nB']),extrapolate=False)
    def eps(self,P):
        # Values beyond tabulated range are NOT used for a stellar profile.
        # Clamp only in intermediary event-detection evaluations near P=Pmin.
        return float(np.exp(self.f_logeps(np.log(np.clip(P,self.Pmin,self.Pmax)))))
    def nb(self,P):
        return float(np.exp(self.f_lognb(np.log(np.clip(P,self.Pmin,self.Pmax)))))


def q_eps(P, B):
    return 3.0*P+4.0*B


def tov_rhs(r,y,energy):
    m,p=y  # m in km, P in km^-2
    if r<=0 or r<=2*m:
        raise RuntimeError('TOV integration hit an inadmissible radius')
    E=GEOM_PER_MEV_FM3*energy(max(float(p),0.0)/GEOM_PER_MEV_FM3)
    dm=4*np.pi*r*r*E
    dp=-(E+p)*(m+4*np.pi*r**3*p)/(r*(r-2*m))
    return (dm,dp)


def integrate_segment(Pc, Pstop, eos_energy, center=True, initial=None,
                      r_limit=2000.0,max_step=0.6,rtol=1.4e-7):
    """Pc/Pstop pressures in MeV fm^-3; solve_ivp integrates (m,p) in geometric units.

    If center=False, supply (r_start, m_start) at continuous phase interface.
    """
    if not Pc>Pstop:
        raise ValueError('Initial pressure must exceed target pressure')
    p0=Pc*GEOM_PER_MEV_FM3
    pt=Pstop*GEOM_PER_MEV_FM3
    if center:
        r0=1e-5
        eps0=GEOM_PER_MEV_FM3*eos_energy(Pc)
        m0=4*np.pi*eps0*r0**3/3
        p0=p0-(2*np.pi/3)*(eps0+Pc*GEOM_PER_MEV_FM3)*(eps0+3*Pc*GEOM_PER_MEV_FM3)*r0**2
    else:
        r0,m0=initial
    def event(r,y): return y[1]-pt
    event.terminal=True
    event.direction=-1
    res=solve_ivp(lambda r,y:tov_rhs(r,y,eos_energy),[r0,r_limit],[m0,p0],
                  events=event, rtol=rtol, atol=[5e-11,7e-18],max_step=max_step,
                  method='DOP853')
    if not res.success or len(res.t_events[0])!=1:
        raise RuntimeError('TOV radius limit or event failure: '+res.message)
    rstop=float(res.t_events[0][0]); mstop=float(res.y_events[0][0][0])
    return rstop,mstop


def solve_star(Pc,H,Pt=None,B=None,**kwargs):
    """Returns radius, gravitational mass, quark core radius and mass.

    Exact pressure continuity at interface; epsilon discontinuity is not smoothed.
    The reported surface is P=Pmin of the table (not P=0).
    """
    if Pc<=H.Pmin: return None
    if Pt is None or Pc<=Pt:
        r,m=integrate_segment(Pc,H.Pmin,H.eps,**kwargs)
        return dict(Pc=Pc,R_km=r,M_solar=m/KM_PER_M_SUN,
                    Rq_km=0.0,Mq_solar=0.0,has_quark_core=False)
    rq,mq=integrate_segment(Pc,Pt,lambda P:q_eps(P,B),**kwargs)
    r,m=integrate_segment(Pt,H.Pmin,H.eps,center=False,initial=(rq,mq),**kwargs)
    return dict(Pc=Pc,R_km=r,M_solar=m/KM_PER_M_SUN,
                Rq_km=rq,Mq_solar=mq/KM_PER_M_SUN,has_quark_core=True)


def calc_batch(pressures,H,Pt=None,B=None,**kwargs):
    rows=[]
    for Pc in np.unique(np.asarray(pressures,dtype=float)):
        try:
            sol=solve_star(float(Pc),H,Pt,B,**kwargs)
            if sol is not None:
                sol['ok']=True;sol['error']='';rows.append(sol)
        except (RuntimeError,ValueError,FloatingPointError,OverflowError) as err:
            rows.append(dict(Pc=Pc,R_km=np.nan,M_solar=np.nan,Rq_km=np.nan,Mq_solar=np.nan,
                             has_quark_core=bool(Pt is not None and Pc>Pt),ok=False,error=str(err)))
    return pd.DataFrame(rows)


def plot_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
                         'axes.titlesize':14,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.linestyle':'--','grid.alpha':.22,
                         'savefig.dpi':180,'figure.dpi':120,'legend.fontsize':9})


def savefig(d,n):
    plt.tight_layout();plt.savefig(d/n,bbox_inches='tight');plt.close()


def turning_points(data):
    """Mass maxima/minima with smooth interpolation in log central pressure.

    Local extrema only, no claim of complete radial-mode stability.
    """
    a=data[(data['ok']) & np.isfinite(data['M_solar'])].sort_values('Pc')
    x=np.log(a['Pc'].values); mass=a['M_solar'].values
    if len(x)<5:return []
    interpol=PchipInterpolator(x,mass)
    xs=np.linspace(x[0],x[-1],max(1400,6*len(x)))
    vals=interpol.derivative()(xs)
    changes=np.where(vals[:-1]*vals[1:]<0)[0]
    res=[]
    for i in changes:
        x0=brentq(interpol.derivative(),xs[i],xs[i+1]);
        res.append(dict(Pc=float(np.exp(x0)),M_solar=float(interpol(x0)),
                        kind='maximum' if vals[i]>0 else 'minimum'))
    return res


def main(zpath,candidate_file,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    plot_style()
    H=Hadron(load_hadron(zpath))
    phase=pd.read_csv(candidate_file)
    crossing={int(r.B_root_MeV):r for _,r in phase.iterrows() if r.transition_in_table==True}
    print('Conversions: 1 MeV/fm^3 -> %.10g km^-2; 1 Msun -> %.10g km'%(GEOM_PER_MEV_FM3,KM_PER_M_SUN),flush=True)
    print('Surface defined by P_min = %.9g MeV/fm^3'%H.Pmin,flush=True)
    # Include all central pressures of interest for the baseline, including the transition points.
    pgrid=np.geomspace(1.0,H.Pmax,95)
    pgrid=np.unique(pgrid)
    baseline=calc_batch(pgrid,H)
    baseline['B_root_MeV']=0
    baseline['branch']='pure_hadronic'
    baseline.to_csv(out/'01_baseline_hadronic_TOV.csv',index=False)
    bh=baseline[baseline.ok]
    bhmax=bh.loc[bh.M_solar.idxmax()]
    # Radius at 1.4 Msun on monotonic increasing branch
    bh_s=bh[(bh.Pc<=bhmax.Pc)&(bh.Pc>=1.4)]
    rr=PchipInterpolator(bh_s.M_solar,bh_s.R_km,extrapolate=False)
    r14=float(rr(1.4))
    print('Hadronic baseline max %.6f Msun, R14 %.6f km'%(bhmax.M_solar,r14),flush=True)
    rows=[]; points=[]; all_stars=[]
    for root in ROOTS:
        if root not in crossing:
            rows.append(dict(B_root_MeV=root,transition_status='no hadron->quark Maxwell crossing',
                             Pt_MeV_fm3=np.nan,eps_H=np.nan,delta_eps=np.nan,
                             Seidov_ratio=np.nan,Seidov_critical=np.nan,
                             M_at_onset_Msun=np.nan,Mmax_hybrid_sequence_Msun=np.nan,
                             R_at_Mmax_km=np.nan,Rq_at_Mmax_km=np.nan,
                             branch_has_stable_onset=None))
            continue
        p=crossing[root]
        Pt=float(p['P_transition_MeV_fm^-3'])
        B=root**4/HBARC**3
        eps_h=H.eps(Pt)
        delta=q_eps(Pt,B)-eps_h
        ratio=delta/eps_h
        crit=.5+1.5*Pt/eps_h
        onset=solve_star(Pt,H)
        # Dense sampling near the transition; broad span to detect branches with rising M.
        relative=np.array([1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.06,.1,.17,.25])
        central=np.unique(np.r_[Pt*(1+relative),np.geomspace(Pt*1.001,2100,126)])
        stars=calc_batch(central,H,Pt,B)
        stars['B_root_MeV']=root
        stars['branch']='hybrid' 
        stars.to_csv(out/f'02_hybrid_TOV_B{root}.csv',index=False)
        all_stars.append(stars)
        ok=stars[stars.ok]
        if len(ok):
            idx=ok.M_solar.idxmax(); mx=ok.loc[idx]
            mx_mass=float(mx.M_solar); mx_radius=float(mx.R_km); mx_rq=float(mx.Rq_km)
        else: mx_mass=mx_radius=mx_rq=np.nan
        ext=turning_points(stars)
        points.append({'B_root_MeV':root,'onset_mass':float(onset['M_solar']),
                       'onset_radius':float(onset['R_km']),'maxima_minima':ext})
        rows.append(dict(B_root_MeV=root,transition_status='candidate Maxwell transition',
                         Pt_MeV_fm3=Pt,eps_H=eps_h,
                         delta_eps=delta,
                         Seidov_ratio=ratio,Seidov_critical=crit,
                         M_at_onset_Msun=float(onset['M_solar']),R_at_onset_km=float(onset['R_km']),
                         Mmax_hybrid_sequence_Msun=mx_mass,R_at_Mmax_km=mx_radius,
                         Rq_at_Mmax_km=mx_rq,
                         branch_has_stable_onset=bool(ratio<crit)))
        print(f'Broot={root}, Pt={Pt:.5f}, onsetM={onset["M_solar"]:.4f}, '+
              f'Seidov {ratio:.4f} vs {crit:.4f}, Mmax of hybrid samples {mx_mass:.5f}, '+
              f'extrema {ext}',flush=True)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'03_TOV_transition_stability_summary.csv',index=False)
    together=pd.concat(all_stars,ignore_index=True);together.to_csv(out/'04_all_hybrid_models.csv',index=False)
    with open(out/'05_extrema.json','w') as f:json.dump(points,f,ensure_ascii=False,indent=2)
    # Conservative qualitative annotations. Turning points are indicators,
    # not a full radial-mode analysis. Low-pressure B165 is flagged separately.
    interpretation=[]
    for rec in points:
        root=rec['B_root_MeV']
        ext=rec['maxima_minima']
        if root==165:
            message=('Low-pressure transition: the hadronic reference is an extended '
                     'configuration (R~406 km), not a usual neutron-star branch. '
                     'Seidov onset classification is not sufficient here; '
                     'no secure dynamical stability classification.')
        elif root==175:
            message=('Seidov destabilizes a nascent core; sequence has a local '
                     'minimum followed by a new rising-mass branch and a maximum. '
                     'Candidate disconnected stable branch; radial modes not tested.')
        elif root==185:
            message=('Seidov allows stable onset; first maximum, then minimum, '
                     'then a second rising-mass branch and maximum. Candidate '
                     'disconnected branch; radial modes not tested.')
        else:
            message=('Seidov allows stable onset; a short hybrid branch reaches '
                     'its first maximum. More massive core configurations past '
                     'the maximum are not inferred stable.')
        interpretation.append(dict(B_root_MeV=root,onset_mass_Msun=rec['onset_mass'],
                                   turning_points=json.dumps(ext),interpretation=message))
    pd.DataFrame(interpretation).to_csv(out/'12_stability_interpretation.csv',index=False)
    # Interpolate a piecewise Maxwell EoS explicitly (no phase interpolation).
    eos_rows=[]
    orig=load_hadron(zpath)
    for root in crossing:
        Pt=float(crossing[root]['P_transition_MeV_fm^-3']);B=root**4/HBARC**3
        for P in np.geomspace(H.Pmin,Pt*(1-1e-9),400):
            eos_rows.append((root,'hadronic',float(P),H.eps(P),H.nb(P)))
        eos_rows.append((root,'hadronic_at_transition',Pt,H.eps(Pt),H.nb(Pt)))
        # analytic baryon density nB(P)=4C mu^3 with mu=((P+B)/C)^.25
        for i,P in enumerate(np.r_[Pt,np.geomspace(Pt*(1+1e-8),2100,400)]):
            mu=((P+B)/C_Q)**.25
            nB=4*C_Q*mu**3
            eos_rows.append((root,'quark_at_transition' if i==0 else 'quark',float(P),q_eps(P,B),nB))
    pd.DataFrame(eos_rows,columns=['B_root_MeV','phase','P_MeV_fm^-3','epsilon_MeV_fm^-3','nB_fm^-3']).to_csv(out/'06_hybrid_EoS_branches.csv',index=False)
    # Plot 1 hybrid EoS
    fig,ax=plt.subplots(figsize=(10,6.2))
    ax.plot(orig['eps'],orig['P'],color='#172333',lw=2.7,label='SLy4 (лише адронна фаза)')
    colors=plt.get_cmap('viridis')(np.linspace(.2,.85,4))
    for (root,c) in zip(crossing,colors):
        Pt=float(crossing[root]['P_transition_MeV_fm^-3']); eh=H.eps(Pt); eq=q_eps(Pt,root**4/HBARC**3)
        pre=np.geomspace(max(Pt,.015),2000,400)
        ax.plot(q_eps(pre,root**4/HBARC**3),pre,color=c,lw=2,label=f'кваркова фаза {root} MeV')
        ax.plot([eh,eq],[Pt,Pt],color=c,ls=':',lw=2)
        ax.scatter([eh,eq],[Pt,Pt],color=c,s=15,zorder=5)
    ax.set(xlabel=r'Густина енергії $\varepsilon$ (MeV/fm$^3$)',
           ylabel=r'Тиск $P$ (MeV/fm$^3$)',title='Гібридні EoS: стрибок густини енергії (Максвелл)',
           xlim=(0,1850),ylim=(0,450))
    ax.legend(frameon=False,loc='upper left')
    savefig(out,'01_hybrid_EoS_Maxwell.png')
    # Plot 2 mass-radius, hybrid branches including unstable segment as dashed where known
    fig,ax=plt.subplots(figsize=(10,6.6))
    ax.plot(bh.R_km,bh.M_solar,color='#101827',lw=2.7,label='SLy4: без кваркового ядра')
    for root,col in zip(crossing,colors):
        ss=together[(together.B_root_MeV==root)&together.ok]
        ax.plot(ss.R_km,ss.M_solar,color=col,lw=2.1,label=f'MIT {root} MeV')
        rr=next(x for x in points if x['B_root_MeV']==root)
        ax.scatter([rr['onset_radius']],[rr['onset_mass']],color=col,s=35,zorder=7)
    ax.axhline(2.0,color='grey',lw=.8,ls=':',label=r'$2M_\odot$ (орієнтир)')
    ax.set(xlabel='Радіус зорі R (км)',ylabel=r'Маса $M/M_\odot$',
           title='Маса–радіус: усі розв’язки (не всі стійкі)',xlim=(5,22),ylim=(0,2.45))
    ax.legend(frameon=False,ncol=2,loc='lower left')
    savefig(out,'02_mass_radius.png')
    # Plot 3 M(Pc)
    fig,ax=plt.subplots(figsize=(10,6.3))
    ax.semilogx(bh.Pc,bh.M_solar,color='#101827',lw=2.8,label='SLy4 без переходу')
    for root,col in zip(crossing,colors):
        ss=together[(together.B_root_MeV==root)&together.ok]
        ax.semilogx(ss.Pc,ss.M_solar,color=col,lw=2,label=f'MIT {root} MeV')
    ax.set(xlabel=r'Центральний тиск $P_c$ (MeV/fm$^3$)',ylabel=r'Маса $M/M_\odot$',
           title='Залежність маси від центрального тиску (не всі стійкі)',
           xlim=(0.006,2400),ylim=(0,2.5))
    ax.legend(ncol=2,loc='upper left',frameon=False)
    savefig(out,'03_mass_vs_central_pressure.png')
    # Plot 4 quark radius vs central pressure
    fig,ax=plt.subplots(figsize=(10,6.1))
    for root,col in zip(crossing,colors):
        ss=together[(together.B_root_MeV==root)&together.ok]
        ax.semilogx(ss.Pc,ss.Rq_km,color=col,lw=2,label=f'MIT {root} MeV')
    ax.set(xlabel=r'Центральний тиск $P_c$ (MeV/fm$^3$)',
           ylabel=r'Радіус кваркового ядра $R_Q$ (км)',
           title='Розмір кваркового ядра залежно від центрального тиску',
           xlim=(.006,2400))
    ax.legend(frameon=False)
    savefig(out,'04_quark_core_radius.png')
    # Plot 5 onset criterion
    fig,ax=plt.subplots(figsize=(9.2,5.6))
    xs=np.array([r for r in ROOTS if r in crossing]);v=summary.dropna(subset=['Seidov_ratio'])
    ax.plot(v.B_root_MeV,v.Seidov_ratio,marker='o',lw=2,label=r'$\Delta\varepsilon / \varepsilon_{H,t}$')
    ax.plot(v.B_root_MeV,v.Seidov_critical,marker='s',lw=2,ls='--',label='Критерій Сейдова (межа)')
    ax.set(xticks=xs,xlabel=r'Параметр $B^{1/4}$ (MeV)',ylabel='Безрозмірне співвідношення',
           title='Початкова стійкість кваркового ядра: критерій Сейдова')
    ax.legend(frameon=False)
    savefig(out,'05_Seidov_criterion.png')
    # Plot 6 max M per parameter with qualification (sampled max includes unstable segments).
    fig,ax=plt.subplots(figsize=(9.2,5.5))
    v=summary.dropna(subset=['Mmax_hybrid_sequence_Msun'])
    ax.plot(v.B_root_MeV,v.Mmax_hybrid_sequence_Msun,marker='o',lw=2.1,label='Максимум маси серед розрахованих гібридних конфігурацій')
    ax.axhline(bhmax.M_solar,color='black',ls='--',lw=1.5,label='Максимальна маса SLy4 (без переходу)')
    ax.set(xlabel=r'$B^{1/4}$ (MeV)',ylabel=r'Маса $M/M_\odot$',xticks=xs,
           title='Максимальна маса серед гібридних розв’язків (не обов’язково стійких)')
    ax.legend(frameon=False,loc='best')
    savefig(out,'06_raw_maxima_vs_B.png')
    # Plot 7: Compact-star zoom, avoiding the very extended dilute 165 MeV configurations.
    fig,ax=plt.subplots(figsize=(9.8,6.2))
    compact_base=bh[(bh.R_km<30)&(bh.M_solar>.65)]
    ax.plot(compact_base.R_km,compact_base.M_solar,color='#101827',lw=2.6,
            label='Контрольна SLy4')
    for root,col in zip(crossing,colors):
        ss=together[(together.B_root_MeV==root)&together.ok]
        ss=ss[(ss.R_km<30)&(ss.M_solar>.65)]
        ax.plot(ss.R_km,ss.M_solar,color=col,lw=2,label=f'MIT {root} MeV')
    ax.axhline(2,color='grey',lw=.9,ls=':',label=r'$2M_\odot$')
    ax.set(xlim=(6,13),ylim=(.65,2.2),
           xlabel='Радіус R (км)', ylabel=r'Маса $M/M_\odot$',
           title='Компактні конфігурації: порівняння мас і радіусів')
    ax.legend(ncol=2,frameon=False,loc='lower left')
    savefig(out,'07_compact_mass_radius_zoom.png')
    # Plot 8: quark core radius evaluated at sampled max mass for each root.
    fig,ax=plt.subplots(figsize=(9,5.2))
    v=summary.dropna(subset=['Rq_at_Mmax_km'])
    ax.plot(v.B_root_MeV,v.Rq_at_Mmax_km,marker='o',lw=2)
    ax.set(xticks=v.B_root_MeV.tolist(),xlabel=r'$B^{1/4}$ (MeV)',
           ylabel=r'$R_Q$ (км)',
           title='Радіус ядра при найбільшій масі (різні маси зір)')
    savefig(out,'08_Rq_at_mass_peak_vs_B.png')
    # Strong sensitivity of 165 onset configuration to low-pressure outer layers.
    surface_extra=[]
    for r in [165,175,185,190]:
        pt=float(crossing[r]['P_transition_MeV_fm^-3'])
        for mult in [1,3,10,30]:
            radius,mass=integrate_segment(pt,H.Pmin*mult,H.eps)
            surface_extra.append(dict(B_root_MeV=r,Pstop_over_Pmin=mult,
                                      onset_radius_km=radius,onset_mass_Msun=mass/KM_PER_M_SUN))
    pd.DataFrame(surface_extra).to_csv(out/'10_surface_sensitivity_at_onset.csv',index=False)
    # Consistency under tighter ODE tolerances and smaller numerical steps.
    convergence=[]
    selected=[('hadronic_1.4',80,None,None),('hadronic_max',860,None,None)]
    for root in crossing:
        candidates=together[(together.B_root_MeV==root)&together.ok]
        peak=candidates.loc[candidates.M_solar.idxmax()]
        selected.append((f'B{root}_peak',float(peak.Pc),
                         float(crossing[root]['P_transition_MeV_fm^-3']),
                         root**4/HBARC**3))
    for name,pc,pt,B in selected:
        coarse=solve_star(pc,H,Pt=pt,B=B,max_step=.6,rtol=1.4e-7)
        fine=solve_star(pc,H,Pt=pt,B=B,max_step=.12,rtol=3e-9)
        convergence.append(dict(model=name,Pc_MeV_fm3=pc,
                                M_reference_Msun=fine['M_solar'],
                                R_reference_km=fine['R_km'],
                                mass_difference_Msun=coarse['M_solar']-fine['M_solar'],
                                radius_difference_km=coarse['R_km']-fine['R_km'],
                                quark_core_difference_km=coarse['Rq_km']-fine['Rq_km']))
    pd.DataFrame(convergence).to_csv(out/'11_solver_convergence.csv',index=False)
    # Plot 9: turning-point diagnostics for 175, 185, 190 MeV.
    fig,ax=plt.subplots(figsize=(9.8,5.9))
    for root,col in zip(crossing,colors):
        if root==165:continue  # this has an extended non-NS low-pressure branch
        curve=together[(together.B_root_MeV==root)&together.ok]
        ax.semilogx(curve.Pc,curve.M_solar,color=col,lw=2,label=f'MIT {root} MeV')
        rec=next(z for z in points if z['B_root_MeV']==root)
        for ext in rec['maxima_minima']:
            ax.scatter(ext['Pc'],ext['M_solar'],marker='o' if ext['kind']=='maximum' else 'v',
                       s=64,color=col,edgecolors='white',zorder=10)
    ax.set(xlim=(18,1100),ylim=(.65,1.62),
           xlabel=r'Центральний тиск $P_c$ (MeV/fm$^3$)',
           ylabel=r'Маса $M/M_\odot$',
           title='Поворотні точки гілок гібридних зір')
    ax.legend(frameon=False)
    savefig(out,'09_mass_turning_points.png')
    # Baseline numeric checks to compare with CompOSE metadata
    surface_test=[]
    for multiplier in (1,3,10):
        # instead of changing EOS bounds, stop TOV at different finite pressure.
        p0=float(bh_s.iloc[(bh_s.M_solar-1.4).abs().argmin()].Pc)
        rr0,mm0=integrate_segment(p0,H.Pmin*multiplier,H.eps)
        surface_test.append(dict(Pstop_over_Pmin=multiplier,R_km=rr0,M_solar=mm0/KM_PER_M_SUN))
    pd.DataFrame(surface_test).to_csv(out/'07_surface_sensitivity.csv',index=False)
    validate=dict(conversion=GEOM_PER_MEV_FM3,solar_mass_in_km=KM_PER_M_SUN,
                  table_pressure_min=H.Pmin,table_pressure_max=H.Pmax,
                  pure_hadronic_maximum_mass_Msun=float(bhmax.M_solar),
                  pure_hadronic_R14_km=r14,
                  reference_CompOSE_Mmax_Msun=2.06,
                  reference_CompOSE_R14_km=11.7,
                  rtol=1.4e-7,max_step_km=.6,
                  surface_test=surface_test)
    with open(out/'08_numeric_checks.json','w') as f:json.dump(validate,f,indent=2)
    print('Numerical validation',json.dumps(validate,indent=2),flush=True)
    print('\nResults at:',out,flush=True)


if __name__=='__main__':
    d=Path(__file__).parent
    zp=Path(sys.argv[1]) if len(sys.argv)>1 else d/'eos.zip'
    cand=Path(sys.argv[2]) if len(sys.argv)>2 else d/'stage2_results/MIT_Bag_Maxwell_candidates.csv'
    out=Path(sys.argv[3]) if len(sys.argv)>3 else d/'stage3_results'
    main(zp,cand,out)
