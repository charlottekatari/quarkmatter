#!/usr/bin/env python3
"""MAN research — Stage II: thermodynamic audit + massless uds MIT bag EoS.

Usage:
    python stage2_quark_phase.py /path/to/eos.zip /path/to/output_directory

Dependencies: numpy, scipy, pandas, matplotlib.
This script does not solve the TOV equations or infer stellar stability.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

HBARC = 197.3269804  # MeV fm
BROOT_VALUES = (145, 155, 165, 175, 185, 190)  # MeV
COEFF = 1.0 / (108.0 * np.pi**2 * HBARC**3)


def read_hadronic(path):
    with zipfile.ZipFile(path) as z:
        lines = z.read('eos.thermo').decode().splitlines()
        mass_n, mass_p, leptons = [float(x) for x in lines[0].split()]
        thermo = np.loadtxt(io.StringIO('\n'.join(lines[1:])))
        nb_lines = z.read('eos.nb').decode()
        nb_file = np.loadtxt(io.StringIO(nb_lines))
        i_min, i_max = int(nb_file[0]), int(nb_file[1])
        nB = nb_file[2:]
        t = np.loadtxt(io.StringIO(z.read('eos.t').decode()))
        yq = np.loadtxt(io.StringIO(z.read('eos.yq').decode()))
    if thermo.ndim != 2 or thermo.shape[1] < 11:
        raise ValueError('Unexpected CompOSE eos.thermo column count')
    if len(nB) != len(thermo) or len(nB) != i_max - i_min + 1:
        raise ValueError('Mismatch between eos.nb and eos.thermo lengths')
    if not np.array_equal(thermo[:, 1].astype(int), np.arange(i_min,i_max+1)):
        raise ValueError('eos.thermo baryon density indices not sequential')
    if not np.all(thermo[:, 10] == 0):
        raise ValueError('Unexpected extra columns; parser needs an update')
    if not (np.all(np.diff(nB)>0) and np.all(nB>0)):
        raise ValueError('Non-increasing or non-positive baryon density')
    if (int(t[0]), int(t[1])) != (0, 0) or abs(t[2]) > 1e-14:
        raise ValueError('This study assumes T=0 only')
    # eos.yq is a dummy one-value axis for this already beta-equilibrated EoS.
    Q1 = thermo[:,3]  # pressure / baryon number density (MeV)
    Q3 = thermo[:,5]  # scaled baryon chemical potential
    Q7 = thermo[:,9]  # scaled energy density per baryon
    P = nB*Q1
    mu = mass_n*(1+Q3)
    eps = nB*mass_n*(1+Q7)
    if not all(np.all(np.isfinite(x)) for x in (P,mu,eps)):
        raise ValueError('Non-finite thermodynamic values')
    if not all(np.all(np.diff(x)>0) for x in (P,mu,eps)):
        raise ValueError('Monotonicity check failed for hadronic EoS')
    return dict(nB=nB, P=P, mu=mu, eps=eps, mn=mass_n, mp=mass_p,
                leptons=int(leptons), yq=yq.tolist())


def bag_B(root):
    """B^(1/4) in MeV -> B in MeV/fm^3."""
    return root**4 / HBARC**3


def quark_pressure(mu, B):
    return COEFF*np.asarray(mu)**4 - B


def quark_density(mu):
    return 4*COEFF*np.asarray(mu)**3


def quark_energy(mu, B):
    return 3*COEFF*np.asarray(mu)**4 + B


def mu_zero(root):
    """Zero-pressure baryon chemical potential (MeV)."""
    return (bag_B(root)/COEFF)**0.25


def plot_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':11,
        'axes.titlesize':14, 'axes.titleweight':'semibold',
        'axes.labelsize':11, 'axes.spines.top':False,
        'axes.spines.right':False, 'legend.fontsize':9,
        'figure.dpi':120, 'savefig.dpi':190,
        'axes.grid':True, 'grid.alpha':0.23,
        'grid.linestyle':'--',
    })


def save_plot(directory, filename):
    plt.tight_layout()
    plt.savefig(directory / filename, bbox_inches='tight')
    plt.close()


def main(archive, outdir):
    directory=Path(outdir)
    directory.mkdir(parents=True,exist_ok=True)
    plot_style()
    h=read_hadronic(archive)
    nB, P_H, mu_H, eps_H = [h[x] for x in ('nB','P','mu','eps')]
    # Cold, charge-neutral, beta-equilibrated Euler identity.
    relative_euler = (eps_H + P_H - mu_H*nB)/(eps_H+P_H)
    mu_dedn = np.gradient(eps_H, nB, edge_order=2)
    relative_grad_eps = (mu_dedn-mu_H)/mu_H
    nB_dPdmu = np.gradient(P_H, mu_H, edge_order=2)
    relative_grad_p = (nB_dPdmu-nB)/nB
    checks={
        'n_points':len(nB),
        'max_abs_euler_relative':float(np.max(abs(relative_euler))),
        'median_abs_euler_relative':float(np.median(abs(relative_euler))),
        'max_abs_dedn_relative':float(np.max(abs(relative_grad_eps))),
        'median_abs_dedn_relative':float(np.median(abs(relative_grad_eps))),
        'max_abs_dPdmu_relative':float(np.max(abs(relative_grad_p))),
        'median_abs_dPdmu_relative':float(np.median(abs(relative_grad_p))),
        'hadronic_mu_interval_MeV':[float(mu_H[0]),float(mu_H[-1])],
        'note': 'Numerical derivatives of a finite grid are sensitive to endpoints and local changes in slope.'
    }
    pd.DataFrame({
        'nB_H_fm^-3':nB, 'muB_H_MeV':mu_H,
        'P_H_MeV_fm^-3':P_H,'epsilon_H_MeV_fm^-3':eps_H,
        'euler_relative_error':relative_euler,
        'muB_from_numeric_deps_dn_MeV':mu_dedn,
        'deps_dn_relative_error':relative_grad_eps,
        'nB_from_numeric_dP_dmu_fm^-3':nB_dPdmu,
        'dP_dmu_relative_error':relative_grad_p
    }).to_csv(directory/'SLy4_thermodynamic_audit.csv',index=False)

    pchip_p = PchipInterpolator(mu_H, P_H, extrapolate=False)
    pchip_nb = PchipInterpolator(mu_H, nB, extrapolate=False)
    pchip_eps = PchipInterpolator(mu_H, eps_H, extrapolate=False)
    parameters=[]; phase_rows=[]; long_quark=[]
    for root in BROOT_VALUES:
        B=bag_B(root)
        mu0=mu_zero(root)
        p_at_min=float(quark_pressure(mu_H[0],B))
        p_at_max=float(quark_pressure(mu_H[-1],B))
        # Physical bulk quark branch from P=0 upward. No negative-pressure results exported.
        mus=np.linspace(mu0, max(mu0+50,mu_H[-1]), 1301)
        ps=np.maximum(quark_pressure(mus,B), 0.0)  # clamp float-roundoff at P=0
        epss=quark_energy(mus,B)
        ns=quark_density(mus)
        # Strict exact consistency in arithmetic and finite-difference check.
        exact_err=np.max(abs((mus*ns-epss-ps)/(mus*ns)))
        numerical_err=np.median(abs((np.gradient(ps,mus)-ns)/ns)[2:-2])
        if exact_err > 1e-12:
            raise AssertionError(f'Quark Euler identity failed for Broot={root}')
        parameters.append({
            'B_root_MeV':root, 'B_MeV_fm^-3':B,
            'muB_at_P_zero_MeV':mu0,
            'epsilon_at_P_zero_MeV_fm^-3':4*B,
            'nB_at_P_zero_fm^-3':float(quark_density(mu0)),
            'cs_squared_over_c_squared':1/3,
            'max_relative_euler_identity_error':float(exact_err),
            'median_relative_numerical_dP_dmu_error':float(numerical_err),
        })
        long_quark.append(pd.DataFrame({
            'B_root_MeV':root,'B_MeV_fm^-3':B,'muB_MeV':mus,
            'P_MeV_fm^-3':ps, 'epsilon_MeV_fm^-3':epss,
            'nB_fm^-3':ns,
            'cs_squared_over_c_squared':np.full_like(mus,1/3)
        }))
        # All sign crossings searched in actual bounded tabulated mu range.
        delta=quark_pressure(mu_H,B)-P_H
        roots=[]
        for i in range(len(mu_H)-1):
            if delta[i] == 0:
                roots.append(float(mu_H[i]))
            elif delta[i]*delta[i+1] < 0:
                roots.append(float(brentq(
                    lambda x:float(quark_pressure(x,B)-pchip_p(x)),
                    float(mu_H[i]),float(mu_H[i+1]),xtol=1e-10)))
        if not roots:
            phase_rows.append({
                'B_root_MeV':root, 'B_MeV_fm^-3':B,
                'transition_in_table':False,
                'status':('Quark pressure exceeds hadronic pressure at all tabulated muB'
                          if np.all(delta>0) else
                          'Hadronic pressure exceeds quark pressure at all tabulated muB'
                          if np.all(delta<0) else
                          'No crossing detected; check potential tangent crossing'),
                'muB_transition_MeV':np.nan,'P_transition_MeV_fm^-3':np.nan,
                'nB_hadronic_fm^-3':np.nan, 'nB_quark_fm^-3':np.nan,
                'epsilon_hadronic_MeV_fm^-3':np.nan,
                'epsilon_quark_MeV_fm^-3':np.nan,
                'delta_epsilon_MeV_fm^-3':np.nan,
                'delta_epsilon_over_epsilon_H':np.nan,
            })
        else:
            for cross in roots:
                p=float(pchip_p(cross))
                nh=float(pchip_nb(cross))
                nq=float(quark_density(cross))
                eh=float(pchip_eps(cross))
                eq=float(quark_energy(cross,B))
                phase_rows.append({
                    'B_root_MeV':root,'B_MeV_fm^-3':B,
                    'transition_in_table':True,
                    'status': 'Candidate hadron-to-quark Maxwell crossing (NOT a stellar stability test)',
                    'muB_transition_MeV':cross,'P_transition_MeV_fm^-3':p,
                    'nB_hadronic_fm^-3':nh,'nB_quark_fm^-3':nq,
                    'epsilon_hadronic_MeV_fm^-3':eh,
                    'epsilon_quark_MeV_fm^-3':eq,
                    'delta_epsilon_MeV_fm^-3':eq-eh,
                    'delta_epsilon_over_epsilon_H':(eq-eh)/eh,
                })
    parameters=pd.DataFrame(parameters)
    phase=pd.DataFrame(phase_rows)
    quark=pd.concat(long_quark,ignore_index=True)
    parameters.to_csv(directory/'MIT_Bag_parameters.csv',index=False)
    phase.to_csv(directory/'MIT_Bag_Maxwell_candidates.csv',index=False)
    quark.to_csv(directory/'MIT_Bag_quark_EoS.csv',index=False)

    colors=plt.get_cmap('viridis')(np.linspace(.1,.9,len(BROOT_VALUES)))
    # 1. Multi-B pressure(muB) plot.
    fig,ax=plt.subplots(figsize=(10,6.2))
    mu_draw=np.linspace(825,1550,1200)
    for root,col in zip(BROOT_VALUES,colors):
        ax.plot(mu_draw,quark_pressure(mu_draw,bag_B(root)),label=fr'$B^{{1/4}}={root}\,\mathrm{{MeV}}$',color=col,lw=2.1)
    ax.axhline(0,color='black',lw=.9,ls=':')
    ax.set(xlabel=r'Баріонний хімічний потенціал $\mu_B$ (MeV)',
           ylabel=r'Тиск $P_Q$ (MeV/fm$^3$)',
           title='Кваркова матерія: тиск за моделлю MIT Bag',
           xlim=(825,1550),ylim=(-90,530))
    ax.legend(loc='upper left',ncol=2,frameon=False)
    save_plot(directory,'01_MIT_pressure_vs_mu.png')

    # 2. Quark energy density versus pressure, domain P>=0.
    fig,ax=plt.subplots(figsize=(9.2,6.0))
    press=np.linspace(0,410,450)
    for root,col in zip(BROOT_VALUES,colors):
        B=bag_B(root)
        ax.plot(3*press+4*B,press,label=fr'{root} MeV',color=col,lw=2)
    ax.set(xlabel=r'Густина енергії $\varepsilon_Q$ (MeV/fm$^3$)',
           ylabel=r'Тиск $P_Q$ (MeV/fm$^3$)',
           title=r'Кваркове рівняння стану: $P_Q(\varepsilon_Q)$')
    ax.legend(title=r'$B^{1/4}$',ncol=2,frameon=False)
    save_plot(directory,'02_MIT_pressure_vs_energy.png')

    # 3. Quark density versus mu, B affects only minimum positive P threshold.
    fig,ax=plt.subplots(figsize=(9.2,6.0))
    for root,col in zip(BROOT_VALUES,colors):
        mu0=mu_zero(root)
        mus=np.linspace(mu0,1500,420)
        ax.plot(mus,quark_density(mus),color=col,lw=2,label=f'{root} MeV')
        ax.scatter([mu0],[float(quark_density(mu0))],color=[col],s=20,zorder=5)
    ax.set(xlabel=r'Баріонний хімічний потенціал $\mu_B$ (MeV)',
           ylabel=r'Баріонна густина $n_{B,Q}$ (fm$^{-3}$)',
           title='Кваркова матерія: баріонна густина та поріг нульового тиску')
    ax.legend(title=r'$B^{1/4}$',ncol=2,frameon=False)
    save_plot(directory,'03_MIT_baryon_density_vs_mu.png')

    # 4. Comparison on a plot chosen to include roots clearly.
    fig,ax=plt.subplots(figsize=(10.0,6.3))
    mu_plot=np.linspace(float(mu_H[0]),1600,1200)
    ax.plot(mu_plot,pchip_p(mu_plot),color='#161b2c',lw=3.0,
            label='Адронна SLy4 (CompOSE)')
    for root,col in zip(BROOT_VALUES,colors):
        ps=quark_pressure(mu_plot,bag_B(root))
        ax.plot(mu_plot,ps,color=col,lw=1.7,alpha=.95,label=f'MIT {root} MeV')
    selected=phase[phase['transition_in_table'] == True]
    for _,row in selected.iterrows():
        ax.scatter([row['muB_transition_MeV']],[row['P_transition_MeV_fm^-3']],
                   color='#e34b42',s=45,zorder=9,edgecolor='white',linewidth=.6)
    ax.axhline(0,color='#888888',lw=.8,ls=':')
    ax.set(xlim=(float(mu_H[0]),1500),ylim=(-85,405),
           xlabel=r'Баріонний хімічний потенціал $\mu_B$ (MeV)',
           ylabel=r'Тиск $P$ (MeV/fm$^3$)',
           title='SLy4 та MIT Bag: порівняння тисків і перетини Максвелла')
    ax.legend(loc='upper left',ncol=2,frameon=False)
    save_plot(directory,'04_SLy4_vs_MIT_Maxwell.png')

    # 5. Thermodynamic audit visual.
    fig,ax=plt.subplots(figsize=(10.0,5.7))
    ax.semilogx(nB,abs(relative_euler)*100,lw=1.9,
                label=r'$|\varepsilon+P-\mu_B n_B|/(\varepsilon+P)$')
    ax.semilogx(nB,abs(relative_grad_eps)*100,lw=1.4,alpha=.85,
                label=r'$|d\varepsilon/dn_B-\mu_B|/\mu_B$')
    ax.set(xlabel=r'Баріонна густина $n_B$ (fm$^{-3}$)',
           ylabel='Відносне відхилення (%)',
           title='Термодинамічна перевірка адронної EoS SLy4')
    ax.legend(frameon=False)
    save_plot(directory,'05_SLy4_thermodynamic_checks.png')

    # 6. Maxwell pressures / energy jump vs parameter, only four crossings.
    fig,ax=plt.subplots(figsize=(9.5,5.8))
    r=selected['B_root_MeV'].to_numpy(dtype=float)
    ax.plot(r,selected['P_transition_MeV_fm^-3'],marker='o',lw=2.3,label=r'$P_t$')
    ax.set(xlabel=r'Параметр $B^{1/4}$ (MeV)',ylabel=r'$P_t$ (MeV/fm$^3$)',
           title='Кандидати переходу Максвелла: залежність від параметра B')
    ax.set_xticks(BROOT_VALUES)
    ax2=ax.twinx()
    ax2.plot(r,selected['delta_epsilon_MeV_fm^-3'],ls='--',marker='s',lw=1.8,
             color='#b66b4a',label=r'$\Delta\varepsilon$')
    ax2.set_ylabel(r'$\Delta\varepsilon$ (MeV/fm$^3$)')
    h1,l1=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,loc='upper left',frameon=False)
    save_plot(directory,'06_Maxwell_pressure_jump_vs_B.png')

    # Plain-text report for reproducibility.
    with open(directory/'calculation_summary.txt','w',encoding='utf-8') as f:
        f.write('MAN — SLy4 and massless u,d,s MIT Bag thermodynamic exploration\n')
        f.write('All units: MeV, fm. Broot=B^(1/4) in MeV.\n')
        f.write('Model: T=0, 3 massless quark flavors, no interactions, constant B, beta equilibrium, local neutrality.\n')
        f.write('P_Q=muB^4/[108 pi^2 (hbar c)^3]-B; epsilon_Q=3 P_Q+4 B; nB_Q=dP_Q/dmuB.\n')
        f.write('No TOV or stellar stability calculations have been performed.\n\n')
        f.write('Hadronic audit\n')
        f.write(json.dumps(checks,indent=2,ensure_ascii=False)+'\n\n')
        f.write('Quark parameters\n'+parameters.round(8).to_string(index=False)+'\n\n')
        f.write('Maxwell crossings\n'+phase.round(6).to_string(index=False)+'\n')
        f.write('\nNo crossing for 145 and 155 MeV within the tabulated hadronic muB interval.\n')
        f.write('For these values quark matter has higher pressure than SLy4 at all tested muB.\n')
        f.write('The graph may show negative-pressure mathematical continuations: physical quark CSV restricts P>=0.\n')
        f.write('Candidate crossings must NOT be interpreted as proof of stable hybrid-star configurations.\n')
    metadata={'hbar_c_MeV_fm':HBARC,'B_root_values_MeV':BROOT_VALUES,
              'eos_model':'RG(SLy4), CompOSE #134','model_approximations':
              'cold massless u,d,s free quarks, no finite strange mass, no pQCD corrections, constant B',
              'checks':checks}
    with open(directory/'metadata.json','w',encoding='utf-8') as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False)

    print('THERMODYNAMIC VALIDATION:',json.dumps(checks,ensure_ascii=False,indent=2))
    print('\nQUARK PARAMETERS:')
    print(parameters[['B_root_MeV','B_MeV_fm^-3','muB_at_P_zero_MeV','nB_at_P_zero_fm^-3']].round(4).to_string(index=False))
    print('\nMAXWELL CANDIDATES:')
    print(phase[['B_root_MeV','transition_in_table','muB_transition_MeV',
                 'P_transition_MeV_fm^-3','delta_epsilon_MeV_fm^-3']].round(5).to_string(index=False))
    print('\nOUTPUT:',directory)


if __name__ == '__main__':
    input_zip=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('eos.zip')
    output=Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).with_name('stage2_results')
    main(input_zip,output)
