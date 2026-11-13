"""Reproduce the phase transitions, equilibrium sequences, radial modes, and figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quarkmatter.model import (BAG_ROOTS_MEV, HadronicEOS, QuarkEOS,
                              find_maxwell_transition, load_compose, solve_star)
from quarkmatter.radial import RadialModeSolver, StellarBackground

RADIAL_PRESSURES = {
    0: [55, 120, 350, 650],
    175: [21, 23, 30, 90, 200, 450, 575, 700],
    185: [80, 81, 100, 140, 200, 370, 425, 600],
    190: [112, 113, 120, 180, 350],
}


def compute_phases(hadronic, directory):
    chemical_potentials = hadronic.table['chemical_potential']
    phase_rows = []
    for bag_root in BAG_ROOTS_MEV:
        quark = QuarkEOS(bag_root)
        transition = find_maxwell_transition(hadronic, quark)
        pressure_difference = quark.pressure(chemical_potentials) - hadronic.table['pressure']
        phase_rows.append({
            'bag_root_MeV': bag_root,
            'bag_constant_MeV_fm3': quark.bag_constant,
            'mu_at_zero_pressure_MeV': quark.zero_pressure_chemical_potential(),
            'phase_interpretation': ('Maxwell crossing' if transition else
                                     'quark favoured throughout table' if np.all(pressure_difference > 0)
                                     else 'no crossing in tabulated interval'),
            'transition_mu_MeV': transition['chemical_potential'] if transition else np.nan,
            'transition_pressure_MeV_fm3': transition['pressure'] if transition else np.nan,
            'energy_jump_MeV_fm3': transition['energy_jump'] if transition else np.nan,
            'seidov_ratio': transition['seidov_ratio'] if transition else np.nan,
            'seidov_threshold': transition['seidov_limit'] if transition else np.nan,
        })
    pd.DataFrame(phase_rows).to_csv(directory/'phase_transitions.csv', index=False)
    density = hadronic.table['density']
    pressure = hadronic.table['pressure']
    energy = hadronic.table['energy_density']
    mu = chemical_potentials
    pd.DataFrame({
        'density_fm3': density,
        'pressure_MeV_fm3': pressure,
        'energy_density_MeV_fm3': energy,
        'chemical_potential_MeV': mu,
        'euler_relative_error': (energy + pressure - mu*density)/(energy+pressure),
        'derivative_relative_error': (np.gradient(energy, density, edge_order=2)-mu)/mu,
    }).to_csv(directory/'thermodynamic_audit.csv', index=False)
    return phase_rows


def compute_equilibrium(hadronic, phase_rows, directory):
    pure_pressures = np.geomspace(1, hadronic.maximum_pressure, 95)
    pure_stars = [solve_star(float(pressure), hadronic) for pressure in pure_pressures]
    pd.DataFrame(pure_stars).to_csv(directory/'tov_hadronic.csv', index=False)
    hybrid_stars = []
    for row in phase_rows:
        if not np.isfinite(row['transition_pressure_MeV_fm3']):
            continue
        bag_root = row['bag_root_MeV']
        transition = find_maxwell_transition(hadronic, QuarkEOS(bag_root))
        pressures = np.unique(np.r_[
            transition['pressure']*(1 + np.array([1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.17, 0.25])),
            np.geomspace(transition['pressure']*1.001, 2100, 126),
        ])
        for pressure in pressures:
            try:
                star = solve_star(float(pressure), hadronic, QuarkEOS(bag_root), transition)
                hybrid_stars.append(dict(bag_root_MeV=bag_root, **star))
            except (ValueError, RuntimeError):
                continue
    pd.DataFrame(hybrid_stars).to_csv(directory/'tov_hybrid.csv', index=False)
    return pure_stars, hybrid_stars


def compute_radial_modes(hadronic, phase_rows, directory, radial_step):
    rows = []
    for bag_root, central_pressures in RADIAL_PRESSURES.items():
        quark = QuarkEOS(bag_root) if bag_root else None
        transition = find_maxwell_transition(hadronic, quark) if quark else None
        for central_pressure in central_pressures:
            if transition is not None and central_pressure <= transition['pressure']:
                continue
            try:
                background = StellarBackground(central_pressure, hadronic, quark, transition)
                solver = RadialModeSolver(background, radial_step=radial_step)
                conversion_options = ('rapid', 'slow') if quark else ('hadronic',)
                for conversion in conversion_options:
                    eigenvalues = solver.eigenvalues(conversion)
                    fundamental_squared = eigenvalues[0][0] if eigenvalues else np.nan
                    rows.append({
                        'bag_root_MeV': bag_root,
                        'central_pressure_MeV_fm3': central_pressure,
                        'mass_solar': background.mass_solar,
                        'radius_km': background.radius_km,
                        'core_radius_km': background.core_radius,
                        'conversion': conversion,
                        'fundamental_omega_squared_km2': fundamental_squared,
                        'nodes_in_lowest_found_mode': eigenvalues[0][1] if eigenvalues else np.nan,
                        'stability': ('unresolved' if not eigenvalues else
                                      'stable' if fundamental_squared > 0 else 'unstable'),
                    })
            except (ValueError, RuntimeError) as error:
                print(f'Radial model B^(1/4)={bag_root}, Pc={central_pressure}: {error}')
    pd.DataFrame(rows).to_csv(directory/'radial_modes.csv', index=False)
    return rows


def make_figures(hadronic, phases, pure_stars, hybrid_stars, radial_rows, directory):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
                         'savefig.dpi': 170})
    mu = hadronic.table['chemical_potential']
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(mu, hadronic.table['pressure'], label='RG(SLy4)', color='black', linewidth=2)
    for bag_root in BAG_ROOTS_MEV:
        quark = QuarkEOS(bag_root)
        axis.plot(mu, quark.pressure(mu), label=f'B^(1/4)={bag_root} MeV')
    axis.set(xlabel='Baryon chemical potential (MeV)', ylabel='Pressure (MeV/fm³)', title='Hadronic and quark equations of state')
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'pressure_vs_chemical_potential.png')
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot([star['radius_km'] for star in pure_stars],
              [star['mass_solar'] for star in pure_stars], color='black', label='RG(SLy4)')
    for bag_root in BAG_ROOTS_MEV:
        subset = [star for star in hybrid_stars if star['bag_root_MeV']==bag_root]
        if subset:
            axis.plot([star['radius_km'] for star in subset],
                      [star['mass_solar'] for star in subset], label=f'{bag_root} MeV')
    axis.set(xlabel='Radius (km)', ylabel='Mass (solar masses)', title='Equilibrium mass–radius sequences', xlim=(0, 50))
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory/'mass_radius.png')
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for bag_root in BAG_ROOTS_MEV:
        subset = [star for star in hybrid_stars if star['bag_root_MeV']==bag_root]
        if subset:
            axis.plot([star['mass_solar'] for star in subset],
                      [star['core_radius_km'] for star in subset], label=f'{bag_root} MeV')
    axis.set(xlabel='Mass (solar masses)', ylabel='Quark-core radius (km)', title='Equilibrium quark-core sizes')
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory/'quark_core_radius.png')
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for bag_root in (175, 185, 190):
        for conversion in ('rapid', 'slow'):
            subset = [row for row in radial_rows if row['bag_root_MeV'] == bag_root and row['conversion'] == conversion]
            if subset:
                axis.plot([row['central_pressure_MeV_fm3'] for row in subset],
                          [row['fundamental_omega_squared_km2'] for row in subset],
                          'o-', label=f'{bag_root} MeV, {conversion}')
    axis.axhline(0, color='black', linewidth=1)
    axis.set(xlabel='Central pressure (MeV/fm³)', ylabel='Lowest calculated omega² (km⁻²)',
             title='Selected radial modes; not a continuous stability map')
    if radial_rows:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'radial_modes.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eos', type=Path, default=Path('data/rg_sly4_compose.zip'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--radial-step', type=float, default=0.023)
    parser.add_argument('--skip-radial', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hadronic = HadronicEOS(load_compose(args.eos))
    phases = compute_phases(hadronic, args.output)
    pure_stars, hybrid_stars = compute_equilibrium(hadronic, phases, args.output)
    radial_rows = [] if args.skip_radial else compute_radial_modes(hadronic, phases, args.output, args.radial_step)
    make_figures(hadronic, phases, pure_stars, hybrid_stars, radial_rows, args.output)
    for row in phases:
        print(f"B^(1/4)={row['bag_root_MeV']} MeV: {row['phase_interpretation']}; transition pressure={row['transition_pressure_MeV_fm3']:.6g}")
    print(f'Wrote tables and figures to {args.output.resolve()}')


if __name__ == '__main__':
    main()
