"""Cold RG(SLy4) and massless uds MIT-bag equations of state; TOV solutions."""
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

HBAR_C = 197.3269804
GRAVITATIONAL_CONSTANT = 6.67430e-11
SPEED_OF_LIGHT = 299792458.0
SOLAR_MASS_KG = 1.98847e30
MEV_IN_JOULES = 1.602176634e-13
PRESSURE_TO_GEOMETRIC = GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**4 * MEV_IN_JOULES / 1e-45 * 1e6
SOLAR_MASS_IN_KM = GRAVITATIONAL_CONSTANT * SOLAR_MASS_KG / SPEED_OF_LIGHT**2 / 1000
BAG_ROOTS_MEV = (145, 155, 165, 175, 185, 190)
QUARK_COEFFICIENT = 1 / (108 * np.pi**2 * HBAR_C**3)


def load_compose(archive_path):
    """Read the cold, beta-equilibrated CompOSE RG(SLy4) tables."""
    with ZipFile(archive_path) as archive:
        density_axis = np.loadtxt(StringIO(archive.read('eos.nb').decode()))
        lines = archive.read('eos.thermo').decode().splitlines()
        neutron_mass_mev = float(lines[0].split()[0])
        thermodynamics = np.loadtxt(StringIO('\n'.join(lines[1:])))
    baryon_density = density_axis[2:]
    if len(baryon_density) != len(thermodynamics):
        raise ValueError('CompOSE density grid does not match thermodynamics')
    pressure = baryon_density * thermodynamics[:, 3]
    chemical_potential = neutron_mass_mev * (1 + thermodynamics[:, 5])
    energy_density = baryon_density * neutron_mass_mev * (1 + thermodynamics[:, 9])
    for values in (baryon_density, pressure, chemical_potential, energy_density):
        if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
            raise ValueError('CompOSE data must be finite and strictly increasing')
    return dict(density=baryon_density, pressure=pressure,
                chemical_potential=chemical_potential, energy_density=energy_density)


class HadronicEOS:
    def __init__(self, table):
        self.table = table
        self.minimum_pressure = float(table['pressure'][0])
        self.maximum_pressure = float(table['pressure'][-1])
        self.pressure_of_mu = PchipInterpolator(
            table['chemical_potential'], table['pressure'], extrapolate=False)
        self.density_of_mu = PchipInterpolator(
            table['chemical_potential'], table['density'], extrapolate=False)
        self.energy_of_mu = PchipInterpolator(
            table['chemical_potential'], table['energy_density'], extrapolate=False)
        self.log_energy_of_log_pressure = PchipInterpolator(
            np.log(table['pressure']), np.log(table['energy_density']), extrapolate=False)
        self.log_energy_derivative = self.log_energy_of_log_pressure.derivative()

    def energy(self, pressure):
        limited_pressure = np.clip(pressure, self.minimum_pressure, self.maximum_pressure)
        return float(np.exp(self.log_energy_of_log_pressure(np.log(limited_pressure))))

    def sound_speed_squared(self, pressure):
        limited_pressure = np.clip(pressure, self.minimum_pressure, self.maximum_pressure)
        energy = self.energy(limited_pressure)
        slope = float(self.log_energy_derivative(np.log(limited_pressure)))
        return limited_pressure / (energy * slope)


class QuarkEOS:
    def __init__(self, bag_root_mev):
        self.bag_root_mev = bag_root_mev
        self.bag_constant = bag_root_mev**4 / HBAR_C**3

    def pressure(self, chemical_potential):
        return QUARK_COEFFICIENT * np.asarray(chemical_potential)**4 - self.bag_constant

    def baryon_density(self, chemical_potential):
        return 4 * QUARK_COEFFICIENT * np.asarray(chemical_potential)**3

    def energy_from_pressure(self, pressure):
        return 3 * pressure + 4 * self.bag_constant

    def zero_pressure_chemical_potential(self):
        return (self.bag_constant / QUARK_COEFFICIENT)**0.25


def find_maxwell_transition(hadronic, quark):
    """Return the hadron-to-quark crossing inside the available CompOSE interval."""
    chemical_potentials = hadronic.table['chemical_potential']
    pressure_difference = quark.pressure(chemical_potentials) - hadronic.table['pressure']
    crossings = []
    for index in range(len(chemical_potentials) - 1):
        if pressure_difference[index] * pressure_difference[index + 1] < 0:
            root = brentq(lambda mu: quark.pressure(mu) - hadronic.pressure_of_mu(mu),
                          chemical_potentials[index], chemical_potentials[index + 1])
            crossings.append(float(root))
    if not crossings:
        return None
    transition_mu = crossings[0]
    transition_pressure = float(hadronic.pressure_of_mu(transition_mu))
    hadronic_energy = float(hadronic.energy_of_mu(transition_mu))
    quark_energy = quark.energy_from_pressure(transition_pressure)
    return dict(chemical_potential=transition_mu, pressure=transition_pressure,
                energy_jump=quark_energy-hadronic_energy,
                hadronic_energy=hadronic_energy,
                seidov_limit=0.5 + 1.5 * transition_pressure / hadronic_energy,
                seidov_ratio=(quark_energy-hadronic_energy)/hadronic_energy)


def _tov_derivative(radius, state, energy_function):
    mass_km, pressure_geom = state
    if radius <= 2 * mass_km:
        raise ValueError('TOV solution crossed the Schwarzschild radius')
    energy_geom = PRESSURE_TO_GEOMETRIC * energy_function(
        max(float(pressure_geom), 0) / PRESSURE_TO_GEOMETRIC)
    mass_gradient = 4 * np.pi * radius**2 * energy_geom
    pressure_gradient = -(energy_geom + pressure_geom) * (mass_km + 4*np.pi*radius**3*pressure_geom) / (radius * (radius - 2*mass_km))
    return mass_gradient, pressure_gradient


def _integrate_layer(initial_pressure, final_pressure, energy_function, initial_state=None):
    if initial_pressure <= final_pressure:
        raise ValueError('The pressure must decrease through each layer')
    if initial_state is None:
        start_radius = 1e-5
        central_pressure = initial_pressure * PRESSURE_TO_GEOMETRIC
        central_energy = energy_function(initial_pressure) * PRESSURE_TO_GEOMETRIC
        start_mass = 4*np.pi*central_energy*start_radius**3/3
        corrected_pressure = central_pressure - 2*np.pi/3 * (central_energy+central_pressure) * (central_energy+3*central_pressure) * start_radius**2
    else:
        start_radius, start_mass = initial_state
        corrected_pressure = initial_pressure * PRESSURE_TO_GEOMETRIC
    def surface_event(radius, state):
        return state[1] - final_pressure * PRESSURE_TO_GEOMETRIC
    surface_event.terminal = True
    surface_event.direction = -1
    solution = solve_ivp(lambda r, state: _tov_derivative(r, state, energy_function),
                         (start_radius, 2000), [start_mass, corrected_pressure],
                         method='DOP853', rtol=1.4e-7, atol=[5e-11, 7e-18],
                         max_step=0.6, events=surface_event)
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError('TOV integration did not reach the target pressure')
    return float(solution.t_events[0][0]), float(solution.y_events[0][0][0])


def solve_star(central_pressure, hadronic, quark=None, transition=None):
    """Return an equilibrium TOV configuration; dynamic stability is separate."""
    if transition is None or quark is None or central_pressure <= transition['pressure']:
        radius, mass = _integrate_layer(central_pressure, hadronic.minimum_pressure, hadronic.energy)
        core_radius = 0.0
    else:
        pressure_at_boundary = transition['pressure']
        core_radius, core_mass = _integrate_layer(
            central_pressure, pressure_at_boundary, quark.energy_from_pressure)
        radius, mass = _integrate_layer(
            pressure_at_boundary, hadronic.minimum_pressure, hadronic.energy,
            initial_state=(core_radius, core_mass))
    return dict(central_pressure=central_pressure, mass_solar=mass/SOLAR_MASS_IN_KM,
                radius_km=radius, core_radius_km=core_radius)
