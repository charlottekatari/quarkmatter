"""Relativistic radial modes with rapid/slow conversion at a Maxwell interface."""
import math

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .model import PRESSURE_TO_GEOMETRIC, SOLAR_MASS_IN_KM


class StellarBackground:
    def __init__(self, central_pressure, hadronic, quark=None, transition=None,
                 background_step=0.12):
        self.central_pressure = central_pressure
        self.hadronic = hadronic
        self.quark = quark
        self.transition = transition
        self.layers = []
        self.core_radius = 0.0
        core_exists = quark is not None and transition is not None and central_pressure > transition['pressure']
        initial_phase = 'quark' if core_exists else 'hadronic'
        initial_radius = 1e-4
        central_pressure_geom = central_pressure * PRESSURE_TO_GEOMETRIC
        central_energy, _ = self.thermodynamics(central_pressure_geom, initial_phase)
        initial_mass = 4*np.pi*central_energy*initial_radius**3/3
        corrected_pressure = central_pressure_geom - 2*np.pi/3 * (central_energy+central_pressure_geom) * (central_energy+3*central_pressure_geom) * initial_radius**2
        initial_state = [initial_mass, corrected_pressure, 0]

        def integrate_layer(phase, radius_start, state_start, target_pressure):
            def equations(radius, state):
                mass, pressure, metric_potential = state
                energy, _ = self.thermodynamics(pressure, phase)
                pressure_derivative = -(energy+pressure)*(mass+4*np.pi*radius**3*pressure)/(radius*(radius-2*mass))
                return [4*np.pi*radius**2*energy, pressure_derivative, -pressure_derivative/(energy+pressure)]
            def event(radius, state):
                return state[1]-target_pressure*PRESSURE_TO_GEOMETRIC
            event.direction = -1
            event.terminal = True
            solution = solve_ivp(equations, (radius_start, 1500), state_start,
                                 events=event, dense_output=True, method='DOP853',
                                 max_step=background_step, rtol=3e-9,
                                 atol=[1e-12, 3e-18, 1e-12])
            if not solution.success or len(solution.t_events[0]) != 1:
                raise RuntimeError('Could not integrate the stellar background')
            end_radius = float(solution.t_events[0][0])
            end_state = solution.y_events[0][0]
            self.layers.append((phase, radius_start, end_radius, solution.sol))
            return end_radius, end_state

        if core_exists:
            self.core_radius, interface_state = integrate_layer(
                'quark', initial_radius, initial_state, transition['pressure'])
            surface_radius, surface_state = integrate_layer(
                'hadronic', self.core_radius, interface_state, hadronic.minimum_pressure)
        else:
            surface_radius, surface_state = integrate_layer(
                'hadronic', initial_radius, initial_state, hadronic.minimum_pressure)
        self.radius_km = surface_radius
        self.mass_solar = float(surface_state[0] / SOLAR_MASS_IN_KM)
        self.metric_shift = 0.5*np.log(1-2*surface_state[0]/surface_radius)-surface_state[2]

    def thermodynamics(self, pressure_geom, phase):
        pressure = float(pressure_geom) / PRESSURE_TO_GEOMETRIC
        if phase == 'quark':
            energy_geom = self.quark.energy_from_pressure(pressure) * PRESSURE_TO_GEOMETRIC
            adiabatic_pressure = (energy_geom + pressure_geom)/3
        else:
            pressure = float(np.clip(pressure, self.hadronic.minimum_pressure,
                                     self.hadronic.maximum_pressure))
            energy = self.hadronic.energy(pressure)
            energy_geom = energy * PRESSURE_TO_GEOMETRIC
            adiabatic_pressure = (energy_geom + pressure * PRESSURE_TO_GEOMETRIC) * self.hadronic.sound_speed_squared(pressure)
        return energy_geom, adiabatic_pressure

    def at_radius(self, radius, layer_index):
        phase, _, _, interpolate = self.layers[layer_index]
        mass, pressure, metric_potential = interpolate(radius)
        energy, adiabatic_pressure = self.thermodynamics(pressure, phase)
        radial_metric = 1/(1-2*mass/radius)
        pressure_gradient = -(energy+pressure)*(mass+4*np.pi*radius**3*pressure)/(radius*(radius-2*mass))
        return pressure, metric_potential+self.metric_shift, energy, adiabatic_pressure, radial_metric, pressure_gradient


@njit(cache=True)
def _propagate_mode(radius_grid, coefficients, frequency_squared, displacement, pressure_perturbation):
    node_count = 0
    for index in range(len(radius_grid)-1):
        dr = radius_grid[index+1]-radius_grid[index]
        left = coefficients[index]
        right = coefficients[index+1]
        middle = 0.5*(left+right)
        a1 = left[0]*displacement + left[1]*pressure_perturbation
        b1 = (left[3]+frequency_squared*left[4])*displacement + left[2]*pressure_perturbation
        x2 = displacement+dr*a1/2
        q2 = pressure_perturbation+dr*b1/2
        a2 = middle[0]*x2+middle[1]*q2
        b2 = (middle[3]+frequency_squared*middle[4])*x2+middle[2]*q2
        x3 = displacement+dr*a2/2
        q3 = pressure_perturbation+dr*b2/2
        a3 = middle[0]*x3+middle[1]*q3
        b3 = (middle[3]+frequency_squared*middle[4])*x3+middle[2]*q3
        x4 = displacement+dr*a3
        q4 = pressure_perturbation+dr*b3
        a4 = right[0]*x4+right[1]*q4
        b4 = (right[3]+frequency_squared*right[4])*x4+right[2]*q4
        new_displacement = displacement+dr*(a1+2*a2+2*a3+a4)/6
        pressure_perturbation += dr*(b1+2*b2+2*b3+b4)/6
        if new_displacement*displacement < 0:
            node_count += 1
        displacement = new_displacement
        if abs(displacement)>1e150 or abs(pressure_perturbation)>1e150:
            return np.nan, node_count, displacement
    return pressure_perturbation, node_count, displacement


class RadialModeSolver:
    def __init__(self, background, radial_step=0.023, surface_pressure_factor=10):
        self.background = background
        self.pressure_scale = background.central_pressure * PRESSURE_TO_GEOMETRIC
        initial_phase = background.layers[0][0]
        _, central_gamma_pressure = background.thermodynamics(self.pressure_scale, initial_phase)
        self.initial_pressure_perturbation = -3*central_gamma_pressure/self.pressure_scale
        self.layer_coefficients = []
        for index, (phase, start, end, interpolation) in enumerate(background.layers):
            if index == len(background.layers)-1 and surface_pressure_factor > 1:
                target = surface_pressure_factor*background.hadronic.minimum_pressure*PRESSURE_TO_GEOMETRIC
                end = brentq(lambda radius: interpolation(radius)[1]-target, start, end)
            radii = np.linspace(start, end, max(10, math.ceil((end-start)/radial_step)+1))
            coefficients = np.empty((len(radii), 5))
            for grid_index, radius in enumerate(radii):
                pressure, metric, energy, gamma_pressure, radial_metric, gradient = background.at_radius(radius, index)
                coefficients[grid_index] = [
                    -3/radius-gradient/(energy+pressure),
                    -self.pressure_scale/(gamma_pressure*radius),
                    gradient/(energy+pressure)-4*np.pi*(energy+pressure)*radius*radial_metric,
                    (-4*gradient-8*np.pi*radial_metric*(energy+pressure)*pressure*radius+radius*gradient**2/(energy+pressure))/self.pressure_scale,
                    radial_metric*np.exp(-2*metric)*(energy+pressure)*radius/self.pressure_scale,
                ]
            self.layer_coefficients.append((radii, coefficients))
        self.interface_jump_factor = 0.0
        if background.core_radius > 0:
            radius = background.core_radius
            pressure_gradient_inside = background.at_radius(radius, 0)[5]
            pressure_gradient_outside = background.at_radius(radius, 1)[5]
            self.interface_jump_factor = self.pressure_scale/radius * (
                1/pressure_gradient_outside - 1/pressure_gradient_inside)

    def surface_residual(self, frequency_squared, conversion='rapid', count_nodes=False):
        displacement = 1.0
        pressure_perturbation = self.initial_pressure_perturbation
        nodes = 0
        for index, (radii, coefficients) in enumerate(self.layer_coefficients):
            pressure_perturbation, layer_nodes, displacement = _propagate_mode(
                radii, coefficients, frequency_squared, displacement, pressure_perturbation)
            nodes += layer_nodes
            if not np.isfinite(pressure_perturbation):
                return (np.nan, nodes) if count_nodes else np.nan
            if index == 0 and len(self.layer_coefficients) == 2 and conversion == 'rapid':
                displacement += pressure_perturbation * self.interface_jump_factor
        return (pressure_perturbation, nodes) if count_nodes else pressure_perturbation

    def eigenvalues(self, conversion='rapid', scan=None):
        if scan is None:
            scan = np.unique(np.r_[np.linspace(-0.012, -0.001, 19),
                                    np.linspace(-0.001, 0.001, 31),
                                    np.linspace(0.001, 0.024, 30)])
        residuals = [self.surface_residual(float(value), conversion) for value in scan]
        eigenvalues = []
        for index in range(len(scan)-1):
            if np.isfinite(residuals[index]) and np.isfinite(residuals[index+1]) and residuals[index]*residuals[index+1] < 0:
                root = brentq(lambda x: self.surface_residual(x, conversion),
                              scan[index], scan[index+1], xtol=1e-10)
                _, nodes = self.surface_residual(root, conversion, count_nodes=True)
                eigenvalues.append((float(root), int(nodes)))
        return eigenvalues
