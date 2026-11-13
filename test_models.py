import unittest
from pathlib import Path

import numpy as np

from quarkmatter.model import HadronicEOS, QuarkEOS, find_maxwell_transition, load_compose

DATA = Path(__file__).resolve().parents[1] / 'data' / 'rg_sly4_compose.zip'


class PhysicalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hadronic = HadronicEOS(load_compose(DATA))

    def test_hadronic_euler_identity(self):
        values = self.hadronic.table
        residual = (values['energy_density'] + values['pressure'] -
                    values['chemical_potential']*values['density'])
        self.assertLess(np.max(np.abs(residual/(values['energy_density']+values['pressure']))), 2e-4)

    def test_quark_thermodynamic_identity(self):
        quark = QuarkEOS(175)
        chemical_potential = np.linspace(1050, 1400, 15)
        pressure = quark.pressure(chemical_potential)
        energy = quark.energy_from_pressure(pressure)
        density = quark.baryon_density(chemical_potential)
        np.testing.assert_allclose(energy+pressure, chemical_potential*density, rtol=1e-12)

    def test_transition_regression(self):
        expected_pressures = {165: 0.005888, 175: 20.72, 185: 79.02, 190: 111.93}
        for root, expected in expected_pressures.items():
            transition = find_maxwell_transition(self.hadronic, QuarkEOS(root))
            self.assertIsNotNone(transition)
            self.assertAlmostEqual(transition['pressure'], expected, delta=max(0.0003, expected*0.001))

    def test_low_bag_constants_have_no_crossing(self):
        for root in (145, 155):
            self.assertIsNone(find_maxwell_transition(self.hadronic, QuarkEOS(root)))


if __name__ == '__main__':
    unittest.main()
