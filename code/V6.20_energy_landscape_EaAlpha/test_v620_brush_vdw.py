"""Tests for the V6.20 brush-extended van der Waals module.

Run with

    python -m unittest discover -s <this directory> -p "test_v620_brush_vdw.py"

The first test class pins the port against the archived V6.20 notebook run
(``_v620cells/run_v620_newref.stdout.log``), so a change that silently moves
the legacy Hamiltonian fails here.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.constants import Boltzmann

import v620_brush_vdw as bv


LEGACY = bv.BrushPhys(brush_model="off", voxel_count=4)
REF = bv.REFERENCE


class TestLegacyReproduction(unittest.TestCase):
    """Numbers quoted in the archived V6.20 run log, to four decimals."""

    @classmethod
    def setUpClass(cls):
        cls.model = bv.BrushRhombohedralSL(REF["L_nm"], LEGACY)
        cls.gap = cls.model.nearest_exact_gap(
            REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
        )
        cls.components = cls.model.energy_components(
            REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
        )

    def test_reference_gap(self):
        self.assertAlmostEqual(self.gap, 3.377736, places=6)

    def test_reference_components(self):
        for key, expected in [
            ("Dipole", -3.5563),
            ("VdW_face6", -2.0838),
            ("VdW_full26", -2.7991),
            ("VdW_tail20", -0.7153),
            ("Zeeman", -141.7936),
            ("Anisotropy", -6.6336),
        ]:
            self.assertAlmostEqual(
                self.components[key], expected, places=4, msg=key
            )

    def test_simple_cubic_check(self):
        a_sc = REF["L_nm"] + self.gap
        components = self.model.energy_components(a_sc, 90.0, 0.0)
        self.assertAlmostEqual(components["VdW_face6"], -2.8707, places=4)
        self.assertAlmostEqual(components["Dipole"], 0.0, places=10)

    def test_no_steric_when_brush_is_off(self):
        self.assertEqual(self.components["Steric"], 0.0)
        self.assertAlmostEqual(
            self.components["Interaction"],
            self.components["Interaction_legacy"],
            places=12,
        )

    def test_sharp_bodies_do_not_overlap_at_reference(self):
        self.assertGreater(
            self.model.nearest_sharp_gap(
                REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
            ),
            0.0,
        )


class TestVoxelSumIdentity(unittest.TestCase):
    """The difference-lattice sum is the V6.20 double sum, exactly."""

    def test_matches_double_sum(self):
        for count in (2, 3, 4, 5):
            model = bv.BrushRhombohedralSL(
                16.0, LEGACY.with_voxels(count)
            )
            offsets = model.neighbour_vectors(20.31, 74.0) @ model.orientation(
                -15.0
            )
            for index in (0, 6, 12, 25):
                fast = model.vdw_pair_energy_voxel_J(offsets[index])
                slow = model.vdw_pair_energy_double_sum_J(offsets[index])
                self.assertAlmostEqual(fast / slow, 1.0, places=12)

    def test_weights_sum_to_square_of_voxel_number(self):
        for count in (3, 6):
            _, weights, _, number = bv.voxel_difference_weights(
                count, edge_m=16e-9
            )
            self.assertAlmostEqual(weights.sum() / number**2, 1.0, places=12)

    def test_rounded_body_has_fewer_voxels(self):
        """The mask must cut corners once the grid resolves them.

        At L = 16 nm and r = 1.5 nm the rounded cube is 98.4 % of the
        enclosing cube by volume, and no voxel centre falls outside it until
        the grid is fine enough to place one in a corner.
        """
        _, _, _, sharp = bv.voxel_difference_weights(
            24, edge_m=16e-9, body="sharp"
        )
        _, _, _, rounded = bv.voxel_difference_weights(
            24, edge_m=16e-9, body="rounded", corner_radius_m=1.5e-9
        )
        self.assertEqual(sharp, 24**3)
        self.assertLess(rounded, sharp)
        self.assertGreater(rounded / 24**3, 0.95)


class TestContinuumLimit(unittest.TestCase):
    """Two independent evaluations of the same continuum integral."""

    @classmethod
    def setUpClass(cls):
        cls.model = bv.BrushRhombohedralSL(16.0, LEGACY)

    def test_laplace_matches_graded_quadrature(self):
        cases = [
            np.array([24.0e-9, 0.0, 0.0]),
            np.array([16.2e-9, 0.0, 0.0]),
            np.array([19.563e-9, 5.449e-9, 0.283e-9]),
            np.array([-5.166e-9, 19.28e-9, -14.114e-9]),
        ]
        for offset in cases:
            exact = self.model.vdw_pair_energy_exact_J(offset)
            quadrature = self.model.vdw_pair_energy_continuum_J(offset, 9, 8)
            self.assertAlmostEqual(exact / quadrature, 1.0, places=8)

    def test_flat_plate_asymptote(self):
        """Close in, the cube pair must approach -A S / (12 pi D^2)."""
        edge = 16e-9
        for gap in (0.2e-9, 0.1e-9):
            offset = np.array([edge + gap, 0.0, 0.0])
            exact = self.model.vdw_pair_energy_exact_J(offset)
            plate = -2.0e-20 * edge**2 / (12.0 * np.pi * gap**2)
            self.assertLess(abs(exact / plate - 1.0), 0.05)

    def test_scale_invariance(self):
        """The integral depends on p / L only."""
        small = bv.BrushRhombohedralSL(8.0, LEGACY)
        large = bv.BrushRhombohedralSL(24.0, LEGACY)
        ratio = 3.0
        offset = np.array([10.0e-9, 2.0e-9, 1.0e-9])
        self.assertAlmostEqual(
            small.vdw_pair_energy_exact_J(offset)
            / large.vdw_pair_energy_exact_J(offset * ratio),
            1.0,
            places=8,
        )

    def test_voxel_ladder_converges_at_second_order(self):
        offset = np.array([19.563e-9, 5.449e-9, 0.283e-9])
        reference = self.model.vdw_pair_energy_exact_J(offset)
        errors = {}
        for count in (16, 32, 64):
            model = bv.BrushRhombohedralSL(16.0, LEGACY.with_voxels(count))
            value = model.vdw_pair_energy_voxel_J(offset)
            errors[count] = abs(value / reference - 1.0)
        self.assertLess(errors[64], errors[32])
        self.assertLess(errors[32], errors[16])
        # Halving the step must cut the error by roughly four.
        for coarse, fine in ((16, 32), (32, 64)):
            ratio = errors[coarse] / errors[fine]
            self.assertGreater(ratio, 2.5)
            self.assertLess(ratio, 6.0)

    def test_voxel_underestimates_attraction(self):
        offset = np.array([19.563e-9, 5.449e-9, 0.283e-9])
        reference = self.model.vdw_pair_energy_exact_J(offset)
        legacy = self.model.vdw_pair_energy_voxel_J(offset)
        self.assertGreater(legacy, reference)          # less negative
        self.assertLess(legacy / reference, 0.7)       # by more than 30 %


class TestAdGPlate(unittest.TestCase):
    """The flat-plate Alexander-de Gennes expression itself."""

    def setUp(self):
        self.l0 = 2.0e-9
        self.s = 0.6324555320336759e-9
        self.temperature = 298.15

    def energy(self, gap_nm):
        return bv.adg_plate_energy_J_per_m2(
            np.asarray(gap_nm) * 1e-9, self.l0, self.s, self.temperature
        )

    def pressure(self, gap_nm):
        return bv.adg_plate_pressure_Pa(
            np.asarray(gap_nm) * 1e-9, self.l0, self.s, self.temperature
        )

    def test_vanishes_at_and_above_two_brush_heights(self):
        self.assertEqual(self.energy(4.0), 0.0)
        self.assertEqual(self.energy(5.0), 0.0)
        self.assertEqual(self.pressure(4.0), 0.0)

    def test_diverges_at_contact(self):
        self.assertEqual(self.energy(0.0), np.inf)
        self.assertGreater(self.energy(1e-6), self.energy(1e-3))

    def test_positive_and_monotonically_decreasing(self):
        gaps = np.linspace(0.05, 3.99, 60)
        values = self.energy(gaps)
        self.assertTrue(np.all(values > 0.0))
        self.assertTrue(np.all(np.diff(values) < 0.0))

    def test_pressure_is_minus_derivative(self):
        gap = 2.5
        step = 1e-6
        numeric = -(
            float(self.energy(gap + step)) - float(self.energy(gap - step))
        ) / (2.0 * step * 1e-9)
        self.assertAlmostEqual(
            float(self.pressure(gap)) / numeric, 1.0, places=6
        )

    def test_literal_prefactor(self):
        """2 kT L0 / s^3 times the Eq. 16.12 bracket, written out again."""
        gap = 2.0e-9
        d = gap / (2.0 * self.l0)
        bracket = (4.0 / 5.0) * (d ** (-5.0 / 4.0) - 1.0) + (4.0 / 7.0) * (
            d ** (7.0 / 4.0) - 1.0
        )
        expected = (
            2.0 * Boltzmann * self.temperature * self.l0 / self.s**3 * bracket
        )
        self.assertAlmostEqual(
            float(self.energy(2.0)) / expected, 1.0, places=12
        )

    def test_divergence_exponent_is_five_quarters(self):
        """The plate energy must go as D^(-5/4), not as the sphere D^(-1/4).

        Confusing the two is the standard transcription error; it understates
        the brush by about an order of magnitude at 15 % compression.
        """
        gaps = np.array([1e-4, 1e-5])
        values = self.energy(gaps)
        slope = np.log(values[1] / values[0]) / np.log(gaps[1] / gaps[0])
        self.assertAlmostEqual(slope, -1.25, places=4)

    def test_sphere_form_is_the_integral_of_the_plate_form(self):
        """Eq. 16.13 must be pi R times the D-integral of Eq. 16.12."""
        radius = 50.0e-9
        gap = 2.0e-9
        grid = np.linspace(gap, 2.0 * self.l0, 200_001)
        integral = np.trapezoid(self.energy(grid * 1e9), grid)
        expected = np.pi * radius * integral
        reference = bv.adg_sphere_energy_J(
            gap, radius, self.l0, self.s, self.temperature
        )
        self.assertAlmostEqual(reference / expected, 1.0, places=4)


class TestGeometryOfTheBrushTerm(unittest.TestCase):
    """Chords, closest approach, and the two brush models."""

    @classmethod
    def setUpClass(cls):
        cls.phys = bv.BrushPhys(derjaguin_transverse_points=96)
        cls.model = bv.BrushRhombohedralSL(16.0, cls.phys)
        cls.kT = Boltzmann * cls.phys.temperature_K

    def test_central_chords(self):
        hb = self.model.box_halfwidth
        radius = self.model.r_round
        for direction, expected in [
            ([1, 0, 0], 2.0 * (hb + radius)),
            ([1, 1, 0], 2.0 * (np.sqrt(2.0) * hb + radius)),
            ([1, 1, 1], 2.0 * (np.sqrt(3.0) * hb + radius)),
        ]:
            direction = np.array(direction, dtype=float)
            direction /= np.linalg.norm(direction)
            chord = bv.rounded_cube_chord(
                direction, np.zeros((1, 3)), hb, radius
            )[0]
            self.assertAlmostEqual(chord / expected, 1.0, places=9)

    def test_chord_is_zero_outside_the_shadow(self):
        far = np.array([[0.0, 30e-9, 0.0]])
        chord = bv.rounded_cube_chord(
            np.array([1.0, 0.0, 0.0]),
            far,
            self.model.box_halfwidth,
            self.model.r_round,
        )
        self.assertEqual(chord[0], 0.0)

    def test_closest_approach_matches_exact_gap(self):
        gaps = self.model.neighbour_exact_gaps(20.31, 74.0, -15.0)
        offsets = self.model.neighbour_vectors(
            20.31, 74.0
        ) @ self.model.orientation(-15.0)
        for offset, gap in zip(offsets, gaps):
            _, value = self.model.closest_approach_normal(offset)
            self.assertAlmostEqual(value * 1e9, gap * 1e9, places=9)

    def test_minimum_local_gap_is_the_exact_gap(self):
        """The Derjaguin window must touch the true closest approach."""
        gaps = self.model.neighbour_exact_gaps(20.31, 74.0, -15.0)
        offsets = self.model.neighbour_vectors(
            20.31, 74.0
        ) @ self.model.orientation(-15.0)
        order = np.argsort(gaps)[:6]
        for index in order:
            offset = offsets[index]
            normal, gap = self.model.closest_approach_normal(offset)
            along = float(np.dot(offset, normal))
            lateral = offset - along * normal
            window = self.model._transverse_window(normal, lateral, along, 192)
            self.assertIsNotNone(window)
            e1, e2, c1, c2, _ = window
            local = along + self.model._local_gap(
                normal, e1, e2, c1, c2, lateral
            )
            self.assertAlmostEqual(
                float(np.min(local)) * 1e9, gap * 1e9, places=3
            )

    def test_facet_model_is_area_times_plate_energy(self):
        gap = 2.5e-9
        offset = np.array([16e-9 + gap, 0.0, 0.0])
        area = (16e-9 - 2.0 * self.model.r_round) ** 2
        plate = bv.adg_plate_energy_J_per_m2(
            gap,
            self.phys.brush_height_nm * 1e-9,
            self.phys.grafting_spacing_nm * 1e-9,
            self.phys.temperature_K,
        )
        self.assertAlmostEqual(
            self.model.facet_repulsion_pair_J(offset) / (area * plate),
            1.0,
            places=9,
        )

    def test_derjaguin_exceeds_facet_on_an_aligned_bond(self):
        """The rounded edge ring adds to the flat facet, it cannot subtract."""
        for gap in (3.5e-9, 2.0e-9, 1.0e-9):
            offset = np.array([16e-9 + gap, 0.0, 0.0])
            facet = self.model.facet_repulsion_pair_J(offset)
            derjaguin = self.model.derjaguin_repulsion_pair_J(offset)
            self.assertGreater(derjaguin, facet)
            self.assertLess(derjaguin / facet, 1.5)

    def test_derjaguin_is_nonzero_where_facet_is_blind(self):
        """Edge contacts carry brush repulsion; the facet model misses it."""
        offsets = self.model.neighbour_vectors(
            20.31, 74.0
        ) @ self.model.orientation(-15.0)
        gaps = self.model.neighbour_exact_gaps(20.31, 74.0, -15.0)
        index = int(np.argmin(gaps))
        self.assertEqual(self.model.facet_repulsion_pair_J(offsets[index]), 0.0)
        self.assertGreater(
            self.model.derjaguin_repulsion_pair_J(offsets[index]), 0.0
        )

    def test_surface_integration_reproduces_the_sphere_formula(self):
        """SEI on two spheres must give Israelachvili Eq. 16.13.

        A rounded cube with zero box half-width is a sphere, so the same
        machinery that carries the brush on the cube facets can be pointed at
        the one geometry for which a closed-form Derjaguin result exists.
        This pins the plate prefactor, the surface-element integration and the
        window finder in one test.  Large spheres are used so that the
        parabolic surface approximation behind Eq. 16.13 is itself accurate.
        """
        radius_nm = 200.0
        phys = self.phys.replace(
            roundness_nm=radius_nm, derjaguin_transverse_points=192
        )
        model = bv.BrushRhombohedralSL(2.0 * radius_nm, phys)
        self.assertLess(model.box_halfwidth, 1e-14)
        for gap_nm in (3.9, 3.0, 2.0, 1.0):
            offset = np.array([(2.0 * radius_nm + gap_nm) * 1e-9, 0.0, 0.0])
            sei = model.derjaguin_repulsion_pair_J(offset)
            closed_form = bv.adg_sphere_energy_J(
                gap_nm * 1e-9,
                radius_nm * 1e-9,
                phys.brush_height_nm * 1e-9,
                phys.grafting_spacing_nm * 1e-9,
                phys.temperature_K,
            )
            self.assertAlmostEqual(sei / closed_form, 1.0, places=2)

    def test_transverse_grid_insensitivity(self):
        coarse = bv.BrushRhombohedralSL(
            16.0, self.phys.replace(derjaguin_transverse_points=32)
        )
        fine = bv.BrushRhombohedralSL(
            16.0, self.phys.replace(derjaguin_transverse_points=192)
        )
        for gap_nm in (3.4, 2.5, 1.5):
            a_nm = coarse.a_for_exact_gap(74.0, -15.0, gap_nm)
            low = coarse.energy_components(a_nm, 74.0, -15.0)["Steric"]
            high = fine.energy_components(a_nm, 74.0, -15.0)["Steric"]
            self.assertLess(abs(low / high - 1.0), 5e-3)

    def test_repulsion_switches_off_beyond_two_brush_heights(self):
        a_nm = self.model.a_for_exact_gap(74.0, -15.0, 4.01)
        components = self.model.energy_components(a_nm, 74.0, -15.0)
        self.assertEqual(components["Steric"], 0.0)

    def test_canonical_half_shell_bookkeeping(self):
        offsets = self.model.neighbour_vectors(
            20.31, 74.0
        ) @ self.model.orientation(-15.0)
        full = 0.5 * sum(
            self.model.repulsion_pair_J(offset) for offset in offsets
        )
        half = sum(
            self.model.repulsion_pair_J(offset)
            for offset, use in zip(offsets, self.model.canonical_half_mask)
            if use
        )
        self.assertAlmostEqual(half / full, 1.0, places=12)


class TestGapRelaxation(unittest.TestCase):
    """The relaxed state is a genuine stationary point of Interaction."""

    @classmethod
    def setUpClass(cls):
        cls.model = bv.BrushRhombohedralSL(
            16.0,
            bv.BrushPhys(vdw_scheme="continuum"),
        )
        cls.relaxed = cls.model.relax_gap(74.0, -15.0, grid_points=32)

    def test_relaxation_converged_inside_the_window(self):
        self.assertTrue(self.relaxed["converged"])
        self.assertGreater(self.relaxed["gap_nm"], 0.05)
        self.assertLess(
            self.relaxed["gap_nm"], 2.0 * self.model.L0 * 1e9
        )

    def test_energy_is_stationary_in_the_gap(self):
        gap = self.relaxed["gap_nm"]
        step = 2e-3
        energy = self.relaxed["components"]["Interaction"]
        for sign in (-1.0, 1.0):
            shifted = self.model.interaction_at_gap(
                74.0, -15.0, gap + sign * step
            )
            self.assertGreater(shifted, energy - 1e-9)

    def test_gap_constraint_is_satisfied(self):
        self.assertAlmostEqual(
            self.model.nearest_exact_gap(
                self.relaxed["a_nm"], 74.0, -15.0
            ),
            self.relaxed["gap_nm"],
            places=7,
        )


class TestFittedRepulsionTwistScan(unittest.TestCase):
    """The fixed-a twist scan of the R series must not reach sharp overlap.

    The exact Hamaker integral uses sharp cubes and the gap geometry rounded
    ones.  At (a, alpha) = (22.4 nm, 63 deg) the twist gamma = +/-60 deg puts
    the primitive bond [0, 0, -1] into an edge contact whose rounded gap is
    1.004 nm while its sharp cubes overlap by 0.17 nm; the integral then
    returns about -2e17 kBT, which is how the first landscape run failed.
    """

    @classmethod
    def setUpClass(cls):
        import v620_repulsion_fit as rf

        cls.rf = rf
        cls.model = rf.exponential_model(32192.79, 0.48984)

    def test_the_singular_twist_passes_the_rounded_floor_alone(self):
        floor = self.rf.PHYSICAL_GAP_FLOOR_NM
        self.assertGreater(
            self.model.nearest_exact_gap(22.4, 63.0, -60.0), floor
        )
        self.assertLess(
            self.rf.vdw_sharp_gap_nm(self.model, 22.4, 63.0, -60.0), 0.0
        )

    def test_twist_scan_excludes_sharp_overlap(self):
        gammas = np.linspace(-60.0, 60.0, 37, endpoint=False)
        gamma, components = self.rf.relax_twist_at_fixed_a(
            self.model, 22.4, 63.0, gammas
        )
        self.assertGreaterEqual(
            self.rf.vdw_sharp_gap_nm(self.model, 22.4, 63.0, gamma),
            self.rf.PHYSICAL_GAP_FLOOR_NM,
        )
        self.assertGreater(components["Interaction"], -10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
