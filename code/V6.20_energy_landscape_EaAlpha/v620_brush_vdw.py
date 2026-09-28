"""V6.20 van der Waals module extended with an Alexander-de Gennes brush.

Scope
-----
This module is a faithful port of the V6.20 ``ReferenceGapRhombohedralSL``
Hamiltonian (notebook ``V6.20rhombo_EaAlpha.ipynb``, cell 2) plus three
additions that the notebook does not have:

1. a ligand-brush repulsion built from the Alexander-de Gennes (AdG)
   flat-plate brush free energy, applied to the rounded-cube surfaces of the
   V6.20 geometry;
2. a rewritten attractive van der Waals sum that is algebraically identical to
   the V6.20 voxel double sum but costs ``O((2N-1)^3)`` instead of ``O(N^6)``,
   so the voxel count ``N`` can be refined far enough to test convergence;
3. the continuum limit of that same integral, in closed form down to a single
   one-dimensional quadrature (:func:`cube_pair_hamaker_J`), so the voxel
   ladder can be measured against a reference that shares no discretisation
   with it.  A second, independent graded three-dimensional quadrature of the
   autocorrelation kernel is kept as a cross-check and agrees with it to about
   1e-12 from an 8 nm gap down to a 0.05 nm gap.

Nothing in the notebook is modified.  The legacy behaviour is reproduced
exactly when ``BrushPhys(brush_model="off")`` is used with ``voxel_count=4``;
``test_v620_brush_vdw.py`` pins that against the archived V6.20 run log.

Conventions inherited from V6.20 (kept deliberately)
----------------------------------------------------
* Particle: rounded cube, edge ``L``, Minkowski corner radius 1.5 nm, i.e.
  a box of half-width ``(L - 2r)/2`` dilated by a ball of radius ``r``.  The
  flat facet is ``(L - 2r) x (L - 2r)`` and sits ``L/2`` from the centre.
* The attractive vdW integral uses the *sharp* cube of edge ``L`` on a
  midpoint voxel lattice, while the gap geometry uses the rounded cube.  That
  inconsistency belongs to V6.20, not to this module, and is quantified here
  by the ``voxel_body="rounded"`` option.
* Energies are per particle: one half of the sum over the 26 neighbours of a
  3x3x3 block.  The main Hamiltonian keeps only the six primitive bonds for
  the vdW term (``vdw_neighbour_mode="primitive6"``); the 26-bond sum is a
  diagnostic.
* ``Interaction`` is the quantity minimised over (gamma, alpha) and, in the
  extended model, over the gap as well.

Alexander-de Gennes term
------------------------
Flat-plate brush free energy per unit area, two opposing brushes of unstressed
height ``L0`` grafted at mean spacing ``s``, core-core separation ``D``
(de Gennes 1987; Israelachvili, Intermolecular and Surface Forces, 3rd ed.,
Eqs. 16.11 and 16.12):

    P(D) = (kT / s^3) [ (2 L0 / D)^(9/4) - (D / 2 L0)^(3/4) ]      D < 2 L0
    w(D) = (2 kT L0 / s^3) [ (4/5) ((2 L0 / D)^(5/4) - 1)
                             + (4/7) ((D / 2 L0)^(7/4) - 1) ]      D < 2 L0
    w(D) = 0                                                       D >= 2 L0

The much-copied ``28 / (20/11) / 12`` bracket is the D-integral of ``w``, i.e.
the Derjaguin *sphere-sphere* energy of Eq. 16.13, not the plate energy; it
diverges as ``D^(-1/4)`` instead of ``D^(-5/4)``.  Applying the plate
expression over the actual surfaces, which is what the Derjaguin mode below
does, reproduces Eq. 16.13 for two spheres to 0.2 %; that is the test which
pins the prefactor.

Two ways of putting that on a cube are provided.

``brush_model="facet"``
    The pair of parallel facing flat facets only.  For two co-oriented rounded
    cubes at body-frame offset ``p`` the facet pair normal to axis ``k`` faces
    the neighbour when ``|p_k| - L > 0``; its separation is ``|p_k| - L`` and
    its overlap area is the product over the other two axes of
    ``max(0, (L - 2r) - |p_i|)``.  At most one axis can contribute.  This is
    the textbook nanocube face-to-face model.  It returns exactly zero for
    edge- and corner-type contacts, which in the rhombohedral lattice are the
    *closest* surfaces, so it is a lower bound.

``brush_model="derjaguin"`` (default)
    Derjaguin / surface-element integration on the plane of closest approach.
    The integration axis is the closest-approach normal ``n``, not the bond
    direction: for a rhombohedral lattice of co-oriented cubes the closest
    surfaces are edge-type contacts whose normal is *not* parallel to the bond,
    and integrating along the bond would report no brush contact at all where
    the surfaces are in fact the closest in the whole lattice.  ``n`` and the
    exact pair gap both follow in closed form, because the pair separation is
    the distance from ``p`` to the Minkowski difference body (a box of
    half-width ``L - 2r`` dilated by ``2r``); see
    :meth:`BrushRhombohedralSL.closest_approach_normal`.

    With the two bodies identical and co-oriented, the separation along a line
    parallel to ``n`` at transverse position ``q`` is exactly

        h(q) = p.n + t_lo(q - q_lat) - t_hi(q),

    where ``[t_lo, t_hi]`` is the ray/body intersection and ``q_lat`` is the
    transverse part of ``p``.  Then
    ``U_rep(p) = integral of w(h(q)) d^2q``.  The ray/body intersection is in
    closed form, not ray marched: the body is the set where
    ``sum_i max(|x_i| - hb, 0)^2 <= r^2``, so along a ray the constraint is
    piecewise quadratic in ``t`` with at most six breakpoints.  By
    construction ``min_q h(q)`` equals the exact pair gap, which is asserted
    in the tests.  For an aligned face-to-face bond this gives the flat facet
    term of ``facet`` plus the contribution of the rounded edge ring, which is
    12 to 25 % on top.

Known limitations, stated rather than hidden
--------------------------------------------
* AdG assumes monodisperse, strongly stretched, non-interdigitating brushes
  facing each other symmetrically (``L0 >> s``).  For oleic acid on Fe3O4,
  ``L0/s`` is about 3, so the scaling regime is marginal; the term should be
  read as a stiff, physically motivated steric repulsion, not as a
  quantitative polymer free energy.  Interdigitation, which is what ligand
  shells at these separations are usually believed to do, would soften it.
* ``L0`` and ``s`` are *not* calibrated here.  The defaults are a fully
  extended oleic acid length and a 2.5 per nm^2 grafting density; both are
  swept, and ``L0`` is calibrated against the experimental gap, in
  ``run_brush_refit.py``.
* The brush diverges as ``D^(-5/4)`` and the Hamaker attraction of two flat
  plates as ``D^(-2)``, so a brush facing a fully converged attraction is not
  formally a hard core.  Whether that matters is a quantitative question, and
  ``run_brush_refit.py`` answers it by scanning down to a 1e-4 nm gap instead
  of assuming either way.  With the V6.20 primitive-six neighbour convention
  no crossover exists at all, because the surfaces that actually come into
  contact are the edge-type neighbours and that convention excludes them from
  the attraction.  With ``all26`` the collapse branch does exist, but only
  below a 0.002 nm gap and behind a 1e6 kBT barrier peaking at 0.004 nm - a
  hundredth of an atomic bond length, where neither continuum expression means
  anything.  Either way the relaxed spacing is the global minimum over every
  physically meaningful separation.
* Derjaguin / SEI applies a flat-plate expression to surfaces that are only
  locally parallel.  It is exact for the facet-facet contacts and
  approximate for the edge and corner contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.constants import Boltzmann, mu_0
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf, erfc

__all__ = [
    "BrushPhys",
    "LEGACY_PHYS",
    "EXP_DATA",
    "REFERENCE",
    "adg_plate_energy_J_per_m2",
    "adg_plate_pressure_Pa",
    "adg_sphere_energy_J",
    "exponential_steric_pressure_Pa",
    "exponential_steric_energy_J_per_m2",
    "rounded_cube_chord",
    "voxel_difference_weights",
    "tent_gaussian_integral",
    "cube_pair_hamaker_J",
    "BrushRhombohedralSL",
]


# ----------------------------------------------------------------------
#  Parameters
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BrushPhys:
    """V6.20 physical parameters plus the brush parameters."""

    # --- V6.20 block, values identical to the notebook PHYS dict ---
    temperature_K: float = 298.15
    Ms: float = 285_000.0            # A/m
    B_field: float = 0.5             # T
    K_ani: float = 2.0e4             # J/m^3
    Hamaker: float = 2.0e-20         # J
    roundness_nm: float = 1.5
    vdw_d2_floor_m2: float = 1.0e-19
    voxel_count: int = 4             # the V6.20 "N_voxel"
    vdw_neighbour_mode: str = "primitive6"   # "primitive6" | "all26"
    voxel_body: str = "sharp"                # "sharp" | "rounded"

    # Quadrature rule for the attractive integral.  "voxel" is the V6.20
    # midpoint rule at ``voxel_count``; "continuum" is the exact limit and is
    # the recommended production rule, since it is both more accurate and
    # cheaper than a converged voxel sum (about 0.2 ms per bond).
    # "richardson" is the h^2 extrapolation of the voxel pair (N, 2N); it is
    # accurate to about 0.1 % at a 4 nm gap but degrades badly once the gap
    # falls below the voxel step (58 % error at a 0.2 nm gap with N = 16), so
    # it is kept for the convergence study rather than for production.
    vdw_scheme: str = "voxel"        # "voxel" | "richardson" | "continuum"
    richardson_base: int = 16
    # With diagnostics off, only the bonds the Hamiltonian needs are summed
    # and the 26-bond diagnostics come back as NaN.  That is 4.3x faster on
    # the primitive-6 Hamiltonian and is meant for scans.
    vdw_diagnostics: bool = True
    continuum_grading_levels: int = 12
    continuum_gauss_points: int = 6

    # --- Alexander-de Gennes brush block (new) ---
    # Fully extended oleic acid, and a 2.5 per nm^2 grafting density
    # (0.40 nm^2 per chain).  Placeholders; see run_brush_refit.py.
    brush_height_nm: float = 2.0
    grafting_spacing_nm: float = 0.6324555320336759   # sigma = 2.5 nm^-2
    brush_model: str = "derjaguin"             # "derjaguin" | "facet" | "off"
    brush_neighbour_mode: str = "all26"        # "all26" | "primitive6"
    derjaguin_transverse_points: int = 32

    # --- Areal repulsion law carried by brush_model ---
    # "adg" is the Alexander-de Gennes plate energy above.  "exponential" is
    # the empirical soft steric law w(D) = P0 * lambda * exp(-D / lambda),
    # i.e. a disjoining pressure P0 * exp(-D / lambda), used for the fitted
    # repulsion in run_repulsion_fit.py.  The same surface-element geometry
    # carries either law, so only the areal energy changes.
    brush_law: str = "adg"                     # "adg" | "exponential"
    steric_pressure_Pa: float = 1.0e7
    steric_decay_nm: float = 0.5
    # Separation at which steric_pressure_Pa applies (0 = at contact).
    steric_reference_gap_nm: float = 0.0
    # The exponential has no hard cut-off; surfaces further apart than this
    # many decay lengths are treated as non-interacting.  At 14 decay lengths
    # the areal energy is below 1e-6 of its contact value.
    steric_range_decays: float = 14.0

    @property
    def grafting_density_per_nm2(self) -> float:
        return 1.0 / self.grafting_spacing_nm**2

    def with_voxels(self, voxel_count: int) -> "BrushPhys":
        return replace(self, voxel_count=int(voxel_count))

    def replace(self, **kwargs) -> "BrushPhys":
        return replace(self, **kwargs)


LEGACY_PHYS = BrushPhys(brush_model="off")

# V6.20 experimental inputs, copied from the notebook.
EXP_DATA = {
    16.0: {"d": 20.31, "alpha": 74.0},
    18.0: {"d": 22.2, "alpha": 73.0},
    18.2: {"d": 23.5, "alpha": 72.0},
    19.5: {"d": 29.0, "alpha": 60.0},
    29.6: {"d": 39.6, "alpha": 60.0},
}
REFERENCE = {
    "L_nm": 16.0,
    "a_nm": EXP_DATA[16.0]["d"],
    "alpha_deg": EXP_DATA[16.0]["alpha"],
    "gamma_deg": -15.0,
}


# ----------------------------------------------------------------------
#  Alexander-de Gennes flat-plate brush
# ----------------------------------------------------------------------
def adg_plate_pressure_Pa(
    gap_m,
    brush_height_m: float,
    grafting_spacing_m: float,
    temperature_K: float,
):
    """de Gennes disjoining pressure between two opposing flat brushes.

        P(D) = (kT / s^3) [ (2 L0 / D)^(9/4) - (D / 2 L0)^(3/4) ],  D < 2 L0

    de Gennes, C. R. Acad. Sci. Paris 300, 839 (1985) / Adv. Colloid
    Interface Sci. 27, 189 (1987); Israelachvili, Intermolecular and Surface
    Forces, 3rd ed., Eq. 16.11.  Zero at and above ``2 L0``.
    """
    gap = np.asarray(gap_m, dtype=float)
    scalar = gap.ndim == 0
    gap = np.atleast_1d(gap)
    two_l0 = 2.0 * float(brush_height_m)
    prefactor = (
        Boltzmann
        * float(temperature_K)
        / float(grafting_spacing_m) ** 3
    )
    active = (gap > 0.0) & (gap < two_l0)
    d = np.where(active, gap / two_l0, 1.0)
    bracket = d ** (-9.0 / 4.0) - d ** (3.0 / 4.0)
    out = np.where(active, prefactor * bracket, 0.0)
    out = np.where(gap <= 0.0, np.inf, out)
    return float(out[0]) if scalar else out


def adg_plate_energy_J_per_m2(
    gap_m,
    brush_height_m: float,
    grafting_spacing_m: float,
    temperature_K: float,
):
    """AdG free energy per unit area between two opposing flat brushes.

    The integral of :func:`adg_plate_pressure_Pa` from ``D`` up to ``2 L0``:

        w(D) = (2 kT L0 / s^3) [ (4/5) ((2 L0 / D)^(5/4) - 1)
                                 + (4/7) ((D / 2 L0)^(7/4) - 1) ]

    (Israelachvili, 3rd ed., Eq. 16.12.)  ``gap_m`` is the core-core
    separation.  Returns 0 for ``gap >= 2 L0`` and ``+inf`` for ``gap <= 0``.

    Note which expression this is.  The bracket
    ``28 (d^-1/4 - 1) + (20/11)(1 - d^11/4) + 12 (d - 1)`` that appears
    throughout the nanocrystal literature is *not* this function: it is the
    ``D``-integral of it, i.e. the Derjaguin sphere-sphere energy
    (Eq. 16.13), and it diverges only as ``D^(-1/4)`` where the plate energy
    diverges as ``D^(-5/4)``.  Using the sphere bracket as a plate energy
    understates the brush by about a factor of ten at 15 % compression.
    :func:`adg_sphere_energy_J` provides the sphere form, and the tests use it
    to check this function together with the surface-element integration.
    """
    gap = np.asarray(gap_m, dtype=float)
    scalar = gap.ndim == 0
    gap = np.atleast_1d(gap)
    two_l0 = 2.0 * float(brush_height_m)
    prefactor = (
        2.0
        * Boltzmann
        * float(temperature_K)
        * float(brush_height_m)
        / float(grafting_spacing_m) ** 3
    )
    active = (gap > 0.0) & (gap < two_l0)
    d = np.where(active, gap / two_l0, 1.0)
    bracket = (4.0 / 5.0) * (d ** (-5.0 / 4.0) - 1.0) + (4.0 / 7.0) * (
        d ** (7.0 / 4.0) - 1.0
    )
    out = np.where(active, prefactor * bracket, 0.0)
    out = np.where(gap <= 0.0, np.inf, out)
    return float(out[0]) if scalar else out


def adg_sphere_energy_J(
    gap_m,
    sphere_radius_m: float,
    brush_height_m: float,
    grafting_spacing_m: float,
    temperature_K: float,
):
    """Derjaguin AdG energy of two equal brush-coated spheres.

        W(D) = (16 pi kT L0^2 R / (35 s^3))
               [ 28 ((2 L0 / D)^(1/4) - 1)
                 + (20/11) (1 - (D / 2 L0)^(11/4))
                 + 12 (D / 2 L0 - 1) ]

    (Israelachvili, 3rd ed., Eq. 16.13.)  This is ``pi R`` times the integral
    of :func:`adg_plate_energy_J_per_m2`, the Derjaguin result for two equal
    spheres, and is kept here only as an independent reference for the tests.
    """
    gap = np.asarray(gap_m, dtype=float)
    scalar = gap.ndim == 0
    gap = np.atleast_1d(gap)
    two_l0 = 2.0 * float(brush_height_m)
    prefactor = (
        16.0
        * np.pi
        * Boltzmann
        * float(temperature_K)
        * float(brush_height_m) ** 2
        * float(sphere_radius_m)
        / (35.0 * float(grafting_spacing_m) ** 3)
    )
    active = (gap > 0.0) & (gap < two_l0)
    d = np.where(active, gap / two_l0, 1.0)
    bracket = (
        28.0 * (d ** (-0.25) - 1.0)
        + (20.0 / 11.0) * (1.0 - d ** (11.0 / 4.0))
        + 12.0 * (d - 1.0)
    )
    out = np.where(active, prefactor * bracket, 0.0)
    out = np.where(gap <= 0.0, np.inf, out)
    return float(out[0]) if scalar else out


def exponential_steric_pressure_Pa(
    gap_m, pressure_Pa, decay_m, range_m, reference_m=0.0
):
    """Soft steric disjoining pressure ``P_ref * exp(-(D - D_ref) / lambda)``.

    ``pressure_Pa`` is the pressure at the reference separation
    ``reference_m`` (at contact when that is zero).  Referencing the
    amplitude to the working gap keeps the fit well conditioned: at a
    0.1 nm decay length the contact amplitude would be e^34 times larger.
    Zero beyond ``range_m``, ``+inf`` at non-positive gaps.
    """
    gap = np.asarray(gap_m, dtype=float)
    scalar = gap.ndim == 0
    gap = np.atleast_1d(gap)
    active = (gap > 0.0) & (gap < float(range_m))
    exponent = -(np.where(active, gap, 0.0) - float(reference_m)) / float(
        decay_m
    )
    out = np.where(
        active, float(pressure_Pa) * np.exp(np.minimum(exponent, 700.0)), 0.0
    )
    out = np.where(gap <= 0.0, np.inf, out)
    return float(out[0]) if scalar else out


def exponential_steric_energy_J_per_m2(
    gap_m, pressure_Pa, decay_m, range_m, reference_m=0.0
):
    """Areal energy of :func:`exponential_steric_pressure_Pa`.

        w(D) = P_ref lambda [exp(-(D - D_ref)/lambda)
                             - exp(-(D_cut - D_ref)/lambda)]

    The constant makes ``w`` vanish continuously at the truncation
    ``D_cut = range_m``, so the pressure is exactly ``-dw/dD`` below the cut.
    """
    gap = np.asarray(gap_m, dtype=float)
    scalar = gap.ndim == 0
    gap = np.atleast_1d(gap)
    decay = float(decay_m)
    cut = float(range_m)
    reference = float(reference_m)
    active = (gap > 0.0) & (gap < cut)
    exponent = -(np.where(active, gap, 0.0) - reference) / decay
    out = np.where(
        active,
        float(pressure_Pa)
        * decay
        * (
            np.exp(np.minimum(exponent, 700.0))
            - np.exp(-(cut - reference) / decay)
        ),
        0.0,
    )
    out = np.where(gap <= 0.0, np.inf, out)
    return float(out[0]) if scalar else out


# ----------------------------------------------------------------------
#  Exact chord of a rounded cube along a direction
# ----------------------------------------------------------------------
def rounded_cube_chord_bounds(
    direction: np.ndarray,
    transverse_points: np.ndarray,
    box_halfwidth_m: float,
    corner_radius_m: float,
):
    """Entry and exit parameters of a ray through a rounded cube.

    The body is the set where ``sum_i max(|x_i| - hb, 0)^2 <= r^2`` with
    ``hb = box_halfwidth_m`` and ``r = corner_radius_m``, i.e. a box of
    half-width ``hb`` dilated by a ball of radius ``r``.  For each point ``q``
    of ``transverse_points`` (shape ``(n, 3)``) the line ``q + t * direction``
    meets the body on ``[t_lo, t_hi]``.  Returns ``(t_lo, t_hi, hit)``; where
    ``hit`` is False the bounds are meaningless.

    Along a ray the constraint is piecewise quadratic in ``t`` with at most
    six breakpoints, so the bounds are obtained in closed form rather than by
    ray marching.
    """
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    q = np.atleast_2d(np.asarray(transverse_points, dtype=float))
    n = q.shape[0]
    hb = float(box_halfwidth_m)
    radius = float(corner_radius_m)
    r2 = radius**2

    # Breakpoints: where an axis enters or leaves its slab.
    t_span = 2.0 * (hb * float(np.sum(np.abs(u))) + radius) + 1.0
    breaks = np.full((n, 6), t_span)
    for axis in range(3):
        if abs(u[axis]) > 1e-14:
            breaks[:, 2 * axis] = (hb - q[:, axis]) / u[axis]
            breaks[:, 2 * axis + 1] = (-hb - q[:, axis]) / u[axis]
    breaks = np.clip(breaks, -t_span, t_span)
    edges = np.concatenate(
        [
            np.full((n, 1), -t_span),
            np.sort(breaks, axis=1),
            np.full((n, 1), t_span),
        ],
        axis=1,
    )                                       # (n, 8)
    lo = edges[:, :-1]
    hi = edges[:, 1:]                       # (n, 7)
    mid = 0.5 * (lo + hi)

    # Active set and quadratic coefficients on each interval.
    pos = q[:, None, :] + mid[:, :, None] * u[None, None, :]    # (n, 7, 3)
    active = np.abs(pos) > hb
    sign = np.where(pos >= 0.0, 1.0, -1.0)
    alpha = np.where(active, sign * u[None, None, :], 0.0)
    beta = np.where(active, sign * q[:, None, :] - hb, 0.0)
    a_coef = np.sum(alpha**2, axis=2)
    b_coef = 2.0 * np.sum(alpha * beta, axis=2)
    c_coef = np.sum(beta**2, axis=2) - r2

    quadratic = np.abs(a_coef) > 1e-30
    disc = b_coef**2 - 4.0 * a_coef * c_coef
    usable = quadratic & (disc >= 0.0)
    sqrt_disc = np.sqrt(np.where(usable, disc, 0.0))
    denom = np.where(quadratic, 2.0 * a_coef, 1.0)
    root_lo = np.where(usable, (-b_coef - sqrt_disc) / denom, np.nan)
    root_hi = np.where(usable, (-b_coef + sqrt_disc) / denom, np.nan)
    roots = np.stack([root_lo, root_hi], axis=2)                # (n, 7, 2)

    # Degenerate interval: no axis outside its slab, or the ray is parallel to
    # every active slab.  Then g(t) is the constant c_coef and the whole
    # interval lies inside the body when c_coef <= 0.
    flat = (~quadratic) & (c_coef <= 0.0)

    inside = (
        (roots >= lo[:, :, None] - 1e-15) & (roots <= hi[:, :, None] + 1e-15)
    )
    candidates = np.concatenate(
        [
            np.where(inside, roots, np.nan).reshape(n, -1),
            np.where(flat, lo, np.nan),
            np.where(flat, hi, np.nan),
        ],
        axis=1,
    )
    empty = np.all(np.isnan(candidates), axis=1)
    safe = np.where(empty[:, None], 0.0, candidates)
    t_min = np.nanmin(safe, axis=1)
    t_max = np.nanmax(safe, axis=1)
    return t_min, t_max, ~empty


def rounded_cube_chord(
    direction: np.ndarray,
    transverse_points: np.ndarray,
    box_halfwidth_m: float,
    corner_radius_m: float,
) -> np.ndarray:
    """Chord length of a rounded cube along ``direction``, 0 where missed."""
    t_lo, t_hi, hit = rounded_cube_chord_bounds(
        direction, transverse_points, box_halfwidth_m, corner_radius_m
    )
    return np.where(hit, np.maximum(t_hi - t_lo, 0.0), 0.0)


def orthonormal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``direction``."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, u))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - float(np.dot(seed, u)) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


# ----------------------------------------------------------------------
#  Voxel difference multiplicities
# ----------------------------------------------------------------------
def voxel_difference_weights(
    voxel_count: int,
    edge_m: float,
    body: str = "sharp",
    corner_radius_m: float = 0.0,
    subsamples: int = 4,
):
    """Difference lattice of a midpoint voxel grid, with multiplicities.

    ``sum_{i,j} f(x_i - x_j)`` over all ``N^3`` voxel centres of a cube equals
    ``sum_k W_k f(d_k)`` over the ``(2N-1)^3`` distinct differences, with
    ``W_k = prod_axes (N - |k_axis|)`` for the full cube.  For
    ``body="rounded"`` each voxel carries the fraction of its volume inside the
    rounded cube and ``W`` is the autocorrelation of that fractional mask,
    obtained by FFT.

    Returns ``(differences, weights, voxel_volume, voxel_number)``, where
    ``voxel_number`` is the summed fractional occupancy, so
    ``voxel_number * voxel_volume`` is the represented body volume.
    """
    n = int(voxel_count)
    step = float(edge_m) / n
    k = np.arange(-(n - 1), n)
    if body == "sharp":
        w1 = (n - np.abs(k)).astype(float)
        weights = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
        voxel_number = n**3
    elif body == "rounded":
        # Partial volumes, not a hard in/out mask.  A hard mask stair-steps
        # the rounded boundary, which makes the body volume converge only at
        # first order and non-monotonically, and then no Richardson
        # extrapolation of the resulting energies is meaningful.  Each voxel
        # is instead weighted by the fraction of its own volume inside the
        # body, estimated on a subsamples^3 grid.
        hb = 0.5 * (float(edge_m) - 2.0 * float(corner_radius_m))
        radius2 = float(corner_radius_m) ** 2
        lin = (np.arange(n) - (n - 1) / 2.0) * step
        sub = (np.arange(subsamples) + 0.5) / subsamples - 0.5
        offsets = sub * step
        fraction = np.zeros((n, n, n))
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    gx, gy, gz = np.meshgrid(
                        lin + dx, lin + dy, lin + dz, indexing="ij"
                    )
                    inside = (
                        np.maximum(np.abs(gx) - hb, 0.0) ** 2
                        + np.maximum(np.abs(gy) - hb, 0.0) ** 2
                        + np.maximum(np.abs(gz) - hb, 0.0) ** 2
                    ) <= radius2
                    fraction += inside
        mask = fraction / subsamples**3
        voxel_number = float(mask.sum())
        size = 2 * n - 1
        spectrum = np.fft.fftn(mask, s=(size, size, size), axes=(0, 1, 2))
        correlation = np.fft.ifftn(spectrum * np.conjugate(spectrum)).real
        weights = np.roll(
            correlation, shift=(n - 1, n - 1, n - 1), axis=(0, 1, 2)
        )
        weights = np.where(np.abs(weights) < 1e-9, 0.0, weights)
    else:
        raise ValueError(f"Unknown voxel_body {body!r}.")

    gx, gy, gz = np.meshgrid(k * step, k * step, k * step, indexing="ij")
    differences = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    weights = weights.ravel()
    keep = weights > 0.0
    return differences[keep], weights[keep], step**3, voxel_number


# ----------------------------------------------------------------------
#  Exact continuum Hamaker energy of two co-oriented cubes
# ----------------------------------------------------------------------
def tent_gaussian_integral(t, offset_m: float, edge_m: float):
    """``integral_{-L}^{L} (L - |u|) exp(-t (u - p)^2) du``, vectorised in t.

    ``L - |u|`` is the one-dimensional factor of the autocorrelation of a cube
    indicator, so this is the exact Gaussian transform of that factor.  The
    error-function differences are evaluated with ``erfc`` when both arguments
    have the same sign, which is the numerically stable branch.
    """
    t = np.asarray(t, dtype=float)
    p = float(offset_m)
    edge = float(edge_m)
    root_t = np.sqrt(t)

    def piece(a: float, b: float):
        xa = root_t * (a - p)
        xb = root_t * (b - p)
        both_positive = (xa > 0.0) & (xb > 0.0)
        both_negative = (xa < 0.0) & (xb < 0.0)
        difference = np.where(
            both_positive,
            erfc(np.abs(xa)) - erfc(np.abs(xb)),
            np.where(
                both_negative,
                erfc(np.abs(xb)) - erfc(np.abs(xa)),
                erf(xb) - erf(xa),
            ),
        )
        zeroth = 0.5 * np.sqrt(np.pi / t) * difference
        first = p * zeroth - (
            np.exp(-np.minimum(xb**2, 700.0))
            - np.exp(-np.minimum(xa**2, 700.0))
        ) / (2.0 * t)
        return zeroth, first

    zeroth_up, first_up = piece(0.0, edge)
    zeroth_down, first_down = piece(-edge, 0.0)
    return (
        edge * zeroth_up - first_up + edge * zeroth_down + first_down
    )


def _laplace_nodes(subintervals: int = 64, gauss_points: int = 12):
    key = (int(subintervals), int(gauss_points))
    cached = _LAPLACE_CACHE.get(key)
    if cached is not None:
        return cached
    nodes, weights = np.polynomial.legendre.leggauss(int(gauss_points))
    edges = np.linspace(-30.0, 32.0, int(subintervals) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    halfwidths = 0.5 * np.diff(edges)
    y = (centres[:, None] + halfwidths[:, None] * nodes[None, :]).ravel()
    w = (halfwidths[:, None] * weights[None, :]).ravel()
    _LAPLACE_CACHE[key] = (y, w)
    return y, w


_LAPLACE_CACHE: dict = {}


def cube_pair_hamaker_J(
    offset_body_m: np.ndarray,
    edge_m: float,
    hamaker_J: float,
    subintervals: int = 64,
    gauss_points: int = 12,
) -> float:
    """Exact Hamaker energy of two co-oriented sharp cubes, no voxels.

    The six-fold integral is reduced to one dimension without approximation.
    First the double volume integral becomes a single convolution with the
    autocorrelation of the cube indicator,

        U = -(A / pi^2) * integral K(u) |u - p|^-6 d^3u,
        K(u) = prod_i max(L - |u_i|, 0);

    then ``|r|^-6 = (1/2) integral_0^inf t^2 exp(-t r^2) dt`` factorises the
    remaining kernel across the three axes, so

        U = -(A / pi^2) * (1/2) integral_0^inf t^2 prod_i J(t, p_i) dt,
        J(t, p) = integral (L - |u|) exp(-t (u - p)^2) du,

    with ``J`` in closed form.  The ``t`` integral is taken in ``ln t`` over
    ``t L^2`` from about ``1e-13`` to ``1e14``, which brackets the integrand
    (it grows as ``t^3`` and decays as ``exp(-t D^2)``) to well below double
    precision.  The result is exact to quadrature error, is independent of any
    body discretisation, and costs about 0.2 ms.

    Valid for non-overlapping cubes only; the integral diverges at contact.
    """
    offset = np.asarray(offset_body_m, dtype=float)
    edge = float(edge_m)
    y, w = _laplace_nodes(subintervals, gauss_points)
    t = np.exp(y) / edge**2
    product = np.ones_like(t)
    for axis in range(3):
        product = product * tent_gaussian_integral(t, offset[axis], edge)
    # dt = t dy, and the t^2 of the Laplace representation gives t^3 here.
    integral = 0.5 * float(np.sum(w * t**3 * product))
    return -(float(hamaker_J) / np.pi**2) * integral


# ----------------------------------------------------------------------
#  Lattice model
# ----------------------------------------------------------------------
class BrushRhombohedralSL:
    """Rhombohedral lattice of co-oriented rounded nanocubes, with a brush.

    Geometry, dipole, anisotropy, Zeeman and the voxel vdW sum are the V6.20
    originals.  ``Steric`` is the new Alexander-de Gennes term.
    """

    def __init__(self, L_nm: float, phys: BrushPhys = BrushPhys()):
        self.phys = phys
        self.L_nm = float(L_nm)
        self.L = self.L_nm * 1e-9
        self.r_round_nm = float(phys.roundness_nm)
        self.r_round = min(self.r_round_nm * 1e-9, self.L / 2.0 - 1e-15)
        self.core_difference_halfwidth = self.L - 2.0 * self.r_round
        self.box_halfwidth = 0.5 * (self.L - 2.0 * self.r_round)

        self.temperature = float(phys.temperature_K)
        self.kT = Boltzmann * self.temperature
        self.volume = self.L**3
        self.Ms = float(phys.Ms)
        self.mu_magnitude = self.Ms * self.volume
        self.mu_hat = np.array([1.0, 0.0, 0.0])
        self.mu_vec = self.mu_magnitude * self.mu_hat
        self.B_field = float(phys.B_field)
        self.K_ani = float(phys.K_ani)
        self.Cmag = mu_0 / (4.0 * np.pi)
        self.Aham = float(phys.Hamaker)
        self.d2_floor = float(phys.vdw_d2_floor_m2)

        # 26 neighbours in a 3x3x3 cell, excluding the origin.
        grid = np.array(
            np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
        ).T.reshape(-1, 3)
        self.neighbour_indices = grid[np.any(grid != 0, axis=1)]
        self.primitive_face_mask = (
            np.sum(np.abs(self.neighbour_indices), axis=1) == 1
        )
        if np.count_nonzero(self.primitive_face_mask) != 6:
            raise RuntimeError("The primitive VdW shell must contain 6 bonds.")
        # One member of every +/- neighbour pair: the lexicographically
        # positive one.
        keys = self.neighbour_indices @ np.array([9.0, 3.0, 1.0])
        self.canonical_half_mask = keys > 0
        if np.count_nonzero(self.canonical_half_mask) != 13:
            raise RuntimeError("The canonical half shell must hold 13 bonds.")

        self.vdw_neighbour_mode = str(phys.vdw_neighbour_mode)
        if self.vdw_neighbour_mode not in ("primitive6", "all26"):
            raise ValueError("vdw_neighbour_mode must be primitive6 or all26.")
        self.vdw_diagnostics = bool(phys.vdw_diagnostics)

        # Voxel quadrature, in the difference-multiplicity form.
        self.vdw_scheme = str(phys.vdw_scheme)
        if self.vdw_scheme not in ("voxel", "richardson", "continuum"):
            raise ValueError(
                "vdw_scheme must be voxel, richardson or continuum."
            )
        self.voxel_body = str(phys.voxel_body)
        self.N_voxel = int(phys.voxel_count)
        self.richardson_base = int(phys.richardson_base)
        self.continuum_grading_levels = int(phys.continuum_grading_levels)
        self.continuum_gauss_points = int(phys.continuum_gauss_points)
        counts = [self.N_voxel]
        if self.vdw_scheme == "richardson":
            counts = [self.richardson_base, 2 * self.richardson_base]
        self._quadrature_cache = {}
        for count in counts:
            self._quadrature_cache[count] = voxel_difference_weights(
                count,
                edge_m=self.L,
                body=self.voxel_body,
                corner_radius_m=self.r_round,
            )
        (
            self.vox_difference,
            self.vox_weight,
            self.vox_volume,
            self.vox_number,
        ) = self._quadrature_cache[counts[-1]]
        self.vdw_coefficient = -(self.Aham / np.pi**2) * self.vox_volume**2
        self.vdw_floor_hits = 0

        # Brush block.
        self.brush_model = str(phys.brush_model)
        if self.brush_model not in ("derjaguin", "facet", "off"):
            raise ValueError("brush_model must be derjaguin, facet or off.")
        self.brush_neighbour_mode = str(phys.brush_neighbour_mode)
        if self.brush_neighbour_mode not in ("all26", "primitive6"):
            raise ValueError(
                "brush_neighbour_mode must be all26 or primitive6."
            )
        self.L0 = float(phys.brush_height_nm) * 1e-9
        self.graft_spacing = float(phys.grafting_spacing_nm) * 1e-9
        self.n_transverse = int(phys.derjaguin_transverse_points)
        self.brush_law = str(phys.brush_law)
        if self.brush_law not in ("adg", "exponential"):
            raise ValueError("brush_law must be adg or exponential.")
        self.steric_pressure = float(phys.steric_pressure_Pa)
        self.steric_decay = float(phys.steric_decay_nm) * 1e-9
        self.steric_reference = float(phys.steric_reference_gap_nm) * 1e-9
        # Separation beyond which the areal repulsion is treated as zero.
        # Exact for AdG (2 L0); a truncation of the exponential tail.
        self.steric_range = (
            2.0 * self.L0 if self.brush_law == "adg"
            else self.steric_reference
            + float(phys.steric_range_decays) * self.steric_decay
        )

        # At alpha=90 degrees the lattice basis rows are cube face normals.
        self.Q_SC = self.basis_unit(90.0).T
        if not np.allclose(self.Q_SC.T @ self.Q_SC, np.eye(3), atol=1e-12):
            raise RuntimeError("Q_SC is not orthogonal.")
        if np.linalg.det(self.Q_SC) < 0:
            raise RuntimeError("Q_SC must be a proper rotation.")

    # ------------------------------------------------------------------
    #  Geometry, ported from V6.20
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_gamma(gamma_deg: float) -> float:
        """Map gamma to the periodic interval [-60, 60)."""
        return (float(gamma_deg) + 60.0) % 120.0 - 60.0

    @staticmethod
    def rotation_about_field(gamma_deg: float) -> np.ndarray:
        g = np.deg2rad(gamma_deg)
        cg, sg = np.cos(g), np.sin(g)
        return np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])

    def orientation(self, gamma_deg: float) -> np.ndarray:
        """Body-to-lab rotation. gamma=0 is simple-cubic face aligned."""
        return self.rotation_about_field(self.wrap_gamma(gamma_deg)) @ self.Q_SC

    @staticmethod
    def basis_unit(alpha_deg: float) -> np.ndarray:
        alpha = np.deg2rad(alpha_deg)
        cp2 = (2.0 * np.cos(alpha) + 1.0) / 3.0
        cp = np.sqrt(max(cp2, 0.0))
        sp = np.sqrt(max(1.0 - cp**2, 0.0))
        return np.array(
            [
                [cp, sp, 0.0],
                [cp, -0.5 * sp, (np.sqrt(3.0) / 2.0) * sp],
                [cp, -0.5 * sp, -(np.sqrt(3.0) / 2.0) * sp],
            ]
        )

    def basis(self, a_nm: float, alpha_deg: float) -> np.ndarray:
        return self.basis_unit(alpha_deg) * (float(a_nm) * 1e-9)

    def neighbour_vectors(self, a_nm: float, alpha_deg: float) -> np.ndarray:
        return self.neighbour_indices @ self.basis(a_nm, alpha_deg)

    def neighbour_exact_gaps(
        self, a_nm: float, alpha_deg: float, gamma_deg: float
    ) -> np.ndarray:
        Q = self.orientation(gamma_deg)
        rv = self.neighbour_vectors(a_nm, alpha_deg)
        p = rv @ Q
        z = np.abs(p) - self.core_difference_halfwidth
        outside = np.linalg.norm(np.maximum(z, 0.0), axis=1)
        inside = np.minimum(np.max(z, axis=1), 0.0)
        return outside + inside - 2.0 * self.r_round

    def nearest_exact_gap(
        self, a_nm: float, alpha_deg: float, gamma_deg: float
    ) -> float:
        return float(
            np.min(self.neighbour_exact_gaps(a_nm, alpha_deg, gamma_deg)) * 1e9
        )

    def neighbour_sharp_gaps(
        self, a_nm: float, alpha_deg: float, gamma_deg: float
    ) -> np.ndarray:
        """Exact signed gap of the sharp cubes used by the VdW quadrature."""
        Q = self.orientation(gamma_deg)
        rv = self.neighbour_vectors(a_nm, alpha_deg)
        p = rv @ Q
        z = np.abs(p) - self.L
        outside = np.linalg.norm(np.maximum(z, 0.0), axis=1)
        inside = np.minimum(np.max(z, axis=1), 0.0)
        return outside + inside

    def nearest_sharp_gap(
        self, a_nm: float, alpha_deg: float, gamma_deg: float
    ) -> float:
        return float(
            np.min(self.neighbour_sharp_gaps(a_nm, alpha_deg, gamma_deg)) * 1e9
        )

    def a_for_exact_gap(
        self, alpha_deg: float, gamma_deg: float, gap_nm: float
    ) -> float:
        """Solve D_min_exact(a, alpha, gamma) = gap_nm."""
        gap_nm = float(gap_nm)

        def objective(a_nm: float) -> float:
            return (
                self.nearest_exact_gap(a_nm, alpha_deg, gamma_deg) - gap_nm
            )

        lower = max(0.25 * self.L_nm, 0.5)
        upper = max(3.0 * self.L_nm, self.L_nm + gap_nm + 10.0)
        f_lower = objective(lower)
        f_upper = objective(upper)
        while f_lower > 0.0 and lower > 1e-4:
            lower *= 0.5
            f_lower = objective(lower)
        while f_upper < 0.0 and upper < 20.0 * self.L_nm:
            upper *= 1.5
            f_upper = objective(upper)
        if f_lower > 0.0 or f_upper < 0.0:
            raise RuntimeError(
                f"Could not bracket a at alpha={alpha_deg}, gamma={gamma_deg}."
            )
        return float(brentq(objective, lower, upper, xtol=1e-9, rtol=1e-12))

    # ------------------------------------------------------------------
    #  Attractive van der Waals
    # ------------------------------------------------------------------
    def vdw_pair_energy_voxel_J(
        self, offset_body_m: np.ndarray, voxel_count: int | None = None
    ) -> float:
        """Voxel Hamaker energy of one pair at body-frame offset ``p``.

        Algebraically identical to the V6.20 double sum over voxel centres,
        evaluated on the difference lattice.
        """
        count = self.N_voxel if voxel_count is None else int(voxel_count)
        if count not in self._quadrature_cache:
            self._quadrature_cache[count] = voxel_difference_weights(
                count,
                edge_m=self.L,
                body=self.voxel_body,
                corner_radius_m=self.r_round,
            )
        difference, weight, volume, _ = self._quadrature_cache[count]
        separation = difference - np.asarray(offset_body_m)[None, :]
        d2 = np.einsum("ij,ij->i", separation, separation)
        clipped = d2 < self.d2_floor
        if np.any(clipped):
            self.vdw_floor_hits += int(np.count_nonzero(clipped))
        coefficient = -(self.Aham / np.pi**2) * volume**2
        return float(
            coefficient
            * np.sum(weight / np.maximum(d2, self.d2_floor) ** 3)
        )

    def vdw_pair_energy_richardson_J(
        self, offset_body_m: np.ndarray, base: int | None = None
    ) -> float:
        """h^2 Richardson extrapolation of the voxel rule from (N, 2N)."""
        n = self.richardson_base if base is None else int(base)
        coarse = self.vdw_pair_energy_voxel_J(offset_body_m, n)
        fine = self.vdw_pair_energy_voxel_J(offset_body_m, 2 * n)
        return (4.0 * fine - coarse) / 3.0

    def vdw_pair_energy_exact_J(self, offset_body_m: np.ndarray) -> float:
        """Exact continuum Hamaker energy, via :func:`cube_pair_hamaker_J`."""
        if self.voxel_body != "sharp":
            raise RuntimeError(
                "The exact reduction is for the sharp cube; use the voxel "
                "ladder for the rounded body."
            )
        return cube_pair_hamaker_J(offset_body_m, self.L, self.Aham)

    def vdw_pair_energy_J(self, offset_body_m: np.ndarray) -> float:
        """Attractive pair energy under the configured quadrature rule."""
        if self.vdw_scheme == "voxel":
            return self.vdw_pair_energy_voxel_J(offset_body_m)
        if self.vdw_scheme == "richardson":
            return self.vdw_pair_energy_richardson_J(offset_body_m)
        return self.vdw_pair_energy_exact_J(offset_body_m)

    def vdw_pair_energy_double_sum_J(self, offset_body_m: np.ndarray) -> float:
        """The literal V6.20 ``O(N^6)`` double sum, kept for verification."""
        n = self.N_voxel
        step = self.L / n
        lin = np.linspace(
            -self.L / 2.0 + step / 2.0, self.L / 2.0 - step / 2.0, n
        )
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        delta = centres[:, None, :] - centres[None, :, :]
        separation = delta - np.asarray(offset_body_m)[None, None, :]
        d2 = np.einsum("...i,...i->...", separation, separation)
        coefficient = -(self.Aham / np.pi**2) * step**6
        return float(
            coefficient * np.sum(1.0 / np.maximum(d2, self.d2_floor) ** 3)
        )

    def _continuum_rule(self, levels: int, gauss_points: int):
        """Graded Gauss-Legendre rule on [-L, L] for the kernel integral.

        Subinterval edges sit on every kink of the kernel (``-L``, 0, ``L``)
        and are bisected dyadically towards each kink, because the integrand
        concentrates against the kernel boundary as the gap closes.
        """
        key = ("continuum", levels, gauss_points)
        cached = self._quadrature_cache.get(key)
        if cached is not None:
            return cached
        half = {0.0, self.L}
        step = 0.5 * self.L
        for _ in range(int(levels)):
            half.add(step)
            half.add(self.L - step)
            step *= 0.5
        right = np.array(sorted(half))
        edges = np.concatenate([-right[::-1][:-1], right])
        nodes, weights = np.polynomial.legendre.leggauss(int(gauss_points))
        centres = 0.5 * (edges[:-1] + edges[1:])
        halfwidths = 0.5 * np.diff(edges)
        x = (centres[:, None] + halfwidths[:, None] * nodes[None, :]).ravel()
        w = (halfwidths[:, None] * weights[None, :]).ravel()
        rule = (x, w * np.maximum(self.L - np.abs(x), 0.0))
        self._quadrature_cache[key] = rule
        return rule

    def vdw_pair_energy_continuum_J(
        self,
        offset_body_m: np.ndarray,
        levels: int | None = None,
        gauss_points: int | None = None,
    ) -> float:
        """Continuum limit of the sharp-cube voxel integral.

        The double volume integral is rewritten as a single convolution with
        the exact autocorrelation of the cube indicator,

            U = -(A / pi^2) * integral K(u) / |u - p|^6 d^3u,
            K(u) = prod_i max(L - |u_i|, 0),

        and evaluated by graded composite Gauss-Legendre quadrature.  This is
        the reference value the voxel ladder is measured against; it is not a
        voxel sum and shares no discretisation with one.
        """
        if self.voxel_body != "sharp":
            raise RuntimeError(
                "The kernel is the autocorrelation of the sharp cube; use "
                "Richardson extrapolation for the rounded voxel body."
            )
        levels = self.continuum_grading_levels if levels is None else levels
        gauss_points = (
            self.continuum_gauss_points if gauss_points is None
            else gauss_points
        )
        x, wk = self._continuum_rule(int(levels), int(gauss_points))
        offset = np.asarray(offset_body_m, dtype=float)
        dx = (x - offset[0]) ** 2
        dy = (x - offset[1]) ** 2
        dz = (x - offset[2]) ** 2
        plane = wk[:, None] * wk[None, :]
        total = 0.0
        for i in range(x.size):
            d2 = dx[i] + dy[:, None] + dz[None, :]
            total += wk[i] * float(np.sum(plane / d2**3))
        return float(-(self.Aham / np.pi**2) * total)

    # ------------------------------------------------------------------
    #  Alexander-de Gennes repulsion
    # ------------------------------------------------------------------
    def _adg_area_energy(self, gap_m):
        """Areal repulsion under the configured law (name kept for callers)."""
        if self.brush_law == "exponential":
            return exponential_steric_energy_J_per_m2(
                gap_m, self.steric_pressure, self.steric_decay,
                self.steric_range, self.steric_reference,
            )
        return adg_plate_energy_J_per_m2(
            gap_m, self.L0, self.graft_spacing, self.temperature
        )

    def facet_repulsion_pair_J(self, offset_body_m: np.ndarray) -> float:
        """Areal repulsion on the pair of parallel facing flat facets only."""
        p = np.abs(np.asarray(offset_body_m, dtype=float))
        total = 0.0
        facet_edge = self.L - 2.0 * self.r_round
        for axis in range(3):
            gap = p[axis] - self.L
            if gap >= self.steric_range:
                continue
            others = [i for i in range(3) if i != axis]
            area = 1.0
            for i in others:
                area *= max(0.0, facet_edge - p[i])
            if area <= 0.0:
                continue
            if gap <= 0.0:
                return np.inf
            total += area * float(self._adg_area_energy(gap))
        return total

    def closest_approach_normal(self, offset_body_m: np.ndarray):
        """Direction and value of the exact closest approach of the pair.

        For two co-oriented rounded cubes the pair separation equals the
        distance from ``p`` to the Minkowski difference body, a box of
        half-width ``L - 2r`` dilated by ``2r``.  The closest point on that
        box is the componentwise soft threshold of ``p``, so the
        closest-approach unit normal and the exact gap both follow in closed
        form and agree with ``neighbour_exact_gaps``.
        """
        p = np.asarray(offset_body_m, dtype=float)
        z = np.sign(p) * np.maximum(
            np.abs(p) - self.core_difference_halfwidth, 0.0
        )
        norm = float(np.linalg.norm(z))
        if norm == 0.0:
            return None, -2.0 * self.r_round
        return z / norm, norm - 2.0 * self.r_round

    def _transverse_window(self, normal, lateral, along, points_per_axis):
        """Grid on the plane perpendicular to ``normal``.

        A coarse pass finds the window in which the brushes can overlap at
        all, then the requested resolution is spent on that window only.  The
        AdG term is C1 at its cut-off ``gap = 2 L0``, so the midpoint rule on
        the window converges at second order.  The exponential law is only C0
        at its cut-off (a pressure step of ``P_ref exp(-(D_cut - D_ref)/
        lambda)``, 3e-4 P_ref at the fitted 8 lambda), which is negligible.

        The coarse pass is centred on the transverse position of a
        closest-approach point (which one is explained below).  Centring it
        there guarantees that the coarse grid contains at least one live cell
        however small the contact patch is; a grid centred on the body
        instead can step straight over a patch and report no brush contact at
        all.
        """
        e1, e2 = orthonormal_frame(normal)
        reach = (np.sqrt(3.0) * self.box_halfwidth + self.r_round) * 1.000001
        # Centre on the midpoint of the contact region, (p - z) / 2, where z
        # is the soft-threshold separating vector.  For a face contact that
        # is the centre of the overlapping facets, for an edge contact the
        # middle of the overlapping edge segment, for a corner the corner.
        # It is always a closest-approach point (so the first coarse pass has
        # a live node) and, unlike the support point hb*sign(n) + r*n, it is
        # continuous in p: sign(n_i) jumps from 0 to +/-1 exactly when a
        # contact turns from face-type to edge-type, which moved the window
        # by hb and restarted the search at the degenerate contacts the
        # relaxed twist tends to sit on, putting ~1e-3 kBT jumps into E.
        offset = along * normal + lateral
        separation = np.sign(offset) * np.maximum(
            np.abs(offset) - self.core_difference_halfwidth, 0.0
        )
        middle = 0.5 * (offset - separation)
        centre = middle - float(np.dot(middle, normal)) * normal
        c1 = float(np.dot(centre, e1))
        c2 = float(np.dot(centre, e2))
        coarse = 24
        lo1, hi1 = c1 - reach, c1 + reach
        lo2, hi2 = c2 - reach, c2 + reach
        # Two tightening passes.  Each samples the current window on a
        # node-centred grid, keeps the cells where the brushes overlap, and
        # shrinks to their bounding box with a one-cell pad.  The
        # closest-approach point is always appended, so the live set is never
        # empty and the window cannot collapse away from the true contact.
        for _ in range(2):
            s1 = np.append(np.linspace(lo1, hi1, coarse), c1)
            s2 = np.append(np.linspace(lo2, hi2, coarse), c2)
            g1, g2 = np.meshgrid(s1, s2, indexing="ij")
            gap = along + self._local_gap(
                normal, e1, e2, g1.ravel(), g2.ravel(), lateral
            )
            live = np.isfinite(gap) & (gap < self.steric_range)
            if not np.any(live):
                return None
            pad1 = (hi1 - lo1) / (coarse - 1)
            pad2 = (hi2 - lo2) / (coarse - 1)
            lo1 = float(np.min(g1.ravel()[live])) - pad1
            hi1 = float(np.max(g1.ravel()[live])) + pad1
            lo2 = float(np.min(g2.ravel()[live])) - pad2
            hi2 = float(np.max(g2.ravel()[live])) + pad2
        n = int(points_per_axis)
        f1 = lo1 + (np.arange(n) + 0.5) / n * (hi1 - lo1)
        f2 = lo2 + (np.arange(n) + 0.5) / n * (hi2 - lo2)
        h1, h2 = np.meshgrid(f1, f2, indexing="ij")
        cell_area = ((hi1 - lo1) / n) * ((hi2 - lo2) / n)
        return e1, e2, h1.ravel(), h2.ravel(), cell_area

    def _local_gap(self, normal, e1, e2, c1, c2, lateral):
        """Surface-to-surface separation along ``normal`` at offsets (c1, c2).

        The two bodies are identical and co-oriented, so for a line parallel
        to ``normal`` at transverse position ``q`` the near body is entered at
        ``t_hi(q)`` and the far body at ``t_lo(q - q_lateral) + p . normal``.
        """
        points = c1[:, None] * e1[None, :] + c2[:, None] * e2[None, :]
        lo_far, _, hit_far = rounded_cube_chord_bounds(
            normal,
            points - lateral[None, :],
            self.box_halfwidth,
            self.r_round,
        )
        _, hi_near, hit_near = rounded_cube_chord_bounds(
            normal, points, self.box_halfwidth, self.r_round
        )
        hit = hit_far & hit_near
        return np.where(hit, lo_far - hi_near, np.inf)

    def derjaguin_repulsion_pair_J(self, offset_body_m: np.ndarray) -> float:
        """AdG integrated over the plane of closest approach of one bond."""
        p = np.asarray(offset_body_m, dtype=float)
        normal, gap_exact = self.closest_approach_normal(p)
        if normal is None or gap_exact <= 0.0:
            return np.inf
        # Exact rejection: every local separation is at least the pair gap.
        if gap_exact >= self.steric_range:
            return 0.0
        along = float(np.dot(p, normal))
        lateral = p - along * normal
        window = self._transverse_window(
            normal, lateral, along, self.n_transverse
        )
        if window is None:
            return 0.0
        e1, e2, c1, c2, cell_area = window
        gap = along + self._local_gap(normal, e1, e2, c1, c2, lateral)
        if np.any(gap <= 0.0):
            return np.inf
        return float(cell_area * np.sum(self._adg_area_energy(gap)))

    def repulsion_pair_J(self, offset_body_m: np.ndarray) -> float:
        if self.brush_model == "off":
            return 0.0
        if self.brush_model == "facet":
            return self.facet_repulsion_pair_J(offset_body_m)
        return self.derjaguin_repulsion_pair_J(offset_body_m)

    # ------------------------------------------------------------------
    #  Energies
    # ------------------------------------------------------------------
    def energy_components(
        self, a_nm: float, alpha_deg: float, gamma_deg: float
    ) -> dict:
        """Energy components in kBT per particle."""
        Q = self.orientation(gamma_deg)
        rv = self.neighbour_vectors(a_nm, alpha_deg)
        dv = np.linalg.norm(rv, axis=1)

        Ez = -self.mu_magnitude * self.B_field
        mu_body = Q.T @ self.mu_hat
        mx, my, mz = mu_body
        Ea = -self.K_ani * self.volume * (
            (mx * my) ** 2 + (mx * mz) ** 2 + (my * mz) ** 2
        )

        mu2 = self.mu_magnitude**2
        Edd = 0.5 * np.sum(
            self.Cmag
            * (mu2 / dv**3 - 3.0 * (rv @ self.mu_vec) ** 2 / dv**5)
        )

        rv_body = rv @ Q
        if self.vdw_diagnostics or self.vdw_neighbour_mode == "all26":
            pair_vdw = np.array(
                [self.vdw_pair_energy_J(offset) for offset in rv_body]
            )
            Evdw_per_neighbour = 0.5 * pair_vdw
            Evdw_face6 = float(
                np.sum(Evdw_per_neighbour[self.primitive_face_mask])
            )
            Evdw_full26 = float(np.sum(Evdw_per_neighbour))
            Evdw_tail20 = Evdw_full26 - Evdw_face6
        else:
            Evdw_face6 = 0.5 * float(
                np.sum(
                    [
                        self.vdw_pair_energy_J(offset)
                        for offset, use in zip(
                            rv_body, self.primitive_face_mask
                        )
                        if use
                    ]
                )
            )
            Evdw_full26 = np.nan
            Evdw_tail20 = np.nan
        Evdw_main = (
            Evdw_face6 if self.vdw_neighbour_mode == "primitive6"
            else Evdw_full26
        )

        if self.brush_model == "off":
            Erep = 0.0
        else:
            mask = (
                self.primitive_face_mask & self.canonical_half_mask
                if self.brush_neighbour_mode == "primitive6"
                else self.canonical_half_mask
            )
            # Each +/- neighbour pair contributes twice to the 26-bond sum and
            # is halved again by the per-particle convention, so summing the
            # 13 canonical bonds once is exactly 0.5 * sum over 26.
            Erep = float(
                np.sum(
                    [
                        self.repulsion_pair_J(offset)
                        for offset, use in zip(rv_body, mask)
                        if use
                    ]
                )
            )

        result = {
            "Zeeman": Ez / self.kT,
            "Anisotropy": Ea / self.kT,
            "Dipole": Edd / self.kT,
            "VdW": Evdw_main / self.kT,
            "VdW_face6": Evdw_face6 / self.kT,
            "VdW_full26": Evdw_full26 / self.kT,
            "VdW_tail20": Evdw_tail20 / self.kT,
            "Steric": Erep / self.kT,
        }
        result["Interaction_legacy"] = result["Dipole"] + result["VdW"]
        result["Interaction"] = (
            result["Dipole"] + result["VdW"] + result["Steric"]
        )
        result["Total"] = (
            result["Zeeman"]
            + result["Anisotropy"]
            + result["Dipole"]
            + result["VdW"]
            + result["Steric"]
        )
        return result

    def state_at_gap(
        self, alpha_deg: float, gamma_deg: float, gap_nm: float
    ) -> tuple[float, dict]:
        a_nm = self.a_for_exact_gap(alpha_deg, gamma_deg, gap_nm)
        return a_nm, self.energy_components(a_nm, alpha_deg, gamma_deg)

    def interaction_at_gap(
        self, alpha_deg: float, gamma_deg: float, gap_nm: float
    ) -> float:
        return self.state_at_gap(alpha_deg, gamma_deg, gap_nm)[1][
            "Interaction"
        ]

    # ------------------------------------------------------------------
    #  Gap relaxation
    # ------------------------------------------------------------------
    def relax_gap(
        self,
        alpha_deg: float,
        gamma_deg: float,
        gap_min_nm: float = 0.05,
        gap_max_nm: float | None = None,
        grid_points: int = 48,
    ) -> dict:
        """Relax the nearest-neighbour gap at fixed (alpha, gamma).

        Returns the outermost local minimum of ``Interaction`` in the gap, the
        barrier that separates it from the contact branch, and whether that
        local minimum is also the global one on the scanned window.  The
        window ends at ``2 L0``, above which the brush term is identically
        zero and the energy is monotonically decreasing in the gap.
        """
        if self.brush_model == "off":
            raise RuntimeError("relax_gap needs a brush model.")
        if self.brush_law == "adg":
            if gap_max_nm is None:
                gap_max_nm = self.steric_range * 1e9
            # Geometric in the compression 2 L0 - D, not uniform in D.  The
            # brush switches on quadratically at 2 L0, so a stiff brush puts
            # the minimum within a few per mille of the cut-off, where a
            # uniform grid in D cannot place an interior triple and the
            # minimum is missed.
            compression = np.geomspace(
                1e-4 * float(gap_max_nm),
                float(gap_max_nm) - float(gap_min_nm),
                int(grid_points),
            )
            gaps = (float(gap_max_nm) - compression)[::-1]
        else:
            # A soft law has no cut-off to hug, so the minimum can sit anywhere
            # the pressure balances the attraction.  Uniform in D over a window
            # that reaches well past the reference gap; the far tail beyond a
            # few decay lengths only lowers the attraction, so it cannot hold a
            # minimum.
            if gap_max_nm is None:
                gap_max_nm = (
                    self.steric_reference + 8.0 * self.steric_decay
                ) * 1e9
            gaps = np.linspace(
                float(gap_min_nm), float(gap_max_nm), int(grid_points)
            )
        energies = np.array(
            [
                self.interaction_at_gap(alpha_deg, gamma_deg, gap)
                for gap in gaps
            ]
        )
        finite = np.isfinite(energies)
        interior = np.where(
            finite[1:-1]
            & (energies[1:-1] <= energies[:-2])
            & (energies[1:-1] <= energies[2:])
        )[0] + 1
        if interior.size == 0:
            return {
                "converged": False,
                "reason": "no interior local minimum on the gap window",
                "gap_grid_nm": gaps,
                "interaction_grid_kBT": energies,
            }
        index = int(interior[-1])
        refined = minimize_scalar(
            lambda gap: self.interaction_at_gap(alpha_deg, gamma_deg, gap),
            bounds=(gaps[index - 1], gaps[index + 1]),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 60},
        )
        gap_star = float(refined.x)
        a_star = self.a_for_exact_gap(alpha_deg, gamma_deg, gap_star)
        components = self.energy_components(a_star, alpha_deg, gamma_deg)

        barrier_gap = np.nan
        barrier_height = np.nan
        left = np.where(
            finite[1:index]
            & (energies[1:index] >= energies[:index - 1])
            & (energies[1:index] >= energies[2:index + 1])
        )[0] + 1
        if left.size:
            top = int(left[-1])
            refined_top = minimize_scalar(
                lambda gap: -self.interaction_at_gap(
                    alpha_deg, gamma_deg, gap
                ),
                bounds=(gaps[top - 1], gaps[top + 1]),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 60},
            )
            barrier_gap = float(refined_top.x)
            barrier_height = float(
                -refined_top.fun - components["Interaction"]
            )
        return {
            "converged": True,
            "gap_nm": gap_star,
            "a_nm": a_star,
            "components": components,
            "barrier_gap_nm": barrier_gap,
            "barrier_height_kBT": barrier_height,
            "contact_branch_energy_kBT": float(energies[0]),
            "is_global_on_window": bool(
                components["Interaction"] <= np.nanmin(energies[finite])
                + 1e-9
            ),
            "gap_grid_nm": gaps,
            "interaction_grid_kBT": energies,
        }

    def relax_gap_and_twist(
        self,
        alpha_deg: float,
        gamma_grid: np.ndarray | None = None,
        gap_seed_nm: float | None = None,
        passes: int = 3,
        grid_points: int = 40,
    ) -> dict:
        """Coordinate descent over (gamma, gap) at fixed alpha.

        The twist and the gap are relaxed alternately.  Each pass minimises
        ``Interaction`` over gamma at fixed gap, then over the gap at fixed
        gamma.  ``passes=3`` is enough for the V6.20 landscape: the two
        coordinates are nearly separable because the twist changes the gap
        geometry only weakly.
        """
        if gamma_grid is None:
            gamma_grid = np.linspace(-60.0, 60.0, 13, endpoint=False)
        gap = (
            float(gap_seed_nm)
            if gap_seed_nm is not None
            else (
                self.steric_reference * 1e9
                if self.brush_law == "exponential" and self.steric_reference
                else 0.75 * min(self.steric_range, 2.0 * self.L0) * 1e9
            )
        )
        gamma = float(gamma_grid[0])
        history = []
        for _ in range(int(passes)):
            values = [
                self.interaction_at_gap(alpha_deg, float(g), gap)
                for g in gamma_grid
            ]
            seed = float(gamma_grid[int(np.argmin(values))])
            step = 120.0 / len(gamma_grid)
            refined = minimize_scalar(
                lambda g: self.interaction_at_gap(
                    alpha_deg, self.wrap_gamma(g), gap
                ),
                bounds=(seed - step, seed + step),
                method="bounded",
                options={"xatol": 2e-4, "maxiter": 60},
            )
            gamma = self.wrap_gamma(float(refined.x))
            # Prefer the negative member of a mirror-degenerate pair, as in
            # V6.20.
            mirror = self.interaction_at_gap(alpha_deg, -gamma, gap)
            here = self.interaction_at_gap(alpha_deg, gamma, gap)
            if abs(mirror - here) < 1e-7 and -gamma < gamma:
                gamma = self.wrap_gamma(-gamma)
            if abs(gamma) < 1e-4:
                gamma = 0.0
            relaxed = self.relax_gap(
                alpha_deg, gamma, grid_points=grid_points
            )
            if not relaxed["converged"]:
                return {**relaxed, "gamma_deg": gamma}
            gap = relaxed["gap_nm"]
            history.append((gamma, gap, relaxed["components"]["Interaction"]))
        relaxed["gamma_deg"] = gamma
        relaxed["history"] = history
        return relaxed
