"""Fit a repulsion term so the converged-vdW V6.20 model hits the experiment.

The target is the 16 nm experimental state, a = 20.31 nm and alpha = 74.0 deg,
with the sample magnetisation Ms = 285 kA/m and the exact continuum Hamaker
attraction.  The repulsion is a soft exponential steric pressure carried on
the same surface-element areas as the brush,

    P(D) = P_ref exp(-(D - D_ref) / lambda),   D_ref = 3.465 nm,

and with the gap free both a and alpha are outputs, so two parameters meet two
targets.  The fit condition went through three versions.  All three are kept
because each failure is informative:

1. ``fit_exponential``: stationarity in (alpha, D), dE/dD = dE/dalpha = 0 with
   a(alpha, gamma, D) matched to 20.31 nm.  Here a is solved from a
   nearest-gap constraint that switches bond family as gamma turns, so the
   residuals have kinks and the Newton iteration stalls.
2. ``fit_exponential_fixed_a``: stationarity in (a, alpha), dE/da = dE/dalpha
   = 0 at a = 20.31 nm.  No gap constraint, quadratic convergence, but
   E_min(a, alpha) is only piecewise smooth (see the robust-fit section) and
   a vanishing central difference can sit on a slope break rather than at the
   bottom of the valley.
3. ``fit_valley_vertex``, the one used: the vertex of a parabola through the
   valley floor min_a E over alpha = 74 +/- 1.2 deg is put on the target.  The
   0.4 deg stencil averages over the kinks.

In every version gamma is re-minimised on the *total* energy at every
evaluation.  At the relaxed twist the nearest contact is degenerate (twelve of
the 26 bonds sit at the nearest gap), and derivatives taken at a frozen gamma
are wrong there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

import v620_brush_vdw as bv
from v620_brush_fitting import minimise_twist

TARGET_A_NM = bv.EXP_DATA[16.0]["d"]
TARGET_ALPHA_DEG = bv.EXP_DATA[16.0]["alpha"]
GAMMAS = np.linspace(-60.0, 60.0, 19, endpoint=False)


@dataclass
class StationarityPoint:
    gap_nm: float
    gamma_deg: float
    a_nm: float
    dE_dD: float          # kBT / nm
    dE_dalpha: float      # kBT / deg
    components: dict


def target_gap_nm(model: bv.BrushRhombohedralSL) -> float:
    """Gap that puts a = 20.31 nm at alpha = 74 deg, with the twist relaxed.

    The twist relaxes on the attraction alone here; the fitted repulsion is
    required to leave it where it is (checked in the report).  This is the
    gap the experiment implies once gamma is not frozen at the legacy -15 deg.
    """

    def residual(gap_nm):
        _, a_nm, _ = minimise_twist(model, TARGET_ALPHA_DEG, gap_nm, GAMMAS)
        return a_nm - TARGET_A_NM

    return float(brentq(residual, 2.5, 4.5, xtol=1e-7))


def stationarity(
    model: bv.BrushRhombohedralSL,
    alpha_deg: float,
    gap_nm: float,
    step_gap_nm: float = 4e-3,
    step_alpha_deg: float = 0.15,
) -> StationarityPoint:
    """Central differences of the twist-relaxed total energy."""

    def energy(alpha, gap):
        return minimise_twist(model, alpha, gap, GAMMAS)[2]["Interaction"]

    gamma, a_nm, components = minimise_twist(model, alpha_deg, gap_nm, GAMMAS)
    dD = (
        energy(alpha_deg, gap_nm + step_gap_nm)
        - energy(alpha_deg, gap_nm - step_gap_nm)
    ) / (2.0 * step_gap_nm)
    dA = (
        energy(alpha_deg + step_alpha_deg, gap_nm)
        - energy(alpha_deg - step_alpha_deg, gap_nm)
    ) / (2.0 * step_alpha_deg)
    return StationarityPoint(gap_nm, gamma, a_nm, dD, dA, components)


# ----------------------------------------------------------------------
#  Stationarity in (a, alpha): version 2, and the solver behind version 3
# ----------------------------------------------------------------------
# In (alpha, D) coordinates a is solved from the nearest-gap constraint, and
# that constraint switches between bond families as gamma turns, so E(gamma)
# has kinks and gamma* can jump between branches inside a finite-difference
# stencil.  A Newton iteration on those residuals stalls.  The free-gap
# problem is an unconstrained minimum over (a, alpha, gamma), so the same
# stationary point can be written in (a, alpha) with gamma relaxed at fixed
# (a, alpha).  There is no gap constraint there: every bond gap is a smooth
# function of gamma, the dipole term does not depend on gamma at all, and the
# target a = 20.31 nm is imposed exactly instead of through an extra unknown.
GAMMAS_FINE = np.linspace(-60.0, 60.0, 61, endpoint=False)
PHYSICAL_GAP_FLOOR_NM = 1.0


def vdw_sharp_gap_nm(model, a_nm, alpha_deg, gamma_deg):
    """Smallest gap between the sharp cubes of the bonds in the vdW sum."""
    gaps = model.neighbour_sharp_gaps(a_nm, alpha_deg, gamma_deg)
    if model.vdw_neighbour_mode == "primitive6":
        gaps = gaps[model.primitive_face_mask]
    return float(np.min(gaps) * 1e9)


def relax_twist_at_fixed_a(model, a_nm, alpha_deg, gammas=GAMMAS_FINE):
    """Minimise Interaction over gamma at fixed (a, alpha).

    Twists that bring two surfaces closer than ``PHYSICAL_GAP_FLOOR_NM`` are
    excluded, and the rule is applied to two sets of surfaces.  The rounded
    cubes carry the geometry and the steric term.  The exact Hamaker integral
    uses the sharp cubes of edge L instead (the V6.20 convention).  The sharp
    cubes stick out beyond the rounded ones at every edge and corner, so the
    sharp cubes of an edge contact already overlap below a rounded gap of
    2r(sqrt(2) - 1) = 1.24 nm, and those of a corner contact below
    2r(sqrt(3) - 1) = 2.20 nm.  The integral diverges there (-2e17 kBT at
    gamma = +/-60 deg in the first landscape run), while the soft repulsion
    stays finite, so without the second check such a twist wins.  Every state
    reported in R2-R5 has a rounded gap above 3.1 nm and a vdW-bond sharp gap
    above 3.5 nm, so the floor removes the artefact and nothing physical.
    """
    step = float(gammas[1] - gammas[0])

    def energy(gamma):
        gamma = model.wrap_gamma(gamma)
        if (
            model.nearest_exact_gap(a_nm, alpha_deg, gamma)
            < PHYSICAL_GAP_FLOOR_NM
            or vdw_sharp_gap_nm(model, a_nm, alpha_deg, gamma)
            < PHYSICAL_GAP_FLOOR_NM
        ):
            return np.inf, None
        components = model.energy_components(a_nm, alpha_deg, gamma)
        return components["Interaction"], components

    values = np.array([energy(g)[0] for g in gammas])
    if not np.any(np.isfinite(values)):
        raise RuntimeError(
            f"No admissible twist at a = {a_nm} nm, alpha = {alpha_deg} deg."
        )
    seed = float(gammas[int(np.nanargmin(np.where(np.isfinite(values),
                                                     values, np.nan)))])
    from scipy.optimize import minimize_scalar

    refined = minimize_scalar(
        lambda g: min(energy(g)[0], 1e12),
        bounds=(seed - step, seed + step),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 60},
    )
    gamma = model.wrap_gamma(float(refined.x))
    value, components = energy(gamma)
    if not np.isfinite(value):
        gamma = model.wrap_gamma(seed)
        value, components = energy(gamma)
    return gamma, components


@dataclass
class StationarityInA:
    a_nm: float
    alpha_deg: float
    gamma_deg: float
    gap_nm: float
    dE_da: float          # kBT / nm
    dE_dalpha: float      # kBT / deg
    components: dict


def stationarity_fixed_a(
    model,
    a_nm: float = TARGET_A_NM,
    alpha_deg: float = TARGET_ALPHA_DEG,
    step_a_nm: float = 4e-3,
    step_alpha_deg: float = 0.15,
) -> StationarityInA:
    """Central differences of min_gamma E at fixed (a, alpha)."""

    def energy(a, alpha):
        return relax_twist_at_fixed_a(model, a, alpha)[1]["Interaction"]

    gamma, components = relax_twist_at_fixed_a(model, a_nm, alpha_deg)
    dA = (
        energy(a_nm + step_a_nm, alpha_deg)
        - energy(a_nm - step_a_nm, alpha_deg)
    ) / (2.0 * step_a_nm)
    dAl = (
        energy(a_nm, alpha_deg + step_alpha_deg)
        - energy(a_nm, alpha_deg - step_alpha_deg)
    ) / (2.0 * step_alpha_deg)
    return StationarityInA(
        a_nm, alpha_deg, gamma,
        model.nearest_exact_gap(a_nm, alpha_deg, gamma),
        dA, dAl, components,
    )


# ----------------------------------------------------------------------
#  The fitted repulsion: soft exponential steric pressure on the SEI areas
# ----------------------------------------------------------------------
STERIC_REFERENCE_GAP_NM = 3.465   # amplitude is the pressure at this gap
STERIC_RANGE_DECAYS = 8.0         # truncation: e^-8 = 3e-4 of the areal law


def exponential_model(
    pressure_Pa: float,
    decay_nm: float,
    Ms: float = 285_000.0,
    area_rule: str = "derjaguin",
    L_nm: float = 16.0,
    **kwargs,
) -> bv.BrushRhombohedralSL:
    """Converged vdW + V6.20 magnetics + exponential steric repulsion."""
    phys = bv.BrushPhys(
        vdw_scheme="continuum",
        vdw_diagnostics=False,
        Ms=Ms,
        brush_model=area_rule,
        brush_law="exponential",
        steric_pressure_Pa=float(pressure_Pa),
        steric_decay_nm=float(decay_nm),
        steric_reference_gap_nm=STERIC_REFERENCE_GAP_NM,
        steric_range_decays=STERIC_RANGE_DECAYS,
        **kwargs,
    )
    return bv.BrushRhombohedralSL(L_nm, phys)


def residuals_fixed_a(x, Ms=285_000.0, area_rule="derjaguin"):
    """Scaled stationarity residuals at (a, alpha) = (20.31 nm, 74 deg).

    ``x = (log10 P_ref, lambda_nm)``.  Two equations, two unknowns; the
    target lattice parameter is imposed exactly by the coordinates.
    """
    log_pressure, decay = (float(v) for v in x)
    model = exponential_model(10.0**log_pressure, decay, Ms, area_rule)
    point = stationarity_fixed_a(model)
    return np.array([point.dE_da / 3.0, point.dE_dalpha / 0.07]), point


def fit_exponential_fixed_a(
    x0,
    Ms: float = 285_000.0,
    area_rule: str = "derjaguin",
    steps=(0.01, 0.005),
    max_iterations: int = 15,
    verbose: bool = True,
):
    """Damped Newton on the two (a, alpha) stationarity conditions."""
    x = np.asarray(x0, dtype=float)
    r, point = residuals_fixed_a(x, Ms, area_rule)
    history = [(x.copy(), r.copy(), point)]
    for iteration in range(max_iterations):
        norm = float(np.linalg.norm(r))
        if verbose:
            print(
                f"    it {iteration:2d}  P_ref={10**x[0]:.5g} Pa  "
                f"lambda={x[1]:.5f} nm  |r|={norm:.3e}  "
                f"dE/da={point.dE_da:+.2e}  dE/dalpha={point.dE_dalpha:+.2e}  "
                f"gap={point.gap_nm:.4f}  gamma*={point.gamma_deg:.3f}",
                flush=True,
            )
        if abs(point.dE_da) < 1e-3 and abs(point.dE_dalpha) < 1e-4:
            break
        jacobian = np.empty((2, 2))
        for j in range(2):
            shifted = x.copy()
            shifted[j] += steps[j]
            jacobian[:, j] = (
                residuals_fixed_a(shifted, Ms, area_rule)[0] - r
            ) / steps[j]
        delta = np.linalg.solve(jacobian, -r)
        scale = 1.0
        for _ in range(8):
            trial = x + scale * delta
            trial[1] = float(np.clip(trial[1], 0.03, 3.0))
            r_trial, point_trial = residuals_fixed_a(trial, Ms, area_rule)
            if np.linalg.norm(r_trial) < norm:
                x, r, point = trial, r_trial, point_trial
                break
            scale *= 0.5
        else:
            raise RuntimeError("Newton step failed to reduce the residual.")
        history.append((x.copy(), r.copy(), point))
    return x, point, history


# ----------------------------------------------------------------------
#  Robust fit: put the vertex of the valley floor on the target
# ----------------------------------------------------------------------
# Why the local-derivative fits above are not the final word: the Derjaguin
# construction integrates along the closest-approach normal, and that normal
# swings quickly where a contact changes from face-type to edge-type, so
# E(gamma) at fixed (a, alpha) has slope breaks and the twist-relaxed
# E_min(a, alpha) is only piecewise smooth.  A central difference that
# straddles a slope break can read zero at a point that is not a minimum;
# the refit with the continuous window centre converged to such a point and
# the independent scan then found the valley bottom 0.8 deg lower.  The
# quantity that is insensitive to kinks is the vertex of a parabola through
# the valley floor over a finite window, so that is what the fit targets.
VALLEY_ALPHAS = TARGET_ALPHA_DEG + np.array([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8,
                                             1.2])
VALLEY_GAMMAS = np.linspace(-30.0, 0.0, 31)   # mirror symmetry: gamma <= 0
VALLEY_SLOPE_NM_PER_DEG = -0.134              # a*(alpha) along the floor


def valley_vertex(model, alphas=VALLEY_ALPHAS, centre_a_nm=TARGET_A_NM,
                  verbose=False):
    """Vertex of a parabola through min_a E_min(a, alpha) over ``alphas``.

    Returns (alpha_vertex, a_vertex, rows) where rows holds
    (alpha, a*, E) for every stencil point.
    """
    from scipy.optimize import minimize_scalar

    rows = []
    for alpha in alphas:
        guess = centre_a_nm + VALLEY_SLOPE_NM_PER_DEG * (alpha - TARGET_ALPHA_DEG)
        best = minimize_scalar(
            lambda a_nm: relax_twist_at_fixed_a(
                model, a_nm, alpha, VALLEY_GAMMAS
            )[1]["Interaction"],
            bounds=(guess - 0.08, guess + 0.08),
            method="bounded",
            options={"xatol": 5e-4},
        )
        rows.append((float(alpha), float(best.x), float(best.fun)))
        if verbose:
            print(f"      alpha={alpha:.2f}  a*={best.x:.4f}  E={best.fun:.6f}",
                  flush=True)
    rows = np.array(rows)
    quadratic = np.polyfit(rows[:, 0], rows[:, 2], 2)
    if quadratic[0] <= 0:
        raise RuntimeError("Valley floor is not convex on the stencil.")
    alpha_vertex = float(-quadratic[1] / (2.0 * quadratic[0]))
    line = np.polyfit(rows[:, 0], rows[:, 1], 1)
    a_vertex = float(np.polyval(line, alpha_vertex))
    return alpha_vertex, a_vertex, rows


def fit_valley_vertex(x0, Ms=285_000.0, area_rule="derjaguin",
                      steps=(0.02, 0.02), max_iterations=8, verbose=True):
    """Broyden-updated Newton on (alpha_vertex - 74, a_vertex - 20.31).

    Each residual costs one valley stencil (seven minimisations over a with
    the twist relaxed inside), so the Jacobian is built once by forward
    differences and then updated with Broyden's rank-one formula.
    """
    def residual(x):
        model = exponential_model(10.0 ** x[0], x[1], Ms, area_rule)
        alpha_v, a_v, rows = valley_vertex(model)
        return (
            np.array([alpha_v - TARGET_ALPHA_DEG, (a_v - TARGET_A_NM) / 0.1]),
            (alpha_v, a_v, rows),
        )

    x = np.asarray(x0, dtype=float)
    r, info = residual(x)
    history = [(x.copy(), r.copy(), info)]
    if verbose:
        print(f"    it  0  P_ref={10**x[0]:.5g} Pa  lambda={x[1]:.5f} nm  "
              f"vertex=({info[0]:.3f} deg, {info[1]:.4f} nm)", flush=True)
    jacobian = np.empty((2, 2))
    for j in range(2):
        shifted = x.copy()
        shifted[j] += steps[j]
        jacobian[:, j] = (residual(shifted)[0] - r) / steps[j]
    for iteration in range(1, max_iterations + 1):
        if abs(r[0]) < 0.02 and abs(r[1]) < 0.02:    # 0.02 deg, 2 pm
            break
        delta = np.linalg.solve(jacobian, -r)
        trial = x + delta
        trial[1] = float(np.clip(trial[1], 0.05, 3.0))
        r_new, info = residual(trial)
        step = trial - x
        jacobian += np.outer(r_new - r - jacobian @ step, step) / float(
            step @ step
        )
        x, r = trial, r_new
        history.append((x.copy(), r.copy(), info))
        if verbose:
            print(f"    it {iteration:2d}  P_ref={10**x[0]:.5g} Pa  "
                  f"lambda={x[1]:.5f} nm  "
                  f"vertex=({info[0]:.3f} deg, {info[1]:.4f} nm)", flush=True)
    return x, info, history


def residuals(x, Ms=285_000.0, area_rule="derjaguin"):
    """Scaled residuals of the three fit conditions at the experimental state.

    ``x = (log10 P_ref, lambda_nm, D_t_nm)``.  The scales are the attraction's
    own derivatives at the target and a 0.05 nm tolerance on ``a``, so a unit
    residual means "as large as the thing being balanced".
    """
    log_pressure, decay, gap = (float(v) for v in x)
    model = exponential_model(10.0**log_pressure, decay, Ms, area_rule)
    point = stationarity(model, TARGET_ALPHA_DEG, gap)
    vector = np.array(
        [
            point.dE_dD / 3.0786,
            point.dE_dalpha / 0.06690,
            (point.a_nm - TARGET_A_NM) / 0.05,
        ]
    )
    return vector, point


def fit_exponential(
    x0,
    Ms: float = 285_000.0,
    area_rule: str = "derjaguin",
    steps=(0.01, 0.005, 0.002),
    max_iterations: int = 12,
    verbose: bool = True,
):
    """Damped Newton on the three stationarity conditions.

    The Jacobian is by forward differences with explicit steps, chosen well
    above the twist-refinement noise in the energy (about 1e-6 kBT) but small
    against the curvature of the residuals.
    """
    x = np.asarray(x0, dtype=float)
    r, point = residuals(x, Ms, area_rule)
    history = [(x.copy(), r.copy(), point)]
    for iteration in range(max_iterations):
        norm = float(np.linalg.norm(r))
        if verbose:
            print(
                f"    it {iteration:2d}  P_ref={10**x[0]:.5g} Pa  "
                f"lambda={x[1]:.5f} nm  D_t={x[2]:.6f} nm  "
                f"|r|={norm:.3e}  dE/dD={point.dE_dD:+.2e}  "
                f"dE/dalpha={point.dE_dalpha:+.2e}  "
                f"a={point.a_nm:.5f}  gamma*={point.gamma_deg:.3f}",
                flush=True,
            )
        if (
            abs(point.dE_dD) < 1e-3
            and abs(point.dE_dalpha) < 2e-4
            and abs(point.a_nm - TARGET_A_NM) < 5e-4
        ):
            break
        jacobian = np.empty((3, 3))
        for j in range(3):
            shifted = x.copy()
            shifted[j] += steps[j]
            jacobian[:, j] = (residuals(shifted, Ms, area_rule)[0] - r) / steps[j]
        delta = np.linalg.solve(jacobian, -r)
        scale = 1.0
        for _ in range(6):
            trial = x + scale * delta
            trial[1] = float(np.clip(trial[1], 0.03, 3.0))
            trial[2] = float(np.clip(trial[2], 2.8, 4.2))
            r_trial, point_trial = residuals(trial, Ms, area_rule)
            if np.linalg.norm(r_trial) < norm:
                x, r, point = trial, r_trial, point_trial
                break
            scale *= 0.5
        else:
            raise RuntimeError("Newton step failed to reduce the residual.")
        history.append((x.copy(), r.copy(), point))
    return x, point, history
