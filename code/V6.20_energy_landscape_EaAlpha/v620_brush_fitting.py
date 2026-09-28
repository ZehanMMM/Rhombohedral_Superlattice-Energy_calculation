"""Minimisation helpers shared by the two V6.20 brush analysis runners.

Two ensembles are provided.

``fixed_gap_alpha_scan``
    The legacy V6.20 ensemble.  The nearest-neighbour gap is pinned at the
    value calibrated from the experimental reference state, ``a`` follows from
    that constraint at every (alpha, gamma), and ``Interaction`` is minimised
    over gamma and then over alpha.  The steric term is identically zero on
    this path, which is why V6.20 could omit it.

``free_gap_alpha_scan``
    The extended ensemble.  The gap is a free coordinate relaxed against the
    brush, so (a, alpha, gamma) are all predicted.  Gamma and the gap are
    relaxed by coordinate descent; ``passes=3`` is checked against a full
    nested relaxation in ``run_brush_refit.py``.

Both follow the V6.20 conventions: gamma is periodic on [-60, 60), the
negative member of a mirror-degenerate pair is reported, and the alpha
minimum is refined by a bounded scalar minimisation between the neighbouring
grid points.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

DEFAULT_ALPHAS = np.linspace(60.0, 90.0, 61)
DEFAULT_GAMMAS = np.linspace(-60.0, 60.0, 31, endpoint=False)


def _mirror_preference(model, alpha_deg, gamma_deg, evaluate):
    """Pick the negative member of a mirror-degenerate gamma pair."""
    gamma = model.wrap_gamma(gamma_deg)
    here = evaluate(alpha_deg, gamma)
    mirror = evaluate(alpha_deg, model.wrap_gamma(-gamma))
    if abs(mirror - here) < 1e-7 and -gamma < gamma:
        gamma = model.wrap_gamma(-gamma)
    if abs(gamma) < 1e-4:
        gamma = 0.0
    return gamma


def minimise_twist(model, alpha_deg, gap_nm, gammas=DEFAULT_GAMMAS):
    """Minimise Interaction over gamma at fixed alpha and fixed gap."""

    def evaluate(alpha, gamma):
        return model.interaction_at_gap(alpha, gamma, gap_nm)

    values = [evaluate(alpha_deg, float(g)) for g in gammas]
    seed = float(gammas[int(np.argmin(values))])
    step = 120.0 / len(gammas)
    refined = minimize_scalar(
        lambda g: evaluate(alpha_deg, model.wrap_gamma(g)),
        bounds=(seed - step, seed + step),
        method="bounded",
        options={"xatol": 2e-4, "maxiter": 80},
    )
    gamma = _mirror_preference(
        model, alpha_deg, float(refined.x), evaluate
    )
    a_nm, components = model.state_at_gap(alpha_deg, gamma, gap_nm)
    return gamma, a_nm, components


def fixed_gap_alpha_scan(
    model, gap_nm, alphas=DEFAULT_ALPHAS, gammas=DEFAULT_GAMMAS
):
    """V6.20 constant-gap scan, with the alpha minimum refined."""
    alphas = np.asarray(alphas, dtype=float)
    rows = []
    for alpha in alphas:
        gamma, a_nm, components = minimise_twist(model, alpha, gap_nm, gammas)
        rows.append((alpha, gamma, a_nm, components))
    energies = np.array([row[3]["Interaction"] for row in rows])
    index = int(np.argmin(energies))

    def objective(alpha):
        return minimise_twist(model, float(alpha), gap_nm, gammas)[2][
            "Interaction"
        ]

    if 0 < index < len(alphas) - 1:
        refined = minimize_scalar(
            objective,
            bounds=(float(alphas[index - 1]), float(alphas[index + 1])),
            method="bounded",
            options={"xatol": 2e-4, "maxiter": 40},
        )
        alpha_star = float(refined.x)
    else:
        alpha_star = float(alphas[index])
    gamma_star, a_star, components_star = minimise_twist(
        model, alpha_star, gap_nm, gammas
    )
    if components_star["Interaction"] > energies[index] + 1e-9:
        alpha_star = float(alphas[index])
        gamma_star, a_star, components_star = (
            rows[index][1],
            rows[index][2],
            rows[index][3],
        )
    return {
        "ensemble": "fixed_gap",
        "gap_nm": float(gap_nm),
        "alphas": alphas,
        "gamma_opt": np.array([row[1] for row in rows]),
        "a_opt": np.array([row[2] for row in rows]),
        "components": [row[3] for row in rows],
        "interaction": energies,
        "alpha_star": alpha_star,
        "gamma_star": float(gamma_star),
        "a_star": float(a_star),
        "gap_star_nm": float(gap_nm),
        "components_star": components_star,
    }


def free_gap_alpha_scan(
    model,
    alphas=DEFAULT_ALPHAS,
    gammas=DEFAULT_GAMMAS,
    passes: int = 3,
    grid_points: int = 40,
    verbose: bool = False,
):
    """Relax gamma and the gap at every alpha, then refine the alpha minimum.

    The gap of the previous alpha seeds the next one, which keeps the
    coordinate descent on the same branch across the scan.
    """
    alphas = np.asarray(alphas, dtype=float)
    rows = []
    seed = None
    for alpha in alphas:
        relaxed = model.relax_gap_and_twist(
            float(alpha),
            gamma_grid=gammas,
            gap_seed_nm=seed,
            passes=passes,
            grid_points=grid_points,
        )
        if not relaxed.get("converged", False):
            rows.append((alpha, np.nan, np.nan, np.nan, None, relaxed))
            continue
        seed = relaxed["gap_nm"]
        rows.append(
            (
                alpha,
                relaxed["gamma_deg"],
                relaxed["a_nm"],
                relaxed["gap_nm"],
                relaxed["components"],
                relaxed,
            )
        )
        if verbose:
            print(
                f"  alpha={alpha:5.1f} gamma={relaxed['gamma_deg']:7.2f} "
                f"gap={relaxed['gap_nm']:6.3f} a={relaxed['a_nm']:7.3f} "
                f"Eint={relaxed['components']['Interaction']:8.3f} kBT",
                flush=True,
            )
    energies = np.array(
        [
            row[4]["Interaction"] if row[4] is not None else np.nan
            for row in rows
        ]
    )
    if np.all(np.isnan(energies)):
        raise RuntimeError("No alpha produced a relaxed state.")
    index = int(np.nanargmin(energies))

    def objective(alpha):
        relaxed = model.relax_gap_and_twist(
            float(alpha),
            gamma_grid=gammas,
            gap_seed_nm=float(rows[index][3]),
            passes=passes,
            grid_points=grid_points,
        )
        if not relaxed.get("converged", False):
            return np.inf
        return relaxed["components"]["Interaction"]

    alpha_star = float(alphas[index])
    if 0 < index < len(alphas) - 1:
        refined = minimize_scalar(
            objective,
            bounds=(float(alphas[index - 1]), float(alphas[index + 1])),
            method="bounded",
            options={"xatol": 1e-3, "maxiter": 30},
        )
        if refined.fun <= energies[index] + 1e-9:
            alpha_star = float(refined.x)
    best = model.relax_gap_and_twist(
        alpha_star,
        gamma_grid=gammas,
        gap_seed_nm=float(rows[index][3]),
        passes=passes,
        grid_points=grid_points,
    )
    if (
        not best.get("converged", False)
        or best["components"]["Interaction"] > energies[index] + 1e-9
    ):
        alpha_star = float(alphas[index])
        best = rows[index][5]
    return {
        "ensemble": "free_gap",
        "alphas": alphas,
        "gamma_opt": np.array([row[1] for row in rows]),
        "a_opt": np.array([row[2] for row in rows]),
        "gap_opt": np.array([row[3] for row in rows]),
        "components": [row[4] for row in rows],
        "interaction": energies,
        "alpha_star": alpha_star,
        "gamma_star": float(best["gamma_deg"]),
        "a_star": float(best["a_nm"]),
        "gap_star_nm": float(best["gap_nm"]),
        "components_star": best["components"],
        "barrier_gap_nm": float(best.get("barrier_gap_nm", np.nan)),
        "barrier_height_kBT": float(best.get("barrier_height_kBT", np.nan)),
        "relaxed_star": best,
    }


def feasible_twist_intervals(
    model, a_nm, alpha_deg, gap_nm, probe_points: int = 121
):
    """Intervals of gamma in [0, 60] where the pair gap is at least ``gap_nm``.

    At fixed ``a`` the twist is not free: rotating the particles can bring the
    surfaces closer than the reference separation.  The V6.20 landscape
    therefore relaxes gamma only inside the feasible set, which this function
    brackets on a probe grid and then closes with a root find on each edge.
    Mirror symmetry in gamma means [0, 60] covers every distinct state.
    """
    probe = np.linspace(0.0, 60.0, int(probe_points))

    def residual(gamma_deg):
        return model.nearest_exact_gap(a_nm, alpha_deg, gamma_deg) - gap_nm

    values = np.array([residual(gamma) for gamma in probe])
    feasible = values >= -2e-8
    if not np.any(feasible):
        # Tangential contact: keep the single widest-gap point if it is
        # physical, so the contact branch itself is not lost.
        peak = minimize_scalar(
            lambda gamma: -residual(gamma),
            bounds=(
                probe[max(int(np.argmax(values)) - 1, 0)],
                probe[min(int(np.argmax(values)) + 1, len(probe) - 1)],
            ),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 60},
        )
        return (
            [(float(peak.x), float(peak.x))] if -peak.fun >= -1e-6 else []
        )

    def edge(inside_index, outside_index):
        """Close the interval on a sign change, or fall back to the node.

        A probe point can be feasible on the ``-2e-8`` tolerance while its
        residual is already negative, in which case the two bracket values
        share a sign and there is no root to find.
        """
        inside = probe[inside_index]
        if values[inside_index] <= 0.0:
            return float(inside)
        return float(
            brentq(residual, inside, probe[outside_index], xtol=1e-9)
        )

    indices = np.flatnonzero(feasible)
    blocks = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    intervals = []
    for block in blocks:
        low, high = int(block[0]), int(block[-1])
        left = 0.0 if low == 0 else edge(low, low - 1)
        right = 60.0 if high == len(probe) - 1 else edge(high, high + 1)
        intervals.append((min(left, right), max(left, right)))
    return intervals


def relaxed_state_at_fixed_a(model, a_nm, alpha_deg, gap_nm, xatol=0.06):
    """Minimise Interaction over the feasible twists at fixed (a, alpha)."""
    intervals = feasible_twist_intervals(model, a_nm, alpha_deg, gap_nm)
    if not intervals:
        return np.nan, None
    cache = {}

    def evaluate(gamma_deg):
        key = round(float(gamma_deg), 10)
        if key not in cache:
            cache[key] = model.energy_components(a_nm, alpha_deg, key)
        return cache[key]

    candidates = []
    for left, right in intervals:
        candidates.extend([left, right])
        if right - left > 2e-4:
            refined = minimize_scalar(
                lambda gamma: evaluate(gamma)["Interaction"],
                bounds=(left, right),
                method="bounded",
                options={"xatol": xatol, "maxiter": 30},
            )
            candidates.append(float(refined.x))

    best_gamma, best = None, None
    for gamma_deg in candidates:
        if model.nearest_exact_gap(a_nm, alpha_deg, gamma_deg) < gap_nm - 5e-7:
            continue
        components = evaluate(gamma_deg)
        if best is None or components["Interaction"] < best["Interaction"]:
            best_gamma, best = float(gamma_deg), components
    return best_gamma, best


def calibrate_brush_height(
    build_model,
    target_gap_nm: float,
    alpha_deg: float,
    gamma_deg: float | None = None,
    bracket_nm=(1.2, 3.2),
    gammas=DEFAULT_GAMMAS,
):
    """Solve for the brush height that puts the relaxed gap on a target.

    ``build_model(brush_height_nm)`` must return a model.  With gamma free the
    relaxation is done by coordinate descent; with ``gamma_deg`` given the
    twist is held fixed, which is the cheaper single-state calibration.
    """

    def relaxed_gap(height_nm: float) -> float:
        model = build_model(float(height_nm))
        if gamma_deg is None:
            relaxed = model.relax_gap_and_twist(
                alpha_deg, gamma_grid=gammas, passes=2
            )
        else:
            relaxed = model.relax_gap(alpha_deg, gamma_deg)
        if not relaxed.get("converged", False):
            raise RuntimeError(
                f"No relaxed state at brush height {height_nm} nm."
            )
        return relaxed["gap_nm"]

    objective = lambda height: relaxed_gap(height) - float(target_gap_nm)
    low, high = bracket_nm
    return float(brentq(objective, low, high, xtol=1e-5, rtol=1e-10))
