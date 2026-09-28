"""Fit a repulsion term so the converged model reproduces the 16 nm experiment.

Model: exact continuum Hamaker attraction (A = 2.0e-20 J), the V6.20 magnetic
block as written (Ms = 285 kA/m, K = 2.0e4 J/m^3, B = 0.5 T), and a fitted
steric repulsion.  Target: a = 20.31 nm, alpha = 74.0 deg.

Why a soft exponential and not the Alexander-de Gennes brush
------------------------------------------------------------
With the gap free, the brush has two jobs: balance the attractive force at the
observed gap, and pull the angle down from the 78.6 deg the attraction alone
selects.  AdG does the first but not the second: its minimum sits at the foot
of the brush, where the steric energy is 0.002-0.01 kBT, so its angle lever is
negligible (B4: alpha* only moves to 78.4 deg).  A soft law carries an O(1)
kBT steric energy at balance and therefore has a real angle lever.  The law is

    P(D) = P_ref exp(-(D - D_ref) / lambda),   D_ref = 3.465 nm,

carried on the same surface-element areas as the brush (``derjaguin``).
Two parameters, two targets; the fit is exactly determined.

Sections
    R1  the target and what the attraction alone does there
    R2  the fit (Broyden iteration that puts the valley-floor vertex on the
        target, see v620_repulsion_fit.fit_valley_vertex)
    R3  verification: E(D) down to near contact, a free-gap alpha scan with
        the independent (alpha, D) solver, and a fine valley scan
    R4  transfer to the other sizes with the parameters frozen

    python run_repulsion_fit.py                 # R1-R3
    python run_repulsion_fit.py --transfer      # plus R4
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import v620_brush_vdw as bv
import v620_repulsion_fit as rf
from v620_brush_fitting import fixed_gap_alpha_scan, free_gap_alpha_scan

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Starting point for the valley-vertex fit.  With the continuous window
# centre, the derivative fit at (P_ref, lambda) = (31.06 kPa, 0.541 nm) put
# the valley bottom at about 73.2 deg, and the earlier noisy fit at
# (32.14 kPa, 0.489 nm) at about 74.1 deg, so the solution is near 0.50 nm.
X0 = (np.log10(3.10e4), 0.50)
ALPHAS = np.linspace(60.0, 90.0, 31)
GAMMAS = np.linspace(-60.0, 60.0, 19, endpoint=False)
TRANSFER_SIZES_NM = (13.0, 18.0, 18.2, 19.5, 29.6)


def write_csv(name, header, rows):
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return path


def section_r1():
    print("R1  the target")
    attraction = bv.BrushRhombohedralSL(
        16.0,
        bv.BrushPhys(
            brush_model="off", vdw_scheme="continuum", vdw_diagnostics=False
        ),
    )
    gap = rf.target_gap_nm(attraction)
    point = rf.stationarity(attraction, rf.TARGET_ALPHA_DEG, gap)
    print(
        f"    a = {rf.TARGET_A_NM} nm at alpha = {rf.TARGET_ALPHA_DEG} deg "
        f"needs D = {gap:.6f} nm (gamma* = {point.gamma_deg:.3f} deg)"
    )
    print(
        f"    attraction alone there: dE/dD = {point.dE_dD:+.4f} kBT/nm, "
        f"dE/dalpha = {point.dE_dalpha:+.5f} kBT/deg"
    )
    return gap, point


def section_r2():
    print("R2  fitting (P_ref, lambda): valley-floor vertex on the target")
    started = time.perf_counter()
    x, info, history = rf.fit_valley_vertex(X0)
    pressure, decay = 10.0 ** x[0], x[1]
    alpha_v, a_v, stencil = info
    print(
        f"    converged in {len(history) - 1} steps, "
        f"{time.perf_counter() - started:.0f} s"
    )
    write_csv(
        "R2_fit_history.csv",
        ["iteration", "P_ref_Pa", "lambda_nm", "vertex_alpha_deg",
         "vertex_a_nm"],
        [
            [i, f"{10 ** h[0][0]:.6g}", f"{h[0][1]:.6f}",
             f"{h[2][0]:.5f}", f"{h[2][1]:.5f}"]
            for i, h in enumerate(history)
        ],
    )
    write_csv(
        "R2_valley_stencil.csv",
        ["alpha_deg", "a_star_nm", "Interaction_kBT"],
        [[f"{a:.3f}", f"{b:.5f}", f"{e:.6f}"] for a, b, e in stencil],
    )
    model = rf.exponential_model(pressure, decay)
    gamma, components = rf.relax_twist_at_fixed_a(
        model, a_v, alpha_v, rf.VALLEY_GAMMAS
    )
    gap = model.nearest_exact_gap(a_v, alpha_v, gamma)
    at_gap = float(
        bv.exponential_steric_pressure_Pa(
            gap * 1e-9, pressure, decay * 1e-9, model.steric_range,
            model.steric_reference,
        )
    )
    fitted = {
        "P_ref_Pa": pressure,
        "reference_gap_nm": rf.STERIC_REFERENCE_GAP_NM,
        "lambda_nm": decay,
        "vertex_alpha_deg": alpha_v,
        "vertex_a_nm": a_v,
        "gap_at_target_nm": gap,
        "gamma_at_target_deg": -abs(gamma),
        "pressure_at_target_gap_Pa": at_gap,
        "pressure_extrapolated_to_contact_Pa": pressure
        * float(np.exp(rf.STERIC_REFERENCE_GAP_NM / decay)),
        "components_kBT": {
            key: float(components[key])
            for key in ("Dipole", "VdW", "Steric", "Interaction", "Total")
        },
    }
    print(
        f"    P_ref = {pressure:.5g} Pa at {rf.STERIC_REFERENCE_GAP_NM} nm, "
        f"lambda = {decay:.4f} nm"
    )
    print(
        f"    valley vertex: alpha* = {alpha_v:.3f} deg, a* = {a_v:.4f} nm, "
        f"gap = {gap:.4f} nm, gamma* = {-abs(gamma):.3f} deg, "
        f"pressure there = {at_gap / 1e3:.2f} kPa"
    )
    print(
        f"    energies: Dip = {components['Dipole']:.4f}, "
        f"VdW = {components['VdW']:.4f}, "
        f"Steric = {components['Steric']:.4f}, "
        f"Interaction = {components['Interaction']:.4f} kBT"
    )
    return fitted


def section_r3(fitted):
    print("R3  verification")
    model = rf.exponential_model(fitted["P_ref_Pa"], fitted["lambda_nm"])

    # E(D) at the experimental angle and fitted twist, down to near contact,
    # so a collapse branch would be seen rather than assumed away.
    gaps = np.concatenate(
        [np.geomspace(1e-3, 1.0, 14, endpoint=False), np.linspace(1.0, 7.0, 61)]
    )
    rows = []
    energies = []
    for gap in gaps:
        a_nm = model.a_for_exact_gap(
            rf.TARGET_ALPHA_DEG, fitted["gamma_at_target_deg"], gap
        )
        c = model.energy_components(
            a_nm, rf.TARGET_ALPHA_DEG, fitted["gamma_at_target_deg"]
        )
        energies.append(c["Interaction"])
        rows.append(
            [f"{gap:.6f}", f"{a_nm:.5f}", f"{c['Dipole']:.6f}",
             f"{c['VdW']:.6f}", f"{c['Steric']:.6f}", f"{c['Interaction']:.6f}"]
        )
    write_csv(
        "R3_energy_versus_gap.csv",
        ["gap_nm", "a_nm", "Dipole_kBT", "VdW6_kBT", "Steric_kBT",
         "Interaction_kBT"],
        rows,
    )
    energies = np.array(energies)
    lowest = int(np.argmin(energies))
    print(
        f"    E(D) at alpha = 74: lowest point on the scan at "
        f"D = {gaps[lowest]:.4f} nm (E = {energies[lowest]:.4f} kBT); "
        f"E at 1e-3 nm = {energies[0]:.1f} kBT"
    )

    started = time.perf_counter()
    fit = free_gap_alpha_scan(
        model, ALPHAS, GAMMAS, passes=2, grid_points=48, verbose=True
    )
    star = fit["components_star"]
    print(
        f"    free-gap minimum: alpha* = {fit['alpha_star']:.3f} deg, "
        f"a* = {fit['a_star']:.4f} nm, D* = {fit['gap_star_nm']:.4f} nm, "
        f"gamma* = {fit['gamma_star']:.3f} deg, "
        f"Eint = {star['Interaction']:.4f} kBT "
        f"[{time.perf_counter() - started:.0f} s]"
    )
    write_csv(
        "R3_free_gap_alpha_scan.csv",
        ["alpha_deg", "gamma_opt_deg", "a_opt_nm", "gap_opt_nm",
         "Interaction_kBT"],
        [
            [f"{a:.4f}", f"{g:.4f}", f"{x:.5f}", f"{d:.5f}", f"{e:.6f}"]
            for a, g, x, d, e in zip(
                fit["alphas"], fit["gamma_opt"], fit["a_opt"],
                fit["gap_opt"], fit["interaction"],
            )
        ],
    )
    # Second, independent check with the fine (a, alpha) solver: minimise over
    # a at a ladder of alpha values across the minimum and fit a parabola.
    # The parabola averages whatever quadrature noise is left, so its vertex
    # is the robust estimate of the model's minimum.
    from scipy.optimize import minimize_scalar

    valley = []
    for alpha in np.arange(73.4, 74.61, 0.2):
        guess = 20.302 - 0.133 * (alpha - 74.0)
        best = minimize_scalar(
            lambda a_nm: rf.relax_twist_at_fixed_a(model, a_nm, alpha)[1][
                "Interaction"
            ],
            bounds=(guess - 0.06, guess + 0.06),
            method="bounded",
            options={"xatol": 2e-4},
        )
        valley.append((alpha, float(best.x), float(best.fun)))
        print(
            f"    valley alpha = {alpha:.2f}: a* = {best.x:.4f} nm, "
            f"E = {best.fun:.6f} kBT",
            flush=True,
        )
    valley = np.array(valley)
    coefficients = np.polyfit(valley[:, 0], valley[:, 2], 2)
    vertex_alpha = float(-coefficients[1] / (2.0 * coefficients[0]))
    line = np.polyfit(valley[:, 0], valley[:, 1], 1)
    vertex_a = float(np.polyval(line, vertex_alpha))
    residual = valley[:, 2] - np.polyval(coefficients, valley[:, 0])
    print(
        f"    valley parabola: alpha* = {vertex_alpha:.3f} deg, "
        f"a* = {vertex_a:.4f} nm, curvature = {2 * coefficients[0]:.4f} "
        f"kBT/deg^2, rms residual = {np.sqrt(np.mean(residual**2)):.1e} kBT"
    )
    write_csv(
        "R3_valley_fine_solver.csv",
        ["alpha_deg", "a_star_nm", "Interaction_kBT"],
        [[f"{a:.3f}", f"{x:.5f}", f"{e:.6f}"] for a, x, e in valley],
    )
    return {
        "valley_vertex_alpha_deg": vertex_alpha,
        "valley_vertex_a_nm": vertex_a,
        "valley_curvature_kBT_per_deg2": float(2 * coefficients[0]),
        "valley_rms_residual_kBT": float(np.sqrt(np.mean(residual**2))),
        "alpha_star_deg": fit["alpha_star"],
        "a_star_nm": fit["a_star"],
        "gap_star_nm": fit["gap_star_nm"],
        "gamma_star_deg": fit["gamma_star"],
        "components_kBT": {
            key: float(star[key])
            for key in ("Dipole", "VdW", "Steric", "Interaction", "Total")
        },
        "energy_versus_gap_lowest_nm": float(gaps[lowest]),
        "energy_at_1e-3_nm_kBT": float(energies[0]),
    }


def section_r4(fitted):
    print("R4  transfer to the other sizes, parameters frozen")
    rows = []
    results = {}
    alphas = np.linspace(60.0, 90.0, 16)
    gammas = np.linspace(-60.0, 60.0, 13, endpoint=False)
    for size in TRANSFER_SIZES_NM:
        started = time.perf_counter()
        model = rf.exponential_model(
            fitted["P_ref_Pa"], fitted["lambda_nm"], L_nm=size
        )
        fit = free_gap_alpha_scan(model, alphas, gammas, passes=2,
                                  grid_points=40)
        experiment = bv.EXP_DATA.get(size)
        results[size] = fit
        rows.append(
            [f"{size:g}", f"{fit['alpha_star']:.4f}", f"{fit['a_star']:.5f}",
             f"{fit['gap_star_nm']:.5f}",
             f"{experiment['alpha']:.2f}" if experiment else "",
             f"{experiment['d']:.2f}" if experiment else ""]
        )
        print(
            f"    L = {size:>4g} nm: alpha* = {fit['alpha_star']:7.3f} "
            f"(exp {experiment['alpha'] if experiment else '--'}), "
            f"a* = {fit['a_star']:8.4f} nm "
            f"(exp {experiment['d'] if experiment else '--'}) "
            f"[{time.perf_counter() - started:.0f} s]",
            flush=True,
        )
    write_csv(
        "R4_size_transfer.csv",
        ["L_nm", "alpha_star_deg", "a_star_nm", "gap_star_nm",
         "alpha_experiment_deg", "a_experiment_nm"],
        rows,
    )
    return {
        f"L{size:g}": {
            "alpha_star_deg": fit["alpha_star"],
            "a_star_nm": fit["a_star"],
            "gap_star_nm": fit["gap_star_nm"],
        }
        for size, fit in results.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    gap, point = section_r1()
    fitted = section_r2()
    report = {
        "model": {
            "vdw": "exact continuum, A = 2.0e-20 J, primitive-6 bonds",
            "Ms_Apm": 285_000.0,
            "K_ani_Jpm3": 2.0e4,
            "repulsion": "P(D) = P_ref exp(-(D - D_ref)/lambda) on SEI areas",
        },
        "target": {
            "a_nm": rf.TARGET_A_NM,
            "alpha_deg": rf.TARGET_ALPHA_DEG,
            "gap_with_attraction_twist_nm": gap,
        },
        "fitted": fitted,
    }
    (OUTPUT_DIR / "R_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    report["verification"] = section_r3(fitted)
    (OUTPUT_DIR / "R_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if args.transfer:
        report["transfer"] = section_r4(fitted)
    report["runtime_s"] = time.perf_counter() - started
    (OUTPUT_DIR / "R_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  wrote R_report.json")
    print(f"done in {report['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
