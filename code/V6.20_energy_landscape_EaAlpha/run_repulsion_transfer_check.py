"""R5: what the measured angle costs in the fitted model, size by size.

R4 transfers the fitted repulsion to the other sizes and reports where the
minimum lands.  Where the predicted and measured angles disagree, the useful
number is how much energy the measured angle costs, i.e. whether the miss is
a near-degeneracy inside the numerical scatter or a structure the model
clearly rejects.  At every size with data the relaxed state is computed at the
measured alpha and at the model's alpha*, with two independent solvers:

    (alpha, D)  coordinate descent over (gamma, gap) at fixed alpha
    fixed a     twist relaxed at fixed (a, alpha), then minimised over a

Reads the fitted parameters from R_report.json and alpha* from
R4_size_transfer.csv (run_repulsion_fit.py --transfer).

    python run_repulsion_transfer_check.py     # ~15 min
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import v620_brush_vdw as bv
import v620_repulsion_fit as rf

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
GAMMAS = np.linspace(-60.0, 60.0, 25, endpoint=False)


def coordinate_descent(model, alpha_deg):
    relaxed = model.relax_gap_and_twist(
        alpha_deg, gamma_grid=GAMMAS, passes=3, grid_points=40
    )
    if not relaxed.get("converged", False):
        raise RuntimeError(f"No relaxed state at alpha = {alpha_deg} deg.")
    return (relaxed["components"]["Interaction"], relaxed["a_nm"],
            relaxed["gap_nm"], relaxed["gamma_deg"])


def fixed_a(model, alpha_deg, a_guess_nm):
    def energy(a_nm):
        return rf.relax_twist_at_fixed_a(
            model, a_nm, alpha_deg, rf.GAMMAS_FINE
        )[1]["Interaction"]

    best = minimize_scalar(
        energy, bounds=(a_guess_nm - 0.1, a_guess_nm + 0.1),
        method="bounded", options={"xatol": 5e-4},
    )
    a_nm = float(best.x)
    gamma, components = rf.relax_twist_at_fixed_a(
        model, a_nm, alpha_deg, rf.GAMMAS_FINE
    )
    return (components["Interaction"], a_nm,
            model.nearest_exact_gap(a_nm, alpha_deg, gamma), gamma)


def main():
    report = json.loads((OUTPUT_DIR / "R_report.json").read_text("utf-8"))
    fitted = report["fitted"]
    alpha_star = {16.0: report["verification"]["alpha_star_deg"]}
    with (OUTPUT_DIR / "R4_size_transfer.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            alpha_star[float(row["L_nm"])] = float(row["alpha_star_deg"])

    header = [
        "L_nm", "solver", "alpha_experiment_deg", "a_experiment_nm",
        "alpha_star_deg", "E_star_kBT", "a_star_nm",
        "E_at_alpha_experiment_kBT", "a_at_alpha_experiment_nm",
        "gap_at_alpha_experiment_nm", "gamma_at_alpha_experiment_deg",
        "cost_kBT",
    ]
    rows = []
    started = time.perf_counter()
    print("R5  energy cost of the measured angle, parameters frozen")
    for size, experiment in sorted(bv.EXP_DATA.items()):
        model = rf.exponential_model(
            fitted["P_ref_Pa"], fitted["lambda_nm"], L_nm=size
        )
        star_cd = coordinate_descent(model, alpha_star[size])
        exp_cd = coordinate_descent(model, experiment["alpha"])
        star_fa = fixed_a(model, alpha_star[size], star_cd[1])
        exp_fa = fixed_a(model, experiment["alpha"], exp_cd[1])
        for solver, star, at_exp in (("alpha_D", star_cd, exp_cd),
                                     ("fixed_a", star_fa, exp_fa)):
            cost = at_exp[0] - star[0]
            rows.append([
                f"{size:g}", solver, f"{experiment['alpha']:.2f}",
                f"{experiment['d']:.2f}", f"{alpha_star[size]:.4f}",
                f"{star[0]:.6f}", f"{star[1]:.5f}", f"{at_exp[0]:.6f}",
                f"{at_exp[1]:.5f}", f"{at_exp[2]:.5f}", f"{at_exp[3]:.3f}",
                f"{cost:.6f}",
            ])
            print(
                f"    L = {size:>4g} nm [{solver:>7}]: alpha* = "
                f"{alpha_star[size]:.3f} (E = {star[0]:.4f}), "
                f"alpha_exp = {experiment['alpha']:.1f} (E = {at_exp[0]:.4f}, "
                f"a = {at_exp[1]:.3f} nm vs exp {experiment['d']:.2f}, "
                f"gap = {at_exp[2]:.3f} nm), cost = {cost:+.4f} kBT "
                f"[{time.perf_counter() - started:.0f} s]",
                flush=True,
            )
    path = OUTPUT_DIR / "R5_experimental_angle_cost.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
