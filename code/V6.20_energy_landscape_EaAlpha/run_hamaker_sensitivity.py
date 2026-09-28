"""Which Hamaker constant does the experimental angle imply?

With the attraction converged, the V6.20 constant-gap minimum for L = 16 nm
moves from the experimental angle to the locked-twist branch near 78.6 deg.
The two branches are separated by a fraction of a kBT, and the only term that
tips the balance is the attraction, so the observed angle is a sensitive probe
of the Hamaker constant rather than a confirmation of it.

This script measures that.  ``A`` is scanned, the constant-gap minimisation is
repeated at each value with the exact attraction, and the branch-switch value
is bracketed.  The V6.20 default is 2.0e-20 J; Faure et al. (Langmuir 27, 8659
(2011), DOI 10.1021/la201387d) report 9 to 29 zJ for iron oxide across a
nonpolar solvent, so the whole interesting range is inside the literature
spread.

D1  alpha*, a* and the branch margin against the Hamaker constant.
D2  The Hamaker constant at which the two branches cross, by bisection.
D3  The nearest-neighbour gap that each experimental (a, alpha) pair implies,
    since the brush picture makes that gap the thing being predicted.

Outputs go to ``outputs/V6.20_energy_landscape_EaAlpha/brush_vdw``.

    python run_hamaker_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import v620_brush_vdw as bv
from v620_brush_fitting import fixed_gap_alpha_scan, minimise_twist

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF = bv.REFERENCE
ALPHAS = np.linspace(60.0, 90.0, 31)
GAMMAS = np.linspace(-60.0, 60.0, 19, endpoint=False)
HAMAKER_J = (
    0.9e-20, 1.1e-20, 1.2e-20, 1.26e-20, 1.3e-20, 1.5e-20, 2.0e-20, 2.9e-20
)


def write_csv(name, header, rows):
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return path


def model_for(hamaker_J: float, voxel: str = "exact"):
    phys = bv.BrushPhys(
        brush_model="off",
        vdw_diagnostics=False,
        Hamaker=float(hamaker_J),
        **(
            {"vdw_scheme": "continuum"}
            if voxel == "exact"
            else {"vdw_scheme": "voxel", "voxel_count": 4}
        ),
    )
    return bv.BrushRhombohedralSL(REF["L_nm"], phys)


def branch_margin(hamaker_J: float, gap_nm: float, locked_alpha: float):
    """E(experimental angle) - E(locked-twist angle); negative keeps 74 deg."""
    model = model_for(hamaker_J)
    _, _, here = minimise_twist(model, REF["alpha_deg"], gap_nm, GAMMAS)
    _, _, there = minimise_twist(model, locked_alpha, gap_nm, GAMMAS)
    return here["Interaction"] - there["Interaction"]


def main() -> None:
    started = time.perf_counter()
    print(f"Output directory: {OUTPUT_DIR}")
    gap_model = model_for(2.0e-20, voxel="legacy")
    gap_nm = gap_model.nearest_exact_gap(
        REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
    )
    print(f"reference gap D_ref = {gap_nm:.6f} nm")

    print("D1  the fit against the Hamaker constant, exact attraction")
    rows = []
    locked_alpha = None
    for hamaker in HAMAKER_J:
        fit = fixed_gap_alpha_scan(
            model_for(hamaker), gap_nm, ALPHAS, GAMMAS
        )
        locked = fit["alphas"][np.abs(fit["gamma_opt"]) < 1e-6]
        if locked.size and locked_alpha is None and fit["alpha_star"] > 77.0:
            locked_alpha = fit["alpha_star"]
        rows.append(
            [
                f"{hamaker:.4e}",
                f"{fit['alpha_star']:.4f}",
                f"{fit['gamma_star']:.4f}",
                f"{fit['a_star']:.5f}",
                f"{fit['components_star']['Dipole']:.5f}",
                f"{fit['components_star']['VdW']:.5f}",
                f"{fit['components_star']['Interaction']:.5f}",
            ]
        )
        print(
            f"    A={hamaker:.3e} J  alpha*={fit['alpha_star']:7.3f}"
            f"  a*={fit['a_star']:8.4f}"
            f"  Eint={fit['components_star']['Interaction']:8.4f} kBT",
            flush=True,
        )
    write_csv(
        "D1_hamaker_scan.csv",
        [
            "hamaker_J",
            "alpha_star_deg",
            "gamma_star_deg",
            "a_star_nm",
            "Dipole_kBT",
            "VdW6_kBT",
            "Interaction_kBT",
        ],
        rows,
    )

    print("D3  the gap implied by each experimental point")
    implied = []
    for size_nm, data in sorted(bv.EXP_DATA.items()):
        model = bv.BrushRhombohedralSL(
            size_nm, bv.BrushPhys(brush_model="off")
        )
        legacy_twist = model.nearest_exact_gap(
            data["d"], data["alpha"], REF["gamma_deg"]
        )
        aligned = model.nearest_exact_gap(data["d"], data["alpha"], 0.0)
        implied.append(
            [
                f"{size_nm:g}",
                f"{data['d']:.2f}",
                f"{data['alpha']:.1f}",
                f"{legacy_twist:.4f}",
                f"{aligned:.4f}",
                f"{0.5 * legacy_twist:.4f}",
            ]
        )
        print(
            f"    L={size_nm:>5g} nm  a={data['d']:6.2f} nm"
            f"  alpha={data['alpha']:5.1f} deg  ->  gap={legacy_twist:7.4f} nm"
            f"  (implied brush height {0.5 * legacy_twist:.3f} nm)"
        )
    write_csv(
        "D3_implied_experimental_gaps.csv",
        [
            "L_nm",
            "a_experiment_nm",
            "alpha_experiment_deg",
            "gap_at_legacy_twist_nm",
            "gap_at_zero_twist_nm",
            "implied_brush_height_nm",
        ],
        implied,
    )

    print("D2  where the two branches cross")
    if locked_alpha is None:
        locked_alpha = 78.556
    crossing = brentq(
        lambda hamaker: branch_margin(hamaker, gap_nm, locked_alpha),
        0.5e-20,
        2.9e-20,
        xtol=1e-24,
    )
    print(
        f"    the experimental branch is the global minimum for "
        f"A < {crossing:.4e} J"
    )
    report = {
        "reference_gap_nm": gap_nm,
        "implied_experimental_gaps_nm": {
            f"L{row[0]}": float(row[3]) for row in implied
        },
        "locked_twist_alpha_deg": locked_alpha,
        "branch_crossing_hamaker_J": crossing,
        "v620_default_hamaker_J": 2.0e-20,
        "literature_range_J": [0.9e-20, 2.9e-20],
        "runtime_s": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "D_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  wrote D_report.json")
    print(f"done in {report['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
