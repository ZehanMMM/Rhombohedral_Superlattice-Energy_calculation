"""The other approximation in the V6.20 vdW term: which neighbours are summed.

V6.20 keeps only the six primitive bonds in the van der Waals energy, on the
grounds that the term is short ranged, and reports the 26-bond sum as a
diagnostic.  In a rhombohedral lattice of co-oriented cubes that reasoning does
not hold: at ``alpha < 90 deg`` the six closest surface pairs are the
edge-type contacts of the ``[0, 1, -1]`` family, which are *not* primitive
bonds.  The retained bonds are therefore not the closest ones, and the
neglected tail is dominated by the closest surfaces in the lattice.

This script quantifies that, because it changes the same quantities the brush
study is about.

C1  Bond-by-bond attraction at the experimental reference state, exact rule,
    with the two neighbour conventions separated.
C2  The fit under both conventions, with the gap pinned and with the gap free
    against the calibrated brush.
C3  The size of the other truncation in the same Hamiltonian: the 3x3x3
    dipolar lattice sum.

Outputs go to ``outputs/V6.20_energy_landscape_EaAlpha/brush_vdw``.

    python run_neighbour_convention.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

import v620_brush_vdw as bv
from v620_brush_fitting import (
    calibrate_brush_height,
    fixed_gap_alpha_scan,
    free_gap_alpha_scan,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF = bv.REFERENCE
SIZES_NM = (16.0, 13.0, 19.5)
ALPHAS = np.linspace(60.0, 90.0, 31)
GAMMAS = np.linspace(-60.0, 60.0, 19, endpoint=False)


def write_csv(name: str, header, rows) -> Path:
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return path


def base(mode: str, **kwargs) -> bv.BrushPhys:
    return bv.BrushPhys(
        vdw_scheme="continuum",
        vdw_neighbour_mode=mode,
        vdw_diagnostics=(mode == "all26"),
        **kwargs,
    )


def section_c1():
    print("C1  where the attraction actually sits")
    model = bv.BrushRhombohedralSL(REF["L_nm"], base("all26",
                                                     brush_model="off"))
    offsets = model.neighbour_vectors(
        REF["a_nm"], REF["alpha_deg"]
    ) @ model.orientation(REF["gamma_deg"])
    gaps = model.neighbour_exact_gaps(
        REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
    ) * 1e9
    kT = model.kT
    rows = []
    shells = {}
    for index in np.argsort(gaps):
        energy = 0.5 * model.vdw_pair_energy_J(offsets[index]) / kT
        shell = int(np.sum(np.abs(model.neighbour_indices[index])))
        primitive = bool(model.primitive_face_mask[index])
        shells.setdefault(
            (shell, primitive), [0.0, 0, gaps[index], gaps[index]]
        )
        shells[(shell, primitive)][0] += energy
        shells[(shell, primitive)][1] += 1
        shells[(shell, primitive)][2] = min(
            shells[(shell, primitive)][2], gaps[index]
        )
        shells[(shell, primitive)][3] = max(
            shells[(shell, primitive)][3], gaps[index]
        )
        rows.append(
            [
                "".join(f"{i:+d}" for i in model.neighbour_indices[index]),
                shell,
                primitive,
                f"{gaps[index]:.6f}",
                f"{energy:.6f}",
            ]
        )
    write_csv(
        "C1_attraction_by_neighbour.csv",
        [
            "neighbour_index",
            "shell",
            "is_primitive_bond",
            "exact_gap_nm",
            "vdw_energy_per_particle_kBT",
        ],
        rows,
    )
    for (shell, primitive), (total, count, low, high) in sorted(
        shells.items()
    ):
        print(
            f"    shell {shell} {'primitive' if primitive else 'other    '}  "
            f"{count:2d} bonds  gap={low:.4f} to {high:.4f} nm  "
            f"sum={total:8.4f} kBT per particle"
        )
    return shells


def section_c2():
    print("C2  the fit under both neighbour conventions")
    gap_model = bv.BrushRhombohedralSL(REF["L_nm"], base("primitive6",
                                                         brush_model="off"))
    target_gap = gap_model.nearest_exact_gap(
        REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
    )
    rows = []
    summary = {}
    for mode in ("primitive6", "all26"):
        def build(height_nm, mode=mode):
            return bv.BrushRhombohedralSL(
                REF["L_nm"], base(mode, brush_height_nm=height_nm)
            )

        height = calibrate_brush_height(
            build,
            target_gap,
            REF["alpha_deg"],
            gamma_deg=REF["gamma_deg"],
            bracket_nm=(1.2, 3.4),
        )
        print(f"    {mode:10s} calibrated brush height {height:.4f} nm")
        for size_nm in SIZES_NM:
            started = time.perf_counter()
            fit = fixed_gap_alpha_scan(
                bv.BrushRhombohedralSL(size_nm, base(mode, brush_model="off")),
                target_gap,
                ALPHAS,
                GAMMAS,
            )
            rows.append(_row(size_nm, mode, "fixed_gap", "n/a", fit))
            _say(size_nm, mode, "fixed_gap", fit, started)
            summary[f"L{size_nm:g}_{mode}_fixed_gap"] = _brief(fit)

            started = time.perf_counter()
            fit = free_gap_alpha_scan(
                bv.BrushRhombohedralSL(
                    size_nm, base(mode, brush_height_nm=height)
                ),
                ALPHAS,
                GAMMAS,
                passes=2,
                grid_points=32,
            )
            rows.append(
                _row(size_nm, mode, "free_gap_calibrated", height, fit)
            )
            _say(size_nm, mode, "free_gap_calibrated", fit, started)
            summary[f"L{size_nm:g}_{mode}_free_gap"] = _brief(fit)
        summary[f"{mode}_calibrated_brush_height_nm"] = height
    write_csv(
        "C2_neighbour_convention_fit.csv",
        [
            "L_nm",
            "vdw_neighbour_mode",
            "ensemble",
            "brush_height_nm",
            "alpha_star_deg",
            "gamma_star_deg",
            "a_star_nm",
            "gap_star_nm",
            "Dipole_kBT",
            "VdW_kBT",
            "Steric_kBT",
            "Interaction_kBT",
        ],
        rows,
    )
    return summary


def _row(size_nm, mode, ensemble, height, fit):
    star = fit["components_star"]
    return [
        f"{size_nm:g}",
        mode,
        ensemble,
        height if isinstance(height, str) else f"{height:.5f}",
        f"{fit['alpha_star']:.4f}",
        f"{fit['gamma_star']:.4f}",
        f"{fit['a_star']:.5f}",
        f"{fit['gap_star_nm']:.6f}",
        f"{star['Dipole']:.5f}",
        f"{star['VdW']:.5f}",
        f"{star['Steric']:.5f}",
        f"{star['Interaction']:.5f}",
    ]


def _brief(fit):
    return {
        "alpha_star_deg": fit["alpha_star"],
        "a_star_nm": fit["a_star"],
        "gap_star_nm": fit["gap_star_nm"],
        "interaction_kBT": fit["components_star"]["Interaction"],
    }


def _say(size_nm, mode, ensemble, fit, started):
    print(
        f"    L={size_nm:>4g} {mode:10s} {ensemble:20s} "
        f"alpha*={fit['alpha_star']:7.3f}  a*={fit['a_star']:8.4f}  "
        f"D*={fit['gap_star_nm']:6.3f}  "
        f"Eint={fit['components_star']['Interaction']:8.3f} kBT  "
        f"[{time.perf_counter() - started:.0f} s]",
        flush=True,
    )


def section_c3():
    """How big is the 3x3x3 truncation of the dipolar sum?

    The angle selection in A6 turns on a few hundredths of a kBT, so the other
    truncation in the same Hamiltonian has to be sized before any of it is
    believed.  The dipolar lattice sum is only conditionally convergent, and
    V6.20 truncates it at the 26 neighbours of one 3x3x3 block.
    """
    print("C3  truncation of the dipolar sum")
    from scipy.constants import mu_0

    model = bv.BrushRhombohedralSL(REF["L_nm"], base("primitive6",
                                                     brush_model="off"))
    coefficient = mu_0 / (4.0 * np.pi)
    moment = model.mu_magnitude

    def dipole_kBT(a_nm, alpha_deg, half_width):
        basis = model.basis(a_nm, alpha_deg)
        index = np.arange(-half_width, half_width + 1)
        grid = np.array(
            np.meshgrid(index, index, index, indexing="ij")
        ).reshape(3, -1).T
        grid = grid[np.any(grid != 0, axis=1)]
        vectors = grid @ basis
        distance = np.linalg.norm(vectors, axis=1)
        return 0.5 * float(
            np.sum(
                coefficient
                * (
                    moment**2 / distance**3
                    - 3.0
                    * (vectors @ (moment * model.mu_hat)) ** 2
                    / distance**5
                )
            )
        ) / model.kT

    states = {
        "experimental_branch": (REF["a_nm"], REF["alpha_deg"]),
        "locked_twist_branch": (19.5570, 78.556),
    }
    widths = (1, 2, 3, 4, 6, 8, 12, 16)
    rows = []
    converged = {}
    for label, (a_nm, alpha_deg) in states.items():
        values = [dipole_kBT(a_nm, alpha_deg, half) for half in widths]
        converged[label] = values[-1]
        for half, value in zip(widths, values):
            rows.append(
                [
                    label,
                    f"{a_nm:.5f}",
                    f"{alpha_deg:.4f}",
                    2 * half + 1,
                    f"{value:.6f}",
                    f"{value - values[-1]:+.6f}",
                ]
            )
        print(
            f"    {label:20s} 3x3x3={values[0]:.4f}"
            f"  33x33x33={values[-1]:.4f}"
            f"  truncation error={values[0] - values[-1]:+.4f} kBT"
        )
    write_csv(
        "C3_dipole_truncation.csv",
        [
            "state",
            "a_nm",
            "alpha_deg",
            "shell_edge_cells",
            "dipole_per_particle_kBT",
            "minus_largest_shell_kBT",
        ],
        rows,
    )
    return converged


def main() -> None:
    started = time.perf_counter()
    print(f"Output directory: {OUTPUT_DIR}")
    shells = section_c1()
    summary = section_c2()
    dipole = section_c3()
    report = {
        "dipole_converged_kBT": dipole,
        "attraction_by_shell_kBT": {
            f"shell{shell}_{'primitive' if primitive else 'other'}": value[0]
            for (shell, primitive), value in shells.items()
        },
        "fits": summary,
        "runtime_s": time.perf_counter() - started,
    }
    path = OUTPUT_DIR / "C_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {path.name}")
    print(f"done in {report['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
