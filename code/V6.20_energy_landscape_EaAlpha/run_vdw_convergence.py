"""Does the V6.20 van der Waals attraction converge in the voxel count?

The V6.20 Hamiltonian evaluates the Hamaker double volume integral with a
4x4x4 midpoint voxel rule.  This script refines that rule and measures it
against the exact continuum limit of the very same integral, which is
available in closed form for two co-oriented cubes
(``v620_brush_vdw.cube_pair_hamaker_J``).

Sections

A1  One bond at a time: the voxel ladder N = 2 ... 96 against the exact limit,
    for the three contact types that occur in the rhombohedral lattice.
A2  The lattice sums the Hamiltonian actually uses (VdW6, VdW26) versus N.
A3  How the error depends on the surface gap, which is what sets the size of
    the N = 4 error.
A4  The consequence for the fit: the V6.20 constant-gap minimisation repeated
    at every N, i.e. what refining the attraction alone does to alpha*, a* and
    the binding energy, with no brush term involved.
A5  Sharp versus rounded voxel body, the other approximation in the same term.
A6  The alpha curves behind A4, and the margin between the two candidate
    minima that the quadrature error is large enough to swap.

Outputs go to ``outputs/V6.20_energy_landscape_EaAlpha/brush_vdw``.

    python run_vdw_convergence.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v620_brush_vdw as bv
from v620_brush_fitting import fixed_gap_alpha_scan

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEGACY = bv.BrushPhys(brush_model="off")
EXACT = bv.BrushPhys(brush_model="off", vdw_scheme="continuum")
VOXEL_LADDER = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
FIT_LADDER = (2, 3, 4, 6, 8, 12, 16, 24, 32)
FIT_SIZES_NM = (16.0, 13.0, 19.5)
FIT_ALPHAS = np.linspace(60.0, 90.0, 31)
FIT_GAMMAS = np.linspace(-60.0, 60.0, 19, endpoint=False)

mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "figure.dpi": 160,
    }
)


def write_csv(name: str, header, rows) -> Path:
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return path


def reference_bonds():
    """The three contact types at the V6.20 reference state."""
    model = bv.BrushRhombohedralSL(bv.REFERENCE["L_nm"], LEGACY)
    a_nm = bv.REFERENCE["a_nm"]
    alpha = bv.REFERENCE["alpha_deg"]
    gamma = bv.REFERENCE["gamma_deg"]
    offsets = model.neighbour_vectors(a_nm, alpha) @ model.orientation(gamma)
    gaps = model.neighbour_exact_gaps(a_nm, alpha, gamma) * 1e9
    index_edge = int(np.argmin(gaps))
    primitive = np.where(model.primitive_face_mask)[0]
    index_face = int(primitive[int(np.argmin(gaps[primitive]))])
    corner = np.where(np.sum(np.abs(model.neighbour_indices), axis=1) == 3)[0]
    index_corner = int(corner[int(np.argmin(gaps[corner]))])
    return model, [
        ("nearest_edge_contact", index_edge),
        ("primitive_face_bond", index_face),
        ("corner_neighbour", index_corner),
    ], offsets, gaps


def section_a1():
    print("A1  voxel ladder, one bond at a time")
    model, bonds, offsets, gaps = reference_bonds()
    rows = []
    curves = {}
    for label, index in bonds:
        offset = offsets[index]
        exact = model.vdw_pair_energy_exact_J(offset)
        errors = []
        for count in VOXEL_LADDER:
            refined = bv.BrushRhombohedralSL(
                bv.REFERENCE["L_nm"], LEGACY.with_voxels(count)
            )
            value = refined.vdw_pair_energy_voxel_J(offset)
            error = abs(value / exact - 1.0)
            errors.append(error)
            rows.append(
                [
                    label,
                    "".join(f"{i:+d}" for i in model.neighbour_indices[index]),
                    f"{gaps[index]:.6f}",
                    count,
                    f"{value:.10e}",
                    f"{exact:.10e}",
                    f"{value / exact:.8f}",
                    f"{error:.6e}",
                ]
            )
        curves[label] = (np.array(VOXEL_LADDER), np.array(errors), exact)
    write_csv(
        "A1_bond_voxel_ladder.csv",
        [
            "contact_type",
            "neighbour_index",
            "exact_gap_nm",
            "voxel_count",
            "U_voxel_J",
            "U_exact_J",
            "U_voxel_over_U_exact",
            "relative_error",
        ],
        rows,
    )
    orders = {}
    for label, (counts, errors, _) in curves.items():
        mask = counts >= 16
        slope = np.polyfit(np.log(counts[mask]), np.log(errors[mask]), 1)[0]
        orders[label] = float(-slope)
        print(
            f"    {label:22s} gap={gaps[dict(bonds)[label]]:.3f} nm  "
            f"N=4 error={errors[VOXEL_LADDER.index(4)]:.3f}  "
            f"observed order={-slope:.2f}"
        )
    return curves, orders


def section_a2():
    print("A2  lattice sums versus voxel count")
    a_nm = bv.REFERENCE["a_nm"]
    alpha = bv.REFERENCE["alpha_deg"]
    gamma = bv.REFERENCE["gamma_deg"]
    exact_model = bv.BrushRhombohedralSL(bv.REFERENCE["L_nm"], EXACT)
    exact = exact_model.energy_components(a_nm, alpha, gamma)
    rows = []
    ladder = []
    for count in VOXEL_LADDER:
        model = bv.BrushRhombohedralSL(
            bv.REFERENCE["L_nm"], LEGACY.with_voxels(count)
        )
        components = model.energy_components(a_nm, alpha, gamma)
        ladder.append((count, components))
        rows.append(
            [
                count,
                f"{components['VdW_face6']:.8f}",
                f"{components['VdW_full26']:.8f}",
                f"{components['Dipole']:.8f}",
                f"{components['Interaction_legacy']:.8f}",
                f"{components['VdW_face6'] / exact['VdW_face6']:.8f}",
                f"{abs(components['VdW_face6'] / exact['VdW_face6'] - 1):.6e}",
            ]
        )
    rows.append(
        [
            "exact",
            f"{exact['VdW_face6']:.8f}",
            f"{exact['VdW_full26']:.8f}",
            f"{exact['Dipole']:.8f}",
            f"{exact['Interaction_legacy']:.8f}",
            "1.00000000",
            "0.0",
        ]
    )
    write_csv(
        "A2_lattice_vdw_vs_voxels.csv",
        [
            "voxel_count",
            "VdW6_kBT",
            "VdW26_kBT",
            "Dipole_kBT",
            "Interaction_kBT",
            "VdW6_over_exact",
            "relative_error",
        ],
        rows,
    )
    print(
        f"    N=4   VdW6={ladder[VOXEL_LADDER.index(4)][1]['VdW_face6']:.4f} kBT"
        f"   exact VdW6={exact['VdW_face6']:.4f} kBT"
    )
    return ladder, exact


def section_a3():
    print("A3  error versus surface gap, aligned face-to-face pair")
    edge_nm = bv.REFERENCE["L_nm"]
    model = bv.BrushRhombohedralSL(edge_nm, LEGACY)
    gaps = (0.2, 0.5, 1.0, 2.0, 3.377736, 6.0, 10.0)
    counts = (2, 4, 8, 16, 32, 64)
    rows = []
    grid = np.zeros((len(gaps), len(counts)))
    for i, gap in enumerate(gaps):
        offset = np.array([(edge_nm + gap) * 1e-9, 0.0, 0.0])
        exact = model.vdw_pair_energy_exact_J(offset)
        for j, count in enumerate(counts):
            refined = bv.BrushRhombohedralSL(
                edge_nm, LEGACY.with_voxels(count)
            )
            value = refined.vdw_pair_energy_voxel_J(offset)
            grid[i, j] = abs(value / exact - 1.0)
            rows.append(
                [
                    f"{gap:.6f}",
                    count,
                    f"{edge_nm / count:.4f}",
                    f"{(edge_nm / count) / gap:.4f}",
                    f"{value:.8e}",
                    f"{exact:.8e}",
                    f"{grid[i, j]:.6e}",
                ]
            )
    write_csv(
        "A3_error_versus_gap.csv",
        [
            "gap_nm",
            "voxel_count",
            "voxel_step_nm",
            "step_over_gap",
            "U_voxel_J",
            "U_exact_J",
            "relative_error",
        ],
        rows,
    )
    return gaps, counts, grid


def section_a4():
    print("A4  the V6.20 constant-gap fit repeated at every voxel count")
    gap_model = bv.BrushRhombohedralSL(bv.REFERENCE["L_nm"], LEGACY)
    reference_gap = gap_model.nearest_exact_gap(
        bv.REFERENCE["a_nm"], bv.REFERENCE["alpha_deg"],
        bv.REFERENCE["gamma_deg"],
    )
    print(f"    reference gap D_ref = {reference_gap:.6f} nm")
    rows = []
    collected = {}
    for size_nm in FIT_SIZES_NM:
        for label, phys in [
            *[(str(n), LEGACY.replace(voxel_count=n, vdw_diagnostics=False))
              for n in FIT_LADDER],
            ("exact", EXACT.replace(vdw_diagnostics=False)),
        ]:
            model = bv.BrushRhombohedralSL(size_nm, phys)
            started = time.perf_counter()
            fit = fixed_gap_alpha_scan(
                model, reference_gap, FIT_ALPHAS, FIT_GAMMAS
            )
            star = fit["components_star"]
            rows.append(
                [
                    f"{size_nm:g}",
                    label,
                    f"{fit['alpha_star']:.4f}",
                    f"{fit['gamma_star']:.4f}",
                    f"{fit['a_star']:.5f}",
                    f"{reference_gap:.6f}",
                    f"{star['Dipole']:.5f}",
                    f"{star['VdW']:.5f}",
                    f"{star['Interaction']:.5f}",
                    f"{time.perf_counter() - started:.1f}",
                ]
            )
            collected[(size_nm, label)] = fit
            print(
                f"    L={size_nm:>4g} N={label:>5s}  alpha*={fit['alpha_star']:7.3f}"
                f"  a*={fit['a_star']:8.4f}  VdW6={star['VdW']:8.4f}"
                f"  Eint={star['Interaction']:8.4f} kBT"
                f"  [{time.perf_counter() - started:.0f} s]",
                flush=True,
            )
    write_csv(
        "A4_fixed_gap_fit_vs_voxels.csv",
        [
            "L_nm",
            "voxel_count",
            "alpha_star_deg",
            "gamma_star_deg",
            "a_star_nm",
            "gap_nm",
            "Dipole_kBT",
            "VdW6_kBT",
            "Interaction_kBT",
            "runtime_s",
        ],
        rows,
    )
    return collected


def section_a5():
    """Sharp versus rounded voxel body.

    The rounded body is carried with partial voxel volumes, so its own volume
    converges to the Steiner value (4007.81 nm^3 at L = 16 nm, r = 1.5 nm,
    97.85 % of the sharp cube).  Its energy ladder is still not clean enough
    for a Richardson step - the residual subsampling error leaves it slightly
    non-monotone at the 0.2 % level - so the ratio is quoted at the finest
    grid and rounded to two significant figures rather than extrapolated.
    """
    print("A5  sharp versus rounded voxel body")
    model, bonds, offsets, gaps = reference_bonds()
    rows = []
    for label, index in bonds:
        offset = offsets[index]
        finest = {}
        for body in ("sharp", "rounded"):
            ladder = {}
            volume = None
            for count in (24, 48, 96):
                refined = bv.BrushRhombohedralSL(
                    bv.REFERENCE["L_nm"],
                    LEGACY.replace(voxel_count=count, voxel_body=body),
                )
                ladder[count] = refined.vdw_pair_energy_voxel_J(offset)
                volume = refined.vox_number * refined.vox_volume * 1e27
            finest[body] = ladder[96]
            rows.append(
                [
                    label,
                    body,
                    f"{gaps[index]:.6f}",
                    f"{volume:.4f}",
                    f"{ladder[24]:.8e}",
                    f"{ladder[48]:.8e}",
                    f"{ladder[96]:.8e}",
                    f"{ladder[96] / ladder[48] - 1.0:+.4e}",
                ]
            )
        print(
            f"    {label:22s} rounded/sharp at N=96 = "
            f"{finest['rounded'] / finest['sharp']:.4f}"
        )
    write_csv(
        "A5_voxel_body_shape.csv",
        [
            "contact_type",
            "voxel_body",
            "exact_gap_nm",
            "body_volume_nm3_at_N96",
            "U_N24_J",
            "U_N48_J",
            "U_N96_J",
            "last_refinement_change",
        ],
        rows,
    )
    return rows


def section_a6(reference_gap: float, fits):
    """The margin that decides alpha*, and the alpha curves behind it.

    On the constant-gap path E(alpha) has two candidate minima for L = 16 nm:
    an interior one near the experimental angle, and a kink at the angle where
    the relaxed twist locks to gamma = 0.  Which one wins is decided by a few
    hundredths of a kBT, so the margin is worth tabulating next to the
    quadrature error that dwarfs it.
    """
    print("A6  the margin between the two candidate minima, L = 16 nm")
    rows = []
    for (size_nm, label), fit in fits.items():
        for alpha, gamma, a_nm, energy in zip(
            fit["alphas"], fit["gamma_opt"], fit["a_opt"], fit["interaction"]
        ):
            rows.append(
                [
                    f"{size_nm:g}",
                    label,
                    f"{alpha:.4f}",
                    f"{gamma:.4f}",
                    f"{a_nm:.5f}",
                    f"{energy:.6f}",
                ]
            )
    write_csv(
        "A6_alpha_curves.csv",
        ["L_nm", "voxel_count", "alpha_deg", "gamma_opt_deg", "a_nm",
         "interaction_kBT"],
        rows,
    )

    from v620_brush_fitting import minimise_twist

    experimental_alpha = bv.EXP_DATA[16.0]["alpha"]
    # The locked-twist angle is a property of the geometry, so it is taken
    # once, from the converged fit, and the same angle is then used at every
    # voxel count.  Comparing each N against its *own* minimum would return
    # zero by construction whenever that minimum is the experimental branch.
    exact_fit = fits[(16.0, "exact")]
    locked = exact_fit["alphas"][np.abs(exact_fit["gamma_opt"]) < 1e-6]
    locked_alpha = (
        float(exact_fit["alpha_star"])
        if exact_fit["alpha_star"] > 77.0
        else float(locked.min())
    )
    print(f"    locked-twist branch at alpha = {locked_alpha:.3f} deg")
    margin_rows = []
    for label in (*[str(n) for n in FIT_LADDER], "exact"):
        model = bv.BrushRhombohedralSL(
            16.0,
            (
                EXACT if label == "exact"
                else LEGACY.replace(voxel_count=int(label))
            ).replace(vdw_diagnostics=False),
        )
        _, _, at_experiment = minimise_twist(
            model, experimental_alpha, reference_gap, FIT_GAMMAS
        )
        _, _, at_locked = minimise_twist(
            model, locked_alpha, reference_gap, FIT_GAMMAS
        )
        margin = at_experiment["Interaction"] - at_locked["Interaction"]
        margin_rows.append(
            [
                label,
                f"{experimental_alpha:.2f}",
                f"{at_experiment['Interaction']:.6f}",
                f"{locked_alpha:.4f}",
                f"{at_locked['Interaction']:.6f}",
                f"{margin:.6f}",
                f"{fits[(16.0, label)]['alpha_star']:.4f}",
                "experimental branch" if margin < 0 else "locked-twist branch",
            ]
        )
        print(
            f"    N={label:>5s}  E({experimental_alpha:.0f} deg)="
            f"{at_experiment['Interaction']:8.4f}"
            f"  E({locked_alpha:.3f} deg)={at_locked['Interaction']:8.4f}"
            f"  margin={margin:+.4f} kBT"
            f"  -> {'experimental' if margin < 0 else 'locked-twist'}"
        )
    write_csv(
        "A6_alpha_selection_margin.csv",
        [
            "voxel_count",
            "experimental_alpha_deg",
            "interaction_at_experimental_alpha_kBT",
            "locked_twist_alpha_deg",
            "interaction_at_locked_twist_alpha_kBT",
            "margin_kBT",
            "alpha_star_from_scan_deg",
            "winner",
        ],
        margin_rows,
    )
    return margin_rows


def figure(curves, ladder, exact, gap_data, fits):
    gaps, counts, grid = gap_data
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    ax = axes[0, 0]
    anchor = None
    for label, (ns, errors, _) in curves.items():
        ax.loglog(ns, errors, "o-", ms=3, label=label.replace("_", " "))
        if label == "primitive_face_bond":
            anchor = errors[VOXEL_LADDER.index(8)]
    guide = np.array([6.0, 96.0])
    ax.loglog(
        guide,
        2.2 * anchor * (guide / 8.0) ** -2.0,
        "k--",
        lw=0.8,
        label="second order guide",
    )
    ax.set_xlabel("voxels per axis N")
    ax.set_ylabel("relative error of one pair energy")
    ax.set_title("A1  single-bond convergence to the exact limit")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[0, 1]
    ns = np.array([item[0] for item in ladder])
    vdw6 = np.array([item[1]["VdW_face6"] for item in ladder])
    vdw26 = np.array([item[1]["VdW_full26"] for item in ladder])
    ax.plot(ns, vdw6, "o-", ms=3, label="VdW6 (Hamiltonian)")
    ax.plot(ns, vdw26, "s-", ms=3, label="VdW26 (diagnostic)")
    ax.axhline(
        exact["VdW_face6"], color="C0", ls="--", lw=0.8, label="exact VdW6"
    )
    ax.axhline(
        exact["VdW_full26"], color="C1", ls=":", lw=0.8, label="exact VdW26"
    )
    ax.axvline(4, color="k", lw=0.6, alpha=0.5)
    ax.annotate(
        "V6.20 uses N=4",
        xy=(4, vdw6[VOXEL_LADDER.index(4)]),
        xytext=(9, vdw6[0]),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    ax.set_xscale("log")
    ax.set_xlabel("voxels per axis N")
    ax.set_ylabel("energy per particle (kBT)")
    ax.set_title("A2  lattice vdW sums at the reference state")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for j, count in enumerate(counts):
        ax.loglog(gaps, grid[:, j], "o-", ms=3, label=f"N={count}")
    ax.set_xlabel("surface gap D (nm)")
    ax.set_ylabel("relative error of one pair energy")
    ax.set_title("A3  the error is set by step / gap")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 1]
    for size_nm, marker in zip(FIT_SIZES_NM, ("o", "s", "^")):
        xs, alphas, radii = [], [], []
        for label in FIT_LADDER:
            fit = fits[(size_nm, str(label))]
            xs.append(label)
            alphas.append(fit["alpha_star"])
            radii.append(fit["a_star"])
        ax.plot(
            xs, alphas, marker + "-", ms=3, label=f"L={size_nm:g} nm"
        )
        fit = fits[(size_nm, "exact")]
        ax.axhline(fit["alpha_star"], ls="--", lw=0.7, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("voxels per axis N")
    ax.set_ylabel("alpha* (deg), constant-gap fit")
    ax.set_title("A4  effect on the fitted rhombohedral angle")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = OUTPUT_DIR / f"A_vdw_convergence.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path.name}")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    print(f"Output directory: {OUTPUT_DIR}")
    curves, orders = section_a1()
    ladder, exact = section_a2()
    gap_data = section_a3()
    fits = section_a4()
    section_a5()
    gap_model = bv.BrushRhombohedralSL(bv.REFERENCE["L_nm"], LEGACY)
    margins = section_a6(
        gap_model.nearest_exact_gap(
            bv.REFERENCE["a_nm"],
            bv.REFERENCE["alpha_deg"],
            bv.REFERENCE["gamma_deg"],
        ),
        fits,
    )
    figure(curves, ladder, exact, gap_data, fits)

    index_four = VOXEL_LADDER.index(4)
    report = {
        "reference_state": {
            "L_nm": bv.REFERENCE["L_nm"],
            "a_nm": bv.REFERENCE["a_nm"],
            "alpha_deg": bv.REFERENCE["alpha_deg"],
            "gamma_deg": bv.REFERENCE["gamma_deg"],
            "gap_nm": bv.BrushRhombohedralSL(
                bv.REFERENCE["L_nm"], LEGACY
            ).nearest_exact_gap(
                bv.REFERENCE["a_nm"],
                bv.REFERENCE["alpha_deg"],
                bv.REFERENCE["gamma_deg"],
            ),
        },
        "observed_convergence_order": orders,
        "single_bond_error_at_N4": {
            label: float(errors[index_four])
            for label, (_, errors, _) in curves.items()
        },
        "lattice_VdW6_kBT": {
            "N4": float(ladder[index_four][1]["VdW_face6"]),
            "exact": float(exact["VdW_face6"]),
            "exact_over_N4": float(
                exact["VdW_face6"] / ladder[index_four][1]["VdW_face6"]
            ),
        },
        "fixed_gap_fit": {
            f"L{size:g}_{label}": {
                "alpha_star_deg": fits[(size, label)]["alpha_star"],
                "a_star_nm": fits[(size, label)]["a_star"],
                "interaction_kBT": fits[(size, label)]["components_star"][
                    "Interaction"
                ],
            }
            for size in FIT_SIZES_NM
            for label in ("4", "exact")
        },
        "alpha_selection_margin_kBT": {
            row[0]: float(row[5]) for row in margins
        },
        "runtime_s": time.perf_counter() - started,
    }
    path = OUTPUT_DIR / "A_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {path.name}")
    print(f"done in {report['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
