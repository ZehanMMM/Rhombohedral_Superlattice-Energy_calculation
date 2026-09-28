"""What the Alexander-de Gennes brush does to the V6.20 rhombohedral fit.

The V6.20 landscape pins the nearest-neighbour surface gap at the value
calibrated from the experimental reference state, so its steric energy is
identically zero along the whole scanned path and only the angle alpha is
predicted.  Adding a ligand-brush repulsion makes the gap a free coordinate:
the lattice parameter is then set by the balance between the Hamaker
attraction and the brush, and (a, alpha, gamma) are all predictions.

Sections

B1  The brush term evaluated at the experimental reference state, bond by
    bond, for both brush models.
B2  Energy against gap at the experimental angle, for the legacy N = 4
    attraction and for the exact one, with and without the brush.  Also finds
    where the Hamaker divergence finally overtakes the brush.
B3  Calibration of the brush height against the experimental gap, separately
    for the two attraction rules.
B4  The 2 x 2 refit: {N = 4, exact} x {fixed gap, free gap}.  This is the
    comparison the whole study is for.
B5  Sensitivity to the brush parameters and to the brush model.
B6  Sizes with experimental data, predicted with the calibrated brush.

Outputs go to ``outputs/V6.20_energy_landscape_EaAlpha/brush_vdw``.

    python run_brush_refit.py            # full run, tens of minutes
    python run_brush_refit.py --quick    # coarse grids, for plumbing checks
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann

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

KT = Boltzmann * 298.15
REF = bv.REFERENCE
SIZES_NM = (16.0, 13.0, 19.5)
EXP_SIZES_NM = (13.0, 16.0, 18.0, 18.2, 19.5, 29.6)

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


def phys(vdw: str, **kwargs) -> bv.BrushPhys:
    """Base parameter set; ``vdw`` is "legacy" (N=4) or "exact"."""
    base = dict(vdw_diagnostics=False)
    if vdw == "legacy":
        base.update(vdw_scheme="voxel", voxel_count=4)
    elif vdw == "exact":
        base.update(vdw_scheme="continuum")
    else:
        raise ValueError(vdw)
    base.update(kwargs)
    return bv.BrushPhys(**base)


def reference_gap() -> float:
    model = bv.BrushRhombohedralSL(REF["L_nm"], phys("legacy",
                                                     brush_model="off"))
    return model.nearest_exact_gap(
        REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
    )


# ----------------------------------------------------------------------
def section_b1():
    print("B1  the brush term at the experimental reference state")
    model = bv.BrushRhombohedralSL(REF["L_nm"], phys("exact"))
    offsets = model.neighbour_vectors(
        REF["a_nm"], REF["alpha_deg"]
    ) @ model.orientation(REF["gamma_deg"])
    gaps = model.neighbour_exact_gaps(
        REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
    ) * 1e9
    rows = []
    for index in np.argsort(gaps):
        offset = offsets[index]
        derjaguin = model.derjaguin_repulsion_pair_J(offset)
        facet = model.facet_repulsion_pair_J(offset)
        attraction = model.vdw_pair_energy_J(offset)
        rows.append(
            [
                "".join(f"{i:+d}" for i in model.neighbour_indices[index]),
                bool(model.primitive_face_mask[index]),
                f"{gaps[index]:.6f}",
                f"{derjaguin / KT:.6f}",
                f"{facet / KT:.6f}",
                f"{attraction / KT:.6f}",
            ]
        )
    write_csv(
        "B1_reference_state_bonds.csv",
        [
            "neighbour_index",
            "is_primitive_bond",
            "exact_gap_nm",
            "U_rep_derjaguin_kBT",
            "U_rep_facet_kBT",
            "U_attr_exact_kBT",
        ],
        rows,
    )
    totals = {}
    for label, brush in (("derjaguin", "derjaguin"), ("facet", "facet")):
        for vdw in ("legacy", "exact"):
            sub = bv.BrushRhombohedralSL(
                REF["L_nm"], phys(vdw, brush_model=brush)
            )
            components = sub.energy_components(
                REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"]
            )
            totals[f"{label}_{vdw}"] = {
                key: float(components[key])
                for key in ("Dipole", "VdW", "Steric", "Interaction")
            }
            print(
                f"    {label:9s} vdW={vdw:6s}  Dip={components['Dipole']:8.4f}"
                f"  VdW={components['VdW']:8.4f}"
                f"  Steric={components['Steric']:8.4f}"
                f"  Interaction={components['Interaction']:8.4f} kBT"
            )
    return totals


# ----------------------------------------------------------------------
def section_b2(quick: bool):
    print("B2  energy against gap at the experimental angle")
    alpha = REF["alpha_deg"]
    gamma = REF["gamma_deg"]
    # The brush diverges as D^(-5/4) and the Hamaker attraction as D^(-2),
    # so the attraction must win again at some very small gap.  The grid
    # reaches 1e-4 nm so that crossover is inside it and can be reported
    # rather than assumed.
    gaps = np.concatenate(
        [
            np.geomspace(1e-4, 0.2, 16 if quick else 32, endpoint=False),
            np.linspace(0.2, 3.999, 40 if quick else 120),
        ]
    )
    rows = []
    curves = {}
    for vdw in ("legacy", "exact"):
        model = bv.BrushRhombohedralSL(REF["L_nm"], phys(vdw))
        attraction, repulsion, radii = [], [], []
        for gap in gaps:
            a_nm = model.a_for_exact_gap(alpha, gamma, gap)
            components = model.energy_components(a_nm, alpha, gamma)
            attraction.append(
                components["Dipole"] + components["VdW"]
            )
            repulsion.append(components["Steric"])
            radii.append(a_nm)
        attraction = np.array(attraction)
        repulsion = np.array(repulsion)
        total = attraction + repulsion
        curves[vdw] = (gaps, attraction, repulsion, total, np.array(radii))
        for gap, att, rep, tot, a_nm in zip(
            gaps, attraction, repulsion, total, radii
        ):
            rows.append(
                [
                    vdw,
                    f"{gap:.6f}",
                    f"{a_nm:.5f}",
                    f"{att:.6f}",
                    f"{rep:.6f}",
                    f"{tot:.6f}",
                ]
            )
        # Where does the attraction overtake the brush again?
        crossover = np.nan
        sign = np.sign(np.gradient(total, gaps))
        inner = np.where(gaps < 1.0)[0]
        turning = inner[np.where(np.diff(sign[inner]) != 0)[0]]
        if turning.size:
            crossover = float(gaps[turning[0]])
        relaxed = model.relax_gap(alpha, gamma, grid_points=48)
        print(
            f"    {vdw:6s} relaxed gap={relaxed['gap_nm']:.4f} nm  "
            f"a={relaxed['a_nm']:.4f} nm  "
            f"Interaction={relaxed['components']['Interaction']:.4f} kBT  "
            f"minimum of the scanned curve at "
            f"{gaps[int(np.argmin(total))]:.4f} nm  "
            f"inner turning point={crossover}"
        )
    write_csv(
        "B2_energy_versus_gap.csv",
        [
            "vdw_rule",
            "gap_nm",
            "a_nm",
            "dipole_plus_vdw_kBT",
            "steric_kBT",
            "interaction_kBT",
        ],
        rows,
    )
    return curves


# ----------------------------------------------------------------------
def section_b3(target_gap: float):
    print("B3  brush height calibrated against the experimental gap")
    rows = []
    calibrated = {}
    for vdw in ("legacy", "exact"):
        for brush in ("derjaguin", "facet"):
            def build(height_nm, vdw=vdw, brush=brush):
                return bv.BrushRhombohedralSL(
                    REF["L_nm"],
                    phys(vdw, brush_model=brush, brush_height_nm=height_nm),
                )

            try:
                height = calibrate_brush_height(
                    build,
                    target_gap,
                    REF["alpha_deg"],
                    gamma_deg=REF["gamma_deg"],
                    bracket_nm=(1.2, 3.4),
                )
            except (RuntimeError, ValueError) as error:
                print(f"    {vdw:6s} {brush:9s} calibration failed: {error}")
                rows.append([vdw, brush, "nan", "nan", str(error)])
                continue
            model = build(height)
            relaxed = model.relax_gap(REF["alpha_deg"], REF["gamma_deg"])
            calibrated[(vdw, brush)] = height
            rows.append(
                [
                    vdw,
                    brush,
                    f"{height:.5f}",
                    f"{relaxed['gap_nm']:.6f}",
                    f"{relaxed['components']['Interaction']:.5f}",
                ]
            )
            print(
                f"    {vdw:6s} {brush:9s} L0={height:.4f} nm gives a relaxed "
                f"gap of {relaxed['gap_nm']:.4f} nm"
            )
    write_csv(
        "B3_brush_height_calibration.csv",
        [
            "vdw_rule",
            "brush_model",
            "calibrated_brush_height_nm",
            "relaxed_gap_nm",
            "interaction_kBT",
        ],
        rows,
    )
    return calibrated


# ----------------------------------------------------------------------
def section_b4(target_gap: float, calibrated: dict, quick: bool):
    print("B4  the 2 x 2 refit")
    alphas = np.linspace(60.0, 90.0, 16 if quick else 31)
    gammas = np.linspace(-60.0, 60.0, 13 if quick else 19, endpoint=False)
    rows = []
    fits = {}
    for size_nm in (SIZES_NM[:1] if quick else SIZES_NM):
        for vdw in ("legacy", "exact"):
            model = bv.BrushRhombohedralSL(
                size_nm, phys(vdw, brush_model="off")
            )
            started = time.perf_counter()
            fit = fixed_gap_alpha_scan(model, target_gap, alphas, gammas)
            fits[(size_nm, vdw, "fixed_gap")] = fit
            rows.append(
                _fit_row(size_nm, vdw, "fixed_gap", "n/a", fit, started)
            )
            _report(size_nm, vdw, "fixed_gap", fit, started)

            for tag, height in (
                ("default", bv.BrushPhys().brush_height_nm),
                ("calibrated", calibrated.get((vdw, "derjaguin"), np.nan)),
            ):
                if not np.isfinite(height):
                    continue
                if tag == "calibrated" and (quick or size_nm != 16.0):
                    # The controlled legacy-versus-exact comparison is the
                    # default-brush row; the calibrated brush is carried
                    # through the size series in B6 instead.
                    continue
                model = bv.BrushRhombohedralSL(
                    size_nm, phys(vdw, brush_height_nm=height)
                )
                started = time.perf_counter()
                fit = free_gap_alpha_scan(
                    model,
                    alphas,
                    gammas,
                    passes=2,
                    grid_points=24 if quick else 32,
                )
                fits[(size_nm, vdw, f"free_gap_{tag}")] = fit
                rows.append(
                    _fit_row(
                        size_nm, vdw, f"free_gap_{tag}", height, fit, started
                    )
                )
                _report(size_nm, vdw, f"free_gap_{tag}", fit, started)
    write_csv(
        "B4_refit_two_by_two.csv",
        [
            "L_nm",
            "vdw_rule",
            "ensemble",
            "brush_height_nm",
            "alpha_star_deg",
            "gamma_star_deg",
            "a_star_nm",
            "gap_star_nm",
            "Dipole_kBT",
            "VdW6_kBT",
            "Steric_kBT",
            "Interaction_kBT",
            "runtime_s",
        ],
        rows,
    )
    return fits


def _fit_row(size_nm, vdw, ensemble, height, fit, started):
    star = fit["components_star"]
    return [
        f"{size_nm:g}",
        vdw,
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
        f"{time.perf_counter() - started:.1f}",
    ]


def _report(size_nm, vdw, ensemble, fit, started):
    star = fit["components_star"]
    print(
        f"    L={size_nm:>4g} {vdw:6s} {ensemble:18s} "
        f"alpha*={fit['alpha_star']:7.3f}  a*={fit['a_star']:8.4f}  "
        f"D*={fit['gap_star_nm']:6.3f}  "
        f"Steric={star['Steric']:7.3f}  "
        f"Eint={star['Interaction']:8.3f} kBT  "
        f"[{time.perf_counter() - started:.0f} s]",
        flush=True,
    )


# ----------------------------------------------------------------------
def section_b5(quick: bool):
    print("B5  sensitivity to the brush parameters")
    heights = (1.6, 1.8, 2.0, 2.2, 2.4)
    densities = (1.5, 2.5, 4.0)
    rows = []
    grid = np.full((len(heights), len(densities)), np.nan)
    for i, height in enumerate(heights):
        for j, density in enumerate(densities):
            spacing = float(density) ** -0.5
            for brush in ("derjaguin", "facet"):
                model = bv.BrushRhombohedralSL(
                    REF["L_nm"],
                    phys(
                        "exact",
                        brush_model=brush,
                        brush_height_nm=height,
                        grafting_spacing_nm=spacing,
                    ),
                )
                relaxed = model.relax_gap(
                    REF["alpha_deg"], REF["gamma_deg"], grid_points=40
                )
                if not relaxed["converged"]:
                    rows.append(
                        [brush, height, density, "nan", "nan", "nan"]
                    )
                    continue
                if brush == "derjaguin":
                    grid[i, j] = relaxed["gap_nm"]
                rows.append(
                    [
                        brush,
                        f"{height:.2f}",
                        f"{density:.2f}",
                        f"{relaxed['gap_nm']:.5f}",
                        f"{relaxed['a_nm']:.5f}",
                        f"{relaxed['components']['Interaction']:.5f}",
                    ]
                )
        print(f"    brush height {height:.1f} nm done", flush=True)
    write_csv(
        "B5_brush_parameter_sensitivity.csv",
        [
            "brush_model",
            "brush_height_nm",
            "grafting_density_per_nm2",
            "relaxed_gap_nm",
            "a_nm",
            "interaction_kBT",
        ],
        rows,
    )
    return heights, densities, grid


# ----------------------------------------------------------------------
def section_b6(calibrated: dict, quick: bool):
    print("B6  sizes with experimental data, calibrated brush")
    height = calibrated.get(("exact", "derjaguin"), np.nan)
    if not np.isfinite(height):
        print("    skipped: no calibrated brush height")
        return {}
    alphas = np.linspace(60.0, 90.0, 11 if quick else 16)
    gammas = np.linspace(-60.0, 60.0, 13, endpoint=False)
    rows = []
    fits = {}
    for size_nm in (EXP_SIZES_NM[:2] if quick else EXP_SIZES_NM):
        model = bv.BrushRhombohedralSL(
            size_nm, phys("exact", brush_height_nm=height)
        )
        started = time.perf_counter()
        fit = free_gap_alpha_scan(
            model, alphas, gammas, passes=2, grid_points=28
        )
        fits[size_nm] = fit
        experiment = bv.EXP_DATA.get(size_nm)
        rows.append(
            [
                f"{size_nm:g}",
                f"{height:.5f}",
                f"{fit['alpha_star']:.4f}",
                f"{fit['a_star']:.5f}",
                f"{fit['gap_star_nm']:.6f}",
                f"{fit['components_star']['Interaction']:.5f}",
                f"{experiment['alpha']:.2f}" if experiment else "",
                f"{experiment['d']:.2f}" if experiment else "",
            ]
        )
        print(
            f"    L={size_nm:>4g}  alpha*={fit['alpha_star']:7.3f}"
            f" (exp {experiment['alpha'] if experiment else '--'})"
            f"  a*={fit['a_star']:8.4f}"
            f" (exp {experiment['d'] if experiment else '--'})"
            f"  [{time.perf_counter() - started:.0f} s]",
            flush=True,
        )
    write_csv(
        "B6_size_series_prediction.csv",
        [
            "L_nm",
            "brush_height_nm",
            "alpha_star_deg",
            "a_star_nm",
            "gap_star_nm",
            "Interaction_kBT",
            "alpha_experiment_deg",
            "a_experiment_nm",
        ],
        rows,
    )
    return fits


# ----------------------------------------------------------------------
def figures(curves, fits, sensitivity, size_fits, target_gap):
    heights, densities, grid = sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))

    ax = axes[0, 0]
    for vdw, colour in (("legacy", "C0"), ("exact", "C3")):
        gaps, attraction, repulsion, total, _ = curves[vdw]
        ax.plot(gaps, attraction, colour, ls="--", lw=1.0,
                label=f"{vdw}: dipole + vdW")
        ax.plot(gaps, total, colour, ls="-", label=f"{vdw}: total")
    gaps, _, repulsion, _, _ = curves["exact"]
    ax.plot(gaps, repulsion, "k:", label="AdG brush")
    ax.axvline(target_gap, color="grey", lw=0.7)
    ax.annotate(
        f"experimental gap\n{target_gap:.3f} nm",
        xy=(target_gap, -4.0),
        xytext=(1.2, -12.0),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    ax.set_xlim(0.5, 4.0)
    ax.set_ylim(-14.0, 14.0)
    ax.set_xlabel("nearest-neighbour surface gap D (nm)")
    ax.set_ylabel("energy per particle (kBT)")
    ax.set_title("B2  the gap is now a free coordinate")
    ax.legend(frameon=False, ncol=1)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for key, style in (
        (("legacy", "fixed_gap"), ("C0", "--")),
        (("exact", "fixed_gap"), ("C3", "--")),
        (("legacy", "free_gap_default"), ("C0", "-")),
        (("exact", "free_gap_default"), ("C3", "-")),
    ):
        fit = fits.get((16.0, key[0], key[1]))
        if fit is None:
            continue
        colour, dash = style
        ax.plot(
            fit["alphas"],
            fit["interaction"],
            color=colour,
            ls=dash,
            label=f"{key[0]}, {key[1].replace('_', ' ')}",
        )
        ax.plot(
            [fit["alpha_star"]],
            [fit["components_star"]["Interaction"]],
            marker="o",
            ms=4,
            color=colour,
        )
    ax.axvline(
        bv.EXP_DATA[16.0]["alpha"], color="grey", lw=0.7,
        label="experimental alpha",
    )
    ax.set_xlabel("alpha (deg)")
    ax.set_ylabel("Interaction at the relaxed state (kBT)")
    ax.set_title("B4  alpha scans, L = 16 nm")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for j, density in enumerate(densities):
        ax.plot(
            heights,
            grid[:, j],
            "o-",
            ms=3,
            label=f"{density:.1f} chains per nm2",
        )
    ax.axhline(
        target_gap, color="grey", lw=0.7, label="experimental gap"
    )
    ax.set_xlabel("unstressed brush height L0 (nm)")
    ax.set_ylabel("relaxed gap D* (nm)")
    ax.set_title("B5  the gap is set by the brush, not by the vdW rule")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    if size_fits:
        sizes = sorted(size_fits)
        ax.plot(
            sizes,
            [size_fits[s]["a_star"] for s in sizes],
            "o-",
            ms=4,
            label="predicted a*",
        )
        experiment = [
            (s, bv.EXP_DATA[s]["d"]) for s in sizes if s in bv.EXP_DATA
        ]
        if experiment:
            ax.plot(
                [item[0] for item in experiment],
                [item[1] for item in experiment],
                "k*",
                ms=8,
                label="experiment",
            )
    ax.set_xlabel("particle edge L (nm)")
    ax.set_ylabel("lattice parameter a (nm)")
    ax.set_title("B6  size series with the calibrated brush")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = OUTPUT_DIR / f"B_brush_refit.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path.name}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    print(f"Output directory: {OUTPUT_DIR}")
    target_gap = reference_gap()
    print(f"experimental reference gap D_ref = {target_gap:.6f} nm")

    totals = section_b1()
    curves = section_b2(args.quick)
    calibrated = section_b3(target_gap)
    fits = section_b4(target_gap, calibrated, args.quick)
    sensitivity = section_b5(args.quick)
    size_fits = section_b6(calibrated, args.quick)
    figures(curves, fits, sensitivity, size_fits, target_gap)

    report = {
        "reference_gap_nm": target_gap,
        "reference_state_energies_kBT": totals,
        "calibrated_brush_height_nm": {
            f"{vdw}_{brush}": value
            for (vdw, brush), value in calibrated.items()
        },
        "refit": {
            f"L{size:g}_{vdw}_{ensemble}": {
                "alpha_star_deg": fit["alpha_star"],
                "a_star_nm": fit["a_star"],
                "gap_star_nm": fit["gap_star_nm"],
                "interaction_kBT": fit["components_star"]["Interaction"],
                "steric_kBT": fit["components_star"]["Steric"],
            }
            for (size, vdw, ensemble), fit in fits.items()
        },
        "experiment": {
            f"L{size:g}": bv.EXP_DATA[size] for size in bv.EXP_DATA
        },
        "runtime_s": time.perf_counter() - started,
    }
    path = OUTPUT_DIR / "B_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {path.name}")
    print(f"done in {report['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
