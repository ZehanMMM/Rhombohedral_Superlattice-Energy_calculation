"""Reproduce the three-panel V6.20 figure with the converged van der Waals.

The original figure is the V6.20 constant-gap result computed with the 4x4x4
voxel Hamaker sum.  This script redraws the same three panels with the exact
continuum attraction (``vdw_scheme="continuum"``), and does it twice:

``original``    the magnetic block as written in the notebook,
                Ms = 2.85e5 A/m, K_ani = 2.0e4 J/m^3;
``literature``  the repository's own sourced values,
                Ms = 4.60e5 A/m, |K1| = 1.15e4 J/m^3
                (V11 PARAMETER_SOURCES.md, citing Moreno et al.,
                Sci. Rep. 10, 2722 (2020), DOI 10.1038/s41598-020-58976-7).

Everything else is held at the V6.20 values, including the imposed
nearest-neighbour gap.  There is no ligand brush here: the gap stays a
constraint, exactly as in the original figure, so the only change is the
quadrature of the attraction and the magnetic block.

Panels
    A  the lattice at the fitted state: 3x3x3 block of co-oriented cubes,
       moments along the field, field along the cube body diagonal.
    B  E against alpha at the reference gap, twist relaxed at every alpha.
    C  E_total over (a, alpha), twist relaxed, masked where the surfaces come
       closer than the reference gap.

Panel C is the expensive one: a twist relaxation at every grid node.

    python run_figure_reproduction.py                # both versions
    python run_figure_reproduction.py --version original
    python run_figure_reproduction.py --coarse       # quick look
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
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import v620_brush_vdw as bv
from v620_brush_fitting import (
    fixed_gap_alpha_scan,
    minimise_twist,
    relaxed_state_at_fixed_a,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF = bv.REFERENCE
EXPERIMENT = bv.EXP_DATA[16.0]

VERSIONS = {
    "original": {
        "label": "V6.20 magnetic block",
        "Ms": 285_000.0,
        "K_ani": 2.0e4,
        "note": "Ms = 55 A m^2/kg x 5.18e3 kg/m^3 (sample magnetometry); "
                "K_ani as written in the notebook",
    },
    "literature": {
        "label": "literature magnetic block",
        "Ms": 460_000.0,
        "K_ani": 1.15e4,
        "note": "Ms and |K1| from V11 PARAMETER_SOURCES.md, "
                "Moreno et al. 2020",
    },
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 200,
    }
)


def build(version: str, **kwargs) -> bv.BrushRhombohedralSL:
    """A 16 nm model with the converged attraction and one magnetic block."""
    settings = VERSIONS[version]
    phys = bv.BrushPhys(
        brush_model="off",
        vdw_scheme="continuum",
        vdw_diagnostics=False,
        Ms=settings["Ms"],
        K_ani=settings["K_ani"],
        **kwargs,
    )
    return bv.BrushRhombohedralSL(REF["L_nm"], phys)


def write_csv(name: str, header, rows) -> Path:
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return path


# ----------------------------------------------------------------------
#  Panel A
# ----------------------------------------------------------------------
_CUBE_FACES = (
    (0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
    (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5),
)


def _cube_polygons(centre_nm, rotation, edge_nm):
    signs = np.array(
        [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
        dtype=float,
    )
    corners = centre_nm + (0.5 * edge_nm * signs) @ rotation.T
    return [corners[list(face)] for face in _CUBE_FACES]


def panel_a(ax, model, a_nm, alpha_deg, gamma_deg):
    rotation = model.orientation(gamma_deg)
    basis = model.basis_unit(alpha_deg) * a_nm
    field = np.array([1.0, 0.0, 0.0])
    edge = model.L_nm

    cells = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    centres = cells @ basis
    order = np.argsort(centres @ np.array([0.4, -0.6, -0.7]))

    for index in order:
        centre = centres[index]
        central = not np.any(cells[index])
        polygons = _cube_polygons(centre, rotation, edge)
        # A literal "none" face colour makes Poly3DCollection fail at draw
        # time on matplotlib 3.10; a transparent RGBA does the same job.
        ax.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolors=(0.75, 0.23, 0.17, 0.22) if central
                else (1.0, 1.0, 1.0, 0.0),
                edgecolors="#4a4a4a" if central else "#b8b8b8",
                linewidths=0.9 if central else 0.45,
            )
        )
        if not central:
            tail = centre - 0.34 * edge * field
            ax.quiver(
                *tail, *(0.68 * edge * field),
                color="#4646c8", linewidth=1.0,
                arrow_length_ratio=0.28,
            )

    reach = float(np.abs(centres).max()) + 0.87 * edge
    ax.quiver(
        -1.02 * reach, 0.0, 0.0, 2.25 * reach, 0.0, 0.0,
        color="#e05c5c", linewidth=3.0, arrow_length_ratio=0.10, alpha=0.85,
    )
    ax.text(
        1.32 * reach, 0.0, -0.24 * reach,
        r"$\vec{B}\ \parallel\ [111]_{NC}$",
        color="#c0392b", fontsize=10, fontweight="bold",
    )
    ax.set_xlim(-reach, 1.30 * reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_box_aspect((1.15, 1.0, 1.0), zoom=1.45)
    ax.view_init(elev=16.0, azim=-62.0)
    ax.set_axis_off()
    ax.text2D(
        0.5, 0.02,
        f"$a$ = {a_nm:.2f} nm, "
        r"$\alpha$ = " + f"{alpha_deg:.1f}" + r"$^\circ$, "
        r"$\gamma$ = " + f"{gamma_deg:.1f}" + r"$^\circ$",
        transform=ax.transAxes, ha="center", fontsize=8,
    )


# ----------------------------------------------------------------------
#  Panel B
# ----------------------------------------------------------------------
def panel_b_data(model, gap_nm, alphas, gammas):
    rows = []
    for alpha in alphas:
        gamma, a_nm, components = minimise_twist(model, alpha, gap_nm, gammas)
        rows.append(
            (
                alpha,
                gamma,
                a_nm,
                components["Dipole"],
                components["VdW"],
                components["Interaction"],
                components["Total"],
            )
        )
    return np.array(rows)


def panel_b(ax, data, gap_nm, star):
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.plot(data[:, 0], data[:, 3], color="#6a5acd", lw=1.8, label="Dipole")
    ax.plot(
        data[:, 0], data[:, 4], color="#e8663c", lw=1.8, ls="--", label="VdW"
    )
    ax.plot(
        data[:, 0], data[:, 5], color="black", lw=2.4, label="Dipole + VdW"
    )
    ax.axvline(
        EXPERIMENT["alpha"], color="0.45", lw=1.0, ls=":",
        label=r"$\alpha_{\mathrm{exp}}$",
    )
    ax.plot(
        [star["alpha"]], [star["interaction"]], marker="*", ms=14,
        color="#f1c40f", markeredgecolor="black", markeredgewidth=0.6,
        linestyle="none", label="_nolegend_", zorder=5,
    )
    ax.annotate(
        r"$\alpha^*$ = " + f"{star['alpha']:.2f}" + r"$^\circ$",
        xy=(star["alpha"], star["interaction"]),
        xytext=(star["alpha"] - 1.5, star["interaction"] + 1.1),
        fontsize=8, ha="right",
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    ax.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax.set_ylabel(r"Energy ($k_BT$)")
    ax.set_title(r"$E$ vs $\alpha$ at $D$ = " + f"{gap_nm:.2f} nm")
    ax.set_xlim(data[0, 0], data[-1, 0])
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False, loc="upper left")


# ----------------------------------------------------------------------
#  Panel C
# ----------------------------------------------------------------------
def panel_c_data(model, gap_nm, alphas, contact_nm, a_high_nm, columns):
    """Twist-relaxed landscape on the same contact-aligned mesh as V6.20.

    For every alpha the lattice parameter runs outward from the contact
    branch ``a_contact(alpha)``, where the nearest gap equals the reference
    gap, to a common outer value.  The twist is relaxed only inside the
    feasible set, so no node violates the gap constraint and the left edge of
    the filled region *is* the contact line - the same construction as the
    notebook, which is why no masking is needed.
    """
    fraction = np.linspace(0.0, 1.0, columns)
    a_mesh = (
        contact_nm[:, None]
        + fraction[None, :] * (a_high_nm - contact_nm[:, None])
    )
    alpha_mesh = np.broadcast_to(alphas[:, None], a_mesh.shape).copy()
    total = np.full_like(a_mesh, np.nan)
    interaction = np.full_like(a_mesh, np.nan)
    twist = np.full_like(a_mesh, np.nan)
    gap = np.full_like(a_mesh, np.nan)
    started = time.perf_counter()
    for i, alpha in enumerate(alphas):
        for j in range(a_mesh.shape[1]):
            a_nm = a_mesh[i, j]
            gamma, components = relaxed_state_at_fixed_a(
                model, a_nm, alpha, gap_nm
            )
            if components is None:
                continue
            twist[i, j] = -abs(gamma)
            total[i, j] = components["Total"]
            interaction[i, j] = components["Interaction"]
            gap[i, j] = model.nearest_exact_gap(a_nm, alpha, gamma)
        if i % 10 == 0 or i == len(alphas) - 1:
            print(
                f"      row {i + 1}/{len(alphas)} "
                f"(alpha = {alpha:.1f} deg, "
                f"{time.perf_counter() - started:.0f} s)",
                flush=True,
            )
    worst = np.nanmin(gap) - gap_nm
    print(f"      gap constraint residual: {worst:+.2e} nm")
    if worst < -5e-7:
        raise RuntimeError("The landscape violates the gap constraint.")
    return a_mesh, alpha_mesh, total, interaction, twist


def contact_line(model, alphas, gap_nm, gammas):
    radii, twists = [], []
    for alpha in alphas:
        gamma, a_nm, _ = minimise_twist(model, alpha, gap_nm, gammas)
        radii.append(a_nm)
        twists.append(gamma)
    return np.array(radii), np.array(twists)


def panel_c(ax, figure, a_mesh, alpha_mesh, total, contact, alphas, star):
    levels = np.linspace(np.nanmin(total), np.nanmax(total), 25)
    mesh = ax.contourf(
        a_mesh, alpha_mesh, total, levels=levels, cmap="Spectral_r"
    )
    ax.contour(
        a_mesh, alpha_mesh, total, levels=levels, colors="white",
        linewidths=0.3, linestyles="dashed", alpha=0.55,
    )
    ax.plot(
        contact, alphas, color="#4d7fd1", lw=2.0, ls="--",
        label=r"$D$ = " + f"{star['gap']:.2f} nm",
    )
    ax.axvline(
        star["a"], color="#e8a33d", lw=1.6, ls="--",
        label=r"$a$ = " + f"{star['a']:.2f} nm",
    )
    ax.axhline(
        star["alpha"], color="#4fc3d9", lw=1.6, ls="--",
        label=r"$\alpha$ = " + f"{star['alpha']:.1f}" + r" $^\circ$",
    )
    ax.plot(
        [star["a"]], [star["alpha"]], marker="*", ms=16, color="#f1c40f",
        markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
        label=f"Sim ({star['a']:.2f} nm, {star['alpha']:.1f}"
              + r"$^\circ$)",
    )
    ax.plot(
        [EXPERIMENT["d"]], [EXPERIMENT["alpha"]], marker="^", ms=10,
        color="#5fd35f", markeredgecolor="black", markeredgewidth=0.6,
        linestyle="none",
        label=f"Exp ({EXPERIMENT['d']:.2f} nm, {EXPERIMENT['alpha']:.1f}"
              + r"$^\circ$)",
    )
    ax.set_xlabel(r"$a$ (nm)")
    ax.set_ylabel(r"$\alpha$ ($^\circ$)")
    ax.set_title(r"$E(a, \alpha)$ landscape")
    ax.set_xlim(float(np.nanmin(a_mesh)) - 0.3, float(np.nanmax(a_mesh)))
    ax.set_ylim(alphas[0], alphas[-1])
    bar = figure.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label(r"$E_{\mathrm{total}} / k_BT$")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.20, 0.5), frameon=False,
    )


# ----------------------------------------------------------------------
def run_version(version: str, coarse: bool):
    settings = VERSIONS[version]
    print(f"\n=== {version}: {settings['label']} ===")
    print(f"    Ms = {settings['Ms']:.4g} A/m, "
          f"K_ani = {settings['K_ani']:.4g} J/m^3, converged vdW")
    model = build(version)
    gap_nm = bv.BrushRhombohedralSL(
        REF["L_nm"], bv.BrushPhys(brush_model="off")
    ).nearest_exact_gap(REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"])
    print(f"    imposed gap D_ref = {gap_nm:.6f} nm")

    alphas = np.linspace(60.0, 90.0, 31 if coarse else 61)
    gammas = np.linspace(-60.0, 60.0, 13 if coarse else 19, endpoint=False)

    fit = fixed_gap_alpha_scan(model, gap_nm, alphas, gammas)
    star = {
        "alpha": fit["alpha_star"],
        "gamma": fit["gamma_star"],
        "a": fit["a_star"],
        "gap": gap_nm,
        "interaction": fit["components_star"]["Interaction"],
        "total": fit["components_star"]["Total"],
        "dipole": fit["components_star"]["Dipole"],
        "vdw": fit["components_star"]["VdW"],
    }
    print(
        f"    minimum: alpha* = {star['alpha']:.3f} deg, "
        f"gamma* = {star['gamma']:.3f} deg, a* = {star['a']:.4f} nm, "
        f"Edd = {star['dipole']:.4f}, VdW = {star['vdw']:.4f}, "
        f"Eint = {star['interaction']:.4f} kBT"
    )

    print("    panel B")
    data_b = panel_b_data(model, gap_nm, alphas, gammas)
    write_csv(
        f"F_{version}_panelB_alpha_scan.csv",
        ["alpha_deg", "gamma_opt_deg", "a_nm", "Dipole_kBT", "VdW6_kBT",
         "Interaction_kBT", "Total_kBT"],
        [[f"{value:.6f}" for value in row] for row in data_b],
    )

    print("    panel C (twist relaxed at every node)")
    contact, _ = contact_line(model, alphas, gap_nm, gammas)
    a_high = float(np.ceil((max(contact.max(), EXPERIMENT["d"]) + 7.0) * 2.0)
                   / 2.0)
    a_mesh, alpha_mesh, total, interaction, twist = panel_c_data(
        model, gap_nm, alphas, contact, a_high, 41 if coarse else 81
    )
    rows = []
    for i in range(a_mesh.shape[0]):
        for j in range(a_mesh.shape[1]):
            rows.append(
                [
                    f"{alpha_mesh[i, j]:.4f}", f"{a_mesh[i, j]:.5f}",
                    f"{twist[i, j]:.4f}", f"{interaction[i, j]:.6f}",
                    f"{total[i, j]:.6f}",
                ]
            )
    write_csv(
        f"F_{version}_panelC_landscape.csv",
        ["alpha_deg", "a_nm", "gamma_opt_deg", "Interaction_kBT",
         "Total_kBT"],
        rows,
    )

    figure = plt.figure(figsize=(11.0, 8.4))
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.0, 1.15), width_ratios=(1.0, 1.25),
        hspace=0.28, wspace=0.30,
    )
    ax_a = figure.add_subplot(grid[0, 0], projection="3d")
    panel_a(ax_a, model, star["a"], star["alpha"], star["gamma"])
    ax_b = figure.add_subplot(grid[0, 1])
    panel_b(ax_b, data_b, gap_nm, star)
    ax_c = figure.add_subplot(grid[1, :])
    panel_c(ax_c, figure, a_mesh, alpha_mesh, total, contact, alphas, star)

    for axis, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        axis.text2D(
            -0.02, 1.03, letter, transform=axis.transAxes,
            fontsize=14, fontweight="bold",
        ) if letter == "A" else axis.text(
            -0.08, 1.03, letter, transform=axis.transAxes,
            fontsize=14, fontweight="bold",
        )
    figure.suptitle(
        f"V6.20 with the converged van der Waals integral, "
        f"{settings['label']}\n"
        f"Ms = {settings['Ms'] / 1e3:.0f} kA/m, "
        f"K = {settings['K_ani']:.3g} J/m$^3$, A = 2.0e-20 J, "
        f"gap imposed at {gap_nm:.3f} nm",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        path = OUTPUT_DIR / f"F_reproduction_{version}.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        print(f"  wrote {path.name}")
    plt.close(figure)
    return star


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", choices=("original", "literature", "both"), default="both"
    )
    parser.add_argument("--coarse", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    names = (
        ("original", "literature") if args.version == "both"
        else (args.version,)
    )
    summary = {name: run_version(name, args.coarse) for name in names}
    summary["experiment"] = {
        "a_nm": EXPERIMENT["d"], "alpha_deg": EXPERIMENT["alpha"]
    }
    summary["runtime_s"] = time.perf_counter() - started
    (OUTPUT_DIR / "F_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n  wrote F_report.json")
    print(f"done in {summary['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
