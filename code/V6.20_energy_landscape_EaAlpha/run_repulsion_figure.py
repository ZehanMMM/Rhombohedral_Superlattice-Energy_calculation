"""Three-panel figure for the converged model with the fitted repulsion.

Same layout as the V6.20 figure and the F series, but the gap is no longer
imposed: the fitted steric term holds it, so panel C is the *free* landscape
E_total(a, alpha) with the twist relaxed at every node, and its minimum is a
prediction of both coordinates rather than of alpha along a fixed-gap line.

Reads the fitted parameters from R_report.json (run_repulsion_fit.py).  With
``--size`` the same parameters, fitted at 16 nm, are frozen and applied to
another particle size; the lattice window is the 16 nm one scaled with the
centre spacing, and the outputs carry an ``_L<size>`` suffix.

    python run_repulsion_figure.py              # 31 x 30 landscape, ~50 min
    python run_repulsion_figure.py --coarse     # 16 x 16, ~15 min
    python run_repulsion_figure.py --size 18    # transfer to L = 18 nm
    python run_repulsion_figure.py --replot     # redraw from the saved data
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
from scipy.optimize import minimize_scalar

import v620_brush_vdw as bv
import v620_repulsion_fit as rf
from run_figure_reproduction import panel_a

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
FIT_SIZE_NM = 16.0
GAMMAS_LANDSCAPE = np.linspace(-60.0, 60.0, 37, endpoint=False)
# Measured structure as stated in the manuscript: Table S1 gives the 16 nm
# lattice (a = 20.31 nm, alpha = 74.0 deg) and SI S14 gives alpha = 72-76 deg
# for 16-30 nm cubes.  bv.EXP_DATA, ported from the legacy notebook, lists
# alpha = 60 deg for 19.5 and 29.6 nm, which the manuscript does not support,
# so the other sizes are compared with the measured range instead.
MEASURED_ALPHA_RANGE_DEG = (72.0, 76.0)
MEASURED_RANGE_SIZES_NM = (16.0, 30.0)


def measured(size):
    """The measured point at 16 nm, the SI S14 alpha range elsewhere."""
    if size == FIT_SIZE_NM:
        return {"point": bv.EXP_DATA[FIT_SIZE_NM]}
    low, high = MEASURED_RANGE_SIZES_NM
    if low <= size <= high:
        return {"range": MEASURED_ALPHA_RANGE_DEG}
    return {}


def tag(size):
    """File-name suffix: none at the fitted size, ``_L<size>`` elsewhere."""
    return "" if size == FIT_SIZE_NM else f"_L{size:g}"


def lattice_window(size, points):
    """The 16 nm window, 19.2-25.0 nm, scaled with the spacing L + 3.3 nm."""
    scale = (size + 3.3) / (FIT_SIZE_NM + 3.3)
    return np.linspace(19.2 * scale, 25.0 * scale, points)

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


def relaxed(model, a_nm, alpha_deg):
    try:
        return rf.relax_twist_at_fixed_a(
            model, a_nm, alpha_deg, GAMMAS_LANDSCAPE
        )
    except RuntimeError:
        return np.nan, None


def landscape(model, alphas, lattice):
    total = np.full((len(alphas), len(lattice)), np.nan)
    interaction = np.full_like(total, np.nan)
    started = time.perf_counter()
    for i, alpha in enumerate(alphas):
        for j, a_nm in enumerate(lattice):
            _, components = relaxed(model, a_nm, alpha)
            if components is not None:
                total[i, j] = components["Total"]
                interaction[i, j] = components["Interaction"]
        if i % 5 == 0 or i == len(alphas) - 1:
            print(
                f"      row {i + 1}/{len(alphas)} (alpha = {alpha:.1f} deg, "
                f"{time.perf_counter() - started:.0f} s)",
                flush=True,
            )
    return total, interaction


def profile_row_index(alphas, alpha):
    """Landscape row whose alpha is closest to ``alpha``."""
    return int(np.argmin(np.abs(np.asarray(alphas) - float(alpha))))


def alpha_profile(model, alphas, lattice, interaction):
    """E_min(alpha): refine the grid minimum over a at every alpha."""
    rows = []
    for i, alpha in enumerate(alphas):
        row = interaction[i]
        if not np.any(np.isfinite(row)):
            continue
        j = int(np.nanargmin(row))
        low = lattice[max(j - 1, 0)]
        high = lattice[min(j + 1, len(lattice) - 1)]

        def objective(a_nm):
            _, c = relaxed(model, a_nm, alpha)
            return c["Interaction"] if c is not None else 1e12

        best = minimize_scalar(
            objective, bounds=(low, high), method="bounded",
            options={"xatol": 2e-3, "maxiter": 25},
        )
        gamma, c = relaxed(model, float(best.x), alpha)
        rows.append(
            (alpha, float(best.x), -abs(gamma),
             model.nearest_exact_gap(float(best.x), alpha, gamma),
             c["Dipole"], c["VdW"], c["Steric"], c["Interaction"], c["Total"])
        )
    return np.array(rows)


def make_figure(fitted, star, profile, alphas, lattice, total,
                size=FIT_SIZE_NM, manuscript=False):
    model = rf.exponential_model(fitted["P_ref_Pa"], fitted["lambda_nm"],
                                 L_nm=size)
    reference = measured(size)
    experiment = reference.get("point")
    alpha_range = reference.get("range")
    figure = plt.figure(figsize=(11.0, 8.4))
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.0, 1.15), width_ratios=(1.0, 1.25),
        hspace=0.28, wspace=0.30,
    )

    ax_a = figure.add_subplot(grid[0, 0], projection="3d")
    panel_a(ax_a, model, star["a"], star["alpha"], star["gamma"])

    ax_b = figure.add_subplot(grid[0, 1])
    ax_b.axhline(0.0, color="0.5", lw=0.7)
    ax_b.plot(profile[:, 0], profile[:, 4], color="#6a5acd", lw=1.8,
              label="Dipole")
    ax_b.plot(profile[:, 0], profile[:, 5], color="#e8663c", lw=1.8,
              ls="--", label="VdW")
    ax_b.plot(profile[:, 0], profile[:, 6], color="#2e8b57", lw=1.8,
              ls="-.", label="Steric (fitted)")
    ax_b.plot(profile[:, 0], profile[:, 7], color="black", lw=2.4,
              label="Dipole + VdW + Steric")
    if experiment is not None:
        ax_b.axvline(experiment["alpha"], color="0.45", lw=1.0, ls=":",
                     label=r"$\alpha_{\mathrm{exp}}$")
        # The 60 deg experiments sit on the axis edge, so state the cost.
        cost = float(np.interp(experiment["alpha"], profile[:, 0],
                               profile[:, 7])) - star["interaction"]
        ax_b.text(
            0.98, 0.04,
            r"$E(\alpha_{\mathrm{exp}}) - E^*$ = " + f"{cost:+.3f} "
            + r"$k_BT$",
            transform=ax_b.transAxes, ha="right", fontsize=8,
        )
    if alpha_range is not None:
        ax_b.axvspan(*alpha_range, color="#5fd35f", alpha=0.18, lw=0,
                     label=r"measured $\alpha$ range (SI S14)")
        inside = alpha_range[0] <= star["alpha"] <= alpha_range[1]
        ax_b.text(
            0.98, 0.04,
            r"$\alpha^*$ " + ("inside" if inside else "outside")
            + f" the measured {alpha_range[0]:.0f}"
            + f"-{alpha_range[1]:.0f}" + r"$^\circ$",
            transform=ax_b.transAxes, ha="right", fontsize=8,
        )
    ax_b.plot([star["alpha"]], [star["interaction"]], marker="*", ms=14,
              color="#f1c40f", markeredgecolor="black", markeredgewidth=0.6,
              linestyle="none", zorder=5)
    ax_b.annotate(
        r"$\alpha^*$ = " + f"{star['alpha']:.2f}" + r"$^\circ$",
        xy=(star["alpha"], star["interaction"]),
        xytext=(14, 22), textcoords="offset points",
        fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    ax_b.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax_b.set_ylabel(r"Energy ($k_BT$)")
    ax_b.set_title(r"$E$ vs $\alpha$, gap relaxed at every $\alpha$")
    ax_b.set_xlim(alphas[0], alphas[-1])
    # Head room above the curves for the legend; energies grow with L.
    curves = profile[:, 4:8]
    span = float(np.nanmax(curves) - np.nanmin(curves))
    ax_b.set_ylim(top=float(np.nanmax(curves)) + 0.32 * span)
    ax_b.grid(alpha=0.25)
    ax_b.legend(ncol=2, frameon=False, loc="upper left")

    ax_c = figure.add_subplot(grid[1, :])
    finite = total[np.isfinite(total)]
    low = float(finite.min())
    # 10 kBT at 16 nm.  The energies grow with L (about 7x at 29.6 nm), so the
    # window scales with the depth of the minimum.
    window = max(10.0, 1.8 * abs(star["interaction"]))
    high = low + window
    levels = np.linspace(low, high, 26)
    mesh = ax_c.contourf(
        lattice, alphas, np.minimum(total, high), levels=levels,
        cmap="Spectral_r", extend="max",
    )
    ax_c.contour(
        lattice, alphas, np.minimum(total, high), levels=levels,
        colors="white", linewidths=0.3, linestyles="dashed", alpha=0.55,
    )
    ax_c.plot(profile[:, 1], profile[:, 0], color="#4d7fd1", lw=1.6,
              ls="--", label=r"$a^*(\alpha)$, relaxed gap")
    ax_c.axvline(star["a"], color="#e8a33d", lw=1.6, ls="--",
                 label=r"$a$ = " + f"{star['a']:.2f} nm")
    ax_c.axhline(star["alpha"], color="#4fc3d9", lw=1.6, ls="--",
                 label=r"$\alpha$ = " + f"{star['alpha']:.1f}" + r"$^\circ$")
    ax_c.plot([star["a"]], [star["alpha"]], marker="*", ms=16,
              color="#f1c40f", markeredgecolor="black", markeredgewidth=0.6,
              linestyle="none",
              label=f"Sim ({star['a']:.2f} nm, {star['alpha']:.1f}"
              + r"$^\circ$)")
    if experiment is not None:
        ax_c.plot([experiment["d"]], [experiment["alpha"]], marker="^",
                  ms=10, color="#5fd35f", markeredgecolor="black",
                  markeredgewidth=0.6, linestyle="none", zorder=6,
                  label=f"Exp ({experiment['d']:.2f} nm, "
                  f"{experiment['alpha']:.1f}" + r"$^\circ$)")
    if alpha_range is not None:
        for edge in alpha_range:
            ax_c.axhline(edge, color="#2e8b57", lw=1.2, ls=":")
        ax_c.plot([], [], color="#2e8b57", lw=1.2, ls=":",
                  label=f"measured {alpha_range[0]:.0f}-"
                  f"{alpha_range[1]:.0f}" + r"$^\circ$ (SI S14)")
    ax_c.set_xlabel(r"$a$ (nm)")
    ax_c.set_ylabel(r"$\alpha$ ($^\circ$)")
    ax_c.set_title(
        r"$E(a, \alpha)$ landscape, gap free "
        + ("(fitted repulsion)" if size == FIT_SIZE_NM
           else "(repulsion fitted at 16 nm, frozen)")
    )
    ax_c.set_xlim(lattice[0], lattice[-1])
    ax_c.set_ylim(alphas[0], alphas[-1])
    bar = figure.colorbar(mesh, ax=ax_c, pad=0.02)
    bar.set_label(r"$E_{\mathrm{total}} / k_BT$ (clipped at min + "
                  + f"{window:.0f})")
    ax_c.legend(loc="center left", bbox_to_anchor=(1.20, 0.5),
                frameon=False)

    for axis, letter in ((ax_b, "B"), (ax_c, "C")):
        axis.text(-0.08, 1.03, letter, transform=axis.transAxes,
                  fontsize=14, fontweight="bold")
    ax_a.text2D(-0.02, 1.03, "A", transform=ax_a.transAxes, fontsize=14,
                fontweight="bold")
    if not manuscript:
        figure.suptitle(
            f"V6.20, L = {size:g} nm, converged van der Waals, "
            "Ms = 285 kA/m, "
            + ("with the fitted steric repulsion\n" if size == FIT_SIZE_NM
               else "steric repulsion fitted at 16 nm and frozen\n")
            + r"$P(D) = P_{\mathrm{ref}}\, e^{-(D - D_{\mathrm{ref}})/\lambda}$, "
            f"P_ref = {fitted['P_ref_Pa'] / 1e3:.1f} kPa at "
            f"{fitted['reference_gap_nm']} nm, "
            r"$\lambda$ = " + f"{fitted['lambda_nm']:.3f} nm",
            fontsize=10,
        )
    for suffix in ("png", "pdf"):
        variant = "_manuscript" if manuscript else ""
        path = (OUTPUT_DIR
                / f"R_fitted_repulsion_figure{tag(size)}{variant}.{suffix}")
        figure.savefig(path, bbox_inches="tight")
        print(f"  wrote {path.name}")
    plt.close(figure)


def load_saved(size=FIT_SIZE_NM):
    """Landscape, profile and minimum as written by a previous full run."""
    suffix = tag(size)
    path = OUTPUT_DIR / f"R_landscape{suffix}.csv"
    with path.open(encoding="utf-8") as handle:
        rows = [
            (float(r["alpha_deg"]), float(r["a_nm"]), float(r["Total_kBT"]))
            for r in csv.DictReader(handle)
        ]
    alphas = np.unique([row[0] for row in rows])
    lattice = np.unique([row[1] for row in rows])
    total = np.full((len(alphas), len(lattice)), np.nan)
    for alpha, a_nm, value in rows:
        total[np.searchsorted(alphas, alpha),
              np.searchsorted(lattice, a_nm)] = value
    profile = np.loadtxt(OUTPUT_DIR / f"R_alpha_profile{suffix}.csv",
                         delimiter=",", skiprows=1, ndmin=2)
    star = json.loads(
        (OUTPUT_DIR / f"R_figure_report{suffix}.json").read_text("utf-8")
    )["free_landscape_minimum"]
    return alphas, lattice, total, profile, star


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--size", type=float, default=FIT_SIZE_NM)
    # Without the diagnostic super-title, for the manuscript figure.
    parser.add_argument("--manuscript", action="store_true")
    args = parser.parse_args()
    size = float(args.size)
    suffix = tag(size)
    report = json.loads((OUTPUT_DIR / "R_report.json").read_text("utf-8"))
    fitted = report["fitted"]
    if args.replot:
        alphas, lattice, total, profile, star = load_saved(size)
        make_figure(fitted, star, profile, alphas, lattice, total, size,
                    args.manuscript)
        return
    model = rf.exponential_model(fitted["P_ref_Pa"], fitted["lambda_nm"],
                                 L_nm=size)
    alphas = np.linspace(60.0, 90.0, 16 if args.coarse else 31)
    lattice = lattice_window(size, 16 if args.coarse else 30)
    print(f"L = {size:g} nm, lattice window {lattice[0]:.2f}-"
          f"{lattice[-1]:.2f} nm")

    print("panel C landscape")
    total, interaction = landscape(model, alphas, lattice)
    print("panel B profile")
    profile = alpha_profile(model, alphas, lattice, interaction)
    k = int(np.argmin(profile[:, 7]))
    # Refine alpha* between the neighbouring rows.  At each trial alpha the
    # inner solve is the same as for the profile: the lattice parameter is
    # minimised near the grid minimum of row k, the twist relaxed inside.
    row_k = profile_row_index(alphas, profile[k, 0])
    if 0 < k < len(profile) - 1:
        best = minimize_scalar(
            lambda alpha: alpha_profile(
                model, np.array([alpha]), lattice, interaction[[row_k]]
            )[0, 7],
            bounds=(profile[k - 1, 0], profile[k + 1, 0]),
            method="bounded", options={"xatol": 0.02, "maxiter": 20},
        )
        refined = alpha_profile(
            model, np.array([float(best.x)]), lattice, interaction[[row_k]]
        )[0]
    else:
        refined = profile[k]
    refined[2] = -abs(refined[2])  # V6.20 reports the negative mirror member
    star = {
        "alpha": float(refined[0]), "a": float(refined[1]),
        "gamma": float(refined[2]), "gap": float(refined[3]),
        "interaction": float(refined[7]), "total": float(refined[8]),
    }
    print(
        f"  free-landscape minimum: alpha* = {star['alpha']:.3f} deg, "
        f"a* = {star['a']:.4f} nm, gap = {star['gap']:.4f} nm, "
        f"gamma* = {star['gamma']:.3f} deg, Eint = {star['interaction']:.4f}"
    )

    path = OUTPUT_DIR / f"R_landscape{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_deg", "a_nm", "Interaction_kBT", "Total_kBT"])
        for i, alpha in enumerate(alphas):
            for j, a_nm in enumerate(lattice):
                writer.writerow([f"{alpha:.4f}", f"{a_nm:.4f}",
                                 f"{interaction[i, j]:.6f}",
                                 f"{total[i, j]:.6f}"])
    print(f"  wrote {path.name}")
    path = OUTPUT_DIR / f"R_alpha_profile{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_deg", "a_nm", "gamma_deg", "gap_nm",
                         "Dipole_kBT", "VdW6_kBT", "Steric_kBT",
                         "Interaction_kBT", "Total_kBT"])
        for row in profile:
            writer.writerow([f"{v:.6f}" for v in row])
    print(f"  wrote {path.name}")
    # Own file: run_repulsion_fit.py rewrites R_report.json in full, and the
    # two can run concurrently.  Written before plotting so that --replot can
    # redraw after a plotting error without recomputing the landscape.
    (OUTPUT_DIR / f"R_figure_report{suffix}.json").write_text(
        json.dumps({"L_nm": size, "fitted": fitted,
                    "free_landscape_minimum": star}, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote R_figure_report{suffix}.json")

    make_figure(fitted, star, profile, alphas, lattice, total, size,
                args.manuscript)


if __name__ == "__main__":
    main()
