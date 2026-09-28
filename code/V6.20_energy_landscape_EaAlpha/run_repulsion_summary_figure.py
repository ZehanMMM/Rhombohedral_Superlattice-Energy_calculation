"""Model summary figure: converged attraction, fitted repulsion, transfer.

Six panels drawn from the files written by run_vdw_convergence.py (A2),
run_repulsion_family_probe.py (R0), run_repulsion_fit.py (R2-R4),
run_repulsion_transfer_check.py (R5), run_brush_refit.py (B3, B6) and
run_figure_reproduction.py (F).  Nothing is recomputed except the two areal
laws of panel a, which are closed form.

    a  the fitted exponential law against the calibrated AdG brush
    b  R0: which repulsion families can move the angle at all
    c  A2: the 4 x 4 x 4 voxel vdW sum against its exact continuum limit
    d  E_min(alpha) before (attraction only, fixed gap) and after the fit
    e  R2/R3: the valley floor, and why its vertex depends on the window
    f  R4/R5: the size transfer with the parameters frozen, against the
       measured structure as stated in the manuscript

    python run_repulsion_summary_figure.py      # a few seconds
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann

import v620_brush_vdw as bv

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
TEMPERATURE_K = bv.BrushPhys().temperature_K
KT_PER_NM2 = 1e-18 / (Boltzmann * TEMPERATURE_K)   # J/m^2 -> kBT/nm^2
ALPHA_EXP = bv.EXP_DATA[16.0]["alpha"]
# Measured structure as stated in the manuscript: Table S1 (16 nm lattice),
# SI S14 (alpha = 72-76 deg for 16-30 nm; simple cubic for 13 nm, whose
# moments are not blocked at room temperature).  bv.EXP_DATA from the legacy
# notebook lists alpha = 60 deg for 19.5 and 29.6 nm, which the manuscript
# does not support; those entries are not drawn.
MEASURED_16NM = bv.EXP_DATA[16.0]
MEASURED_ALPHA_RANGE_DEG = (72.0, 76.0)
MEASURED_RANGE_SIZES_NM = (16.0, 30.0)
MEASURED_13NM_ALPHA_DEG = 90.0

# Panel titles: (diagnostic, SI).  The diagnostic set names the output series
# (R0, R2/R3, ...) the panels come from; the SI set does not.
PANEL_TITLES = {
    "a": ("a   fitted law vs the AdG brush",
          "A   steric law: fitted vs AdG brush"),
    "b": ("b   R0: which laws can move the angle",
          "B   which repulsion laws can move the angle"),
    "c": ("c   vdW: voxel sum vs exact continuum",
          "C   vdW: voxel sum vs exact continuum"),
    "d": (r"d   $E_{\min}(\alpha)$ before and after the fit",
          r"D   $E_{\min}(\alpha)$ before and after the fit"),
    "e": ("e   R2/R3: the valley floor is flat",
          "E   valley floor near the minimum"),
    "f": ("f   transfer to other sizes, parameters frozen",
          "F   other sizes, parameters frozen"),
}
VALLEY_LABELS = {
    "stencil": ("fit stencil (0.4$^\\circ$, R2)",
                "0.4$^\\circ$ stencil (fit)"),
    "fine": ("fine scan (0.2$^\\circ$, R3)", "0.2$^\\circ$ scan"),
}
SI = 0          # index into the tuples above; set to 1 by --si


def title(key):
    return PANEL_TITLES[key][SI]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "figure.dpi": 200,
    }
)


def read_csv(name):
    with (OUTPUT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def column(rows, key):
    return np.array([float(row[key]) if row[key] != "" else np.nan
                     for row in rows])


def read_probe():
    """(family, shape, dE/dalpha or None) from R0_repulsion_family_probe.log."""
    family = None
    points = []
    text = (OUTPUT_DIR / "R0_repulsion_family_probe.log").read_text("utf-8")
    for line in text.splitlines():
        if line.startswith("---"):
            family = line[3:].split("(")[0].strip()
            continue
        match = re.search(r"shape=([0-9.]+)", line)
        if family is None or match is None:
            continue
        slope = re.search(r"dE/dalpha=([-+0-9.e]+)", line)
        points.append((family, float(match.group(1)),
                       float(slope.group(1)) if slope else None))
    return points


def panel_laws(ax, fitted, adg_height_nm, adg_gap_nm):
    gap_nm = np.linspace(1.0, 5.0, 600)
    decay = fitted["lambda_nm"]
    reference = fitted["reference_gap_nm"]
    soft = bv.exponential_steric_energy_J_per_m2(
        gap_nm * 1e-9, fitted["P_ref_Pa"], decay * 1e-9,
        (reference + 8.0 * decay) * 1e-9, reference * 1e-9,
    ) * KT_PER_NM2
    spacing = bv.BrushPhys().grafting_spacing_nm
    brush = bv.adg_plate_energy_J_per_m2(
        gap_nm * 1e-9, adg_height_nm * 1e-9, spacing * 1e-9, TEMPERATURE_K,
    ) * KT_PER_NM2
    ax.semilogy(gap_nm, np.where(soft > 0, soft, np.nan), color="#2e8b57",
                lw=1.8, label=rf"fitted: $\lambda$ = {decay:.3f} nm")
    ax.semilogy(gap_nm, np.where(brush > 0, brush, np.nan), color="0.25",
                lw=1.4, ls="--",
                label=("AdG" if SI else "AdG (B3)")
                + rf": $L_0$ = {adg_height_nm:.3f} nm, "
                rf"$\sigma$ = {1.0 / spacing**2:.1f} nm$^{{-2}}$")
    for gap, values, colour in ((fitted["gap_at_target_nm"], soft, "#2e8b57"),
                                (adg_gap_nm, brush, "0.25")):
        value = float(np.interp(gap, gap_nm, values))
        ax.plot([gap], [value], "o", ms=5, color=colour, zorder=5)
        ax.annotate(f"D* = {gap:.2f} nm\n{value:.1e} kBT/nm$^2$",
                    xy=(gap, value), xytext=(6, -22 if colour == "0.25"
                                              else 8),
                    textcoords="offset points", fontsize=6.5, color=colour)
    ax.set_ylim(1e-5, 30.0)
    ax.set_xlim(gap_nm[0], gap_nm[-1])
    ax.set_xlabel("surface gap D (nm)")
    ax.set_ylabel(r"areal energy $w(D)$ ($k_BT$/nm$^2$)")
    ax.set_title(title("a"))
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25, which="both", lw=0.4)


def panel_probe(ax, fitted, attraction_slope):
    styles = {
        "exponential, Derjaguin area rule": ("#2e8b57", "o", "exponential, Derjaguin"),
        "exponential, facet-only area rule": ("#e8a33d", "s", "exponential, facet only"),
        "AdG, Derjaguin": ("0.25", "D", "AdG, Derjaguin (x = spacing s)"),
    }
    points = read_probe()
    for family, (colour, marker, label) in styles.items():
        rows = [(s, v) for f, s, v in points if f == family]
        ok = [(s, v) for s, v in rows if v is not None]
        if ok:
            ax.plot(*zip(*ok), marker=marker, ms=4, color=colour, lw=1.0,
                    label=label)
        failed = [s for s, v in rows if v is None]
        if failed:
            ax.plot(failed, [0.13] * len(failed), marker="x", ls="none",
                    color=colour, ms=5)
            ax.text(0.98, 0.875, "x : facet only, no force balance",
                    transform=ax.transAxes, ha="right", fontsize=6,
                    color=colour)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(attraction_slope, color="#c0392b", lw=0.8, ls=":",
               label="attraction alone")
    ax.axvline(fitted["lambda_nm"], color="#2e8b57", lw=0.8, ls="--")
    ax.text(fitted["lambda_nm"] * 1.06, -0.085,
            rf"fit: $\lambda$ = {fitted['lambda_nm']:.2f} nm",
            color="#2e8b57", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.095, 0.145)
    ax.set_xlabel(r"shape parameter: $\lambda$ or $s$ (nm)")
    ax.set_ylabel(r"$dE/d\alpha$ after force balance ($k_BT$/deg)")
    ax.set_title(title("b"))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80),
              fontsize=6)
    ax.grid(alpha=0.25, which="both", lw=0.4)


def panel_vdw_convergence(ax):
    rows = [row for row in read_csv("A2_lattice_vdw_vs_voxels.csv")
            if row["voxel_count"] != "exact"]
    voxels = column(rows, "voxel_count")
    error = column(rows, "relative_error")
    ax.loglog(voxels, error, "o-", color="#e8663c", ms=4, lw=1.3,
              label="voxel sum (six primitive bonds)")
    ax.loglog(voxels, error[-1] * (voxels / voxels[-1]) ** -2.0,
              color="0.4", lw=0.9, ls="--", label=r"$\propto N^{-2}$")
    k = int(np.flatnonzero(voxels == 4.0)[0])
    ax.plot([voxels[k]], [error[k]], "o", ms=9, mfc="none", mec="black",
            mew=1.0)
    ax.annotate(
        f"N = 4 (previous model):\n{100.0 * (1.0 - error[k]):.0f} % of "
        "the exact value",
        xy=(voxels[k], error[k]), xytext=(10, 2), textcoords="offset points",
        fontsize=6.5, va="bottom",
    )
    ax.axhline(0.01, color="0.6", lw=0.7, ls=":")
    ax.text(2.1, 0.0112, "1 %", fontsize=6.5, color="0.4")
    ax.set_xlim(1.8, 120.0)
    ax.set_ylim(8e-4, 1.3)
    ax.set_xlabel("voxels per cube edge, N")
    ax.set_ylabel(r"relative error of $E_{\mathrm{vdW}}$, reference state")
    ax.set_title(title("c"))
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25, which="both", lw=0.4)


def panel_alpha(ax, f_original, verification):
    before = read_csv("F_original_panelB_alpha_scan.csv")
    after = read_csv("R3_free_gap_alpha_scan.csv")
    alpha_b, energy_b = column(before, "alpha_deg"), column(before, "Interaction_kBT")
    alpha_a, energy_a = column(after, "alpha_deg"), column(after, "Interaction_kBT")
    ax.plot(alpha_b, energy_b - f_original["interaction"], color="#c0392b",
            lw=1.4, ls="--", label="attraction only, gap fixed at 3.378 nm")
    ax.plot(alpha_a, energy_a - verification["components_kBT"]["Interaction"],
            color="black", lw=1.8, label="with the fitted repulsion, gap free")
    ax.plot([f_original["alpha"]], [0.0], "o", color="#c0392b", ms=5)
    ax.plot([verification["alpha_star_deg"]], [0.0], "*", color="#f1c40f",
            ms=11, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.annotate(rf"{f_original['alpha']:.1f}$^\circ$",
                xy=(f_original["alpha"], 0.0), xytext=(4, 8),
                textcoords="offset points", color="#c0392b", fontsize=7)
    ax.annotate(rf"{verification['alpha_star_deg']:.1f}$^\circ$",
                xy=(verification["alpha_star_deg"], 0.0), xytext=(-26, 8),
                textcoords="offset points", fontsize=7)
    ax.axvline(ALPHA_EXP, color="0.45", lw=0.8, ls=":",
               label=r"$\alpha_{\mathrm{exp}}$")
    ax.set_ylim(-0.15, 3.0)
    ax.set_xlim(64.0, 90.0)
    ax.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax.set_ylabel(r"$E - E_{\min}$ ($k_BT$)")
    ax.set_title(title("d"))
    ax.grid(alpha=0.25, lw=0.4)
    twin = ax.twinx()
    twin.plot(alpha_a, column(after, "gap_opt_nm"), color="#4d7fd1", lw=1.0,
              ls="-.", label="gap D* (fitted), right axis")
    twin.set_ylim(2.6, 4.2)
    twin.set_ylabel("relaxed gap D* (nm)", color="#4d7fd1")
    twin.tick_params(axis="y", colors="#4d7fd1")
    lines = ax.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    ax.legend(lines[0] + extra[0], lines[1] + extra[1], frameon=False,
              loc="upper left", fontsize=6)


def panel_valley(ax, verification):
    stencil = read_csv("R2_valley_stencil.csv")
    fine = read_csv("R3_valley_fine_solver.csv")
    alpha_s, energy_s = column(stencil, "alpha_deg"), column(stencil, "Interaction_kBT")
    alpha_f, energy_f = column(fine, "alpha_deg"), column(fine, "Interaction_kBT")
    floor = min(energy_s.min(), energy_f.min())
    scale = 1e3
    ax.plot(alpha_s, (energy_s - floor) * scale, "o", color="#4d7fd1", ms=5,
            label=VALLEY_LABELS["stencil"][SI])
    ax.plot(alpha_f, (energy_f - floor) * scale, "s", color="black", ms=3.5,
            mfc="none", label=VALLEY_LABELS["fine"][SI])
    for alpha, energy, colour, name in (
        (alpha_s, energy_s, "#4d7fd1", "stencil"),
        (alpha_f, energy_f, "black", "fine"),
    ):
        coefficients = np.polyfit(alpha, energy, 2)
        grid = np.linspace(alpha.min(), alpha.max(), 200)
        vertex = -coefficients[1] / (2.0 * coefficients[0])
        ax.plot(grid, (np.polyval(coefficients, grid) - floor) * scale,
                color=colour, lw=1.0, ls="--" if name == "fine" else "-")
        ax.axvline(vertex, color=colour, lw=0.7, ls=":")
        ax.text(vertex, 11.0 if name == "stencil" else 9.9,
                f" vertex {vertex:.2f}$^\\circ$", color=colour, fontsize=6.5)
    ax.axvline(ALPHA_EXP, color="#5fd35f", lw=1.2, alpha=0.8,
               label=r"$\alpha_{\mathrm{exp}}$ = 74.0$^\circ$")
    ax.annotate("branch step\n(+3.5e-3 kBT)", xy=(73.9, 3.6),
                xytext=(72.9, 6.4), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_ylim(-0.6, 12.2)
    ax.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax.set_ylabel(r"$E - E_{\min}$ ($10^{-3}\,k_BT$)")
    ax.set_title(title("e"))
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    ax.grid(alpha=0.25, lw=0.4)


def panel_transfer(fig, cell, verification):
    transfer = read_csv("R4_size_transfer.csv")
    adg = read_csv("B6_size_series_prediction.csv")
    # R5 evaluated the model at alpha = 60 deg for 19.5 and 29.6 nm; with the
    # measured alpha of 72-76 deg that is the cost of an fcc-like lattice.
    fcc_cost = {}
    for row in read_csv("R5_experimental_angle_cost.csv"):
        if float(row["alpha_experiment_deg"]) == 60.0:
            fcc_cost.setdefault(float(row["L_nm"]), []).append(
                float(row["cost_kBT"]))
    sizes = np.concatenate([[16.0], column(transfer, "L_nm")])
    alpha = np.concatenate([[verification["alpha_star_deg"]],
                            column(transfer, "alpha_star_deg")])
    a_star = np.concatenate([[verification["a_star_nm"]],
                             column(transfer, "a_star_nm")])
    order = np.argsort(sizes)
    sizes, alpha, a_star = sizes[order], alpha[order], a_star[order]
    inner = cell.subgridspec(2, 1, hspace=0.08)
    ax_top = fig.add_subplot(inner[0])
    ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)
    measured_colour = "#5fd35f"
    ax_top.fill_between(MEASURED_RANGE_SIZES_NM, *MEASURED_ALPHA_RANGE_DEG,
                        color=measured_colour, alpha=0.30, lw=0,
                        label="measured 16-30 nm (SI S14)")
    ax_top.plot(sizes, alpha, "o-", color="black", ms=4, lw=1.2,
                label="fitted repulsion (frozen)")
    ax_top.plot(column(adg, "L_nm"), column(adg, "alpha_star_deg"), "s--",
                color="0.55", ms=3.5, lw=0.9, mfc="none",
                label="AdG brush" if SI else "AdG brush (B6)")
    ax_top.plot([16.0], [MEASURED_16NM["alpha"]], "^", color=measured_colour,
                ms=7, markeredgecolor="black", markeredgewidth=0.5,
                label="measured 16 nm (Table S1)")
    ax_top.plot([13.0], [MEASURED_13NM_ALPHA_DEG], "^", ms=7, mfc="none",
                mec="#2e8b57", mew=1.0)
    ax_top.annotate("13 nm: sc,\nmoments unblocked",
                    xy=(13.0, MEASURED_13NM_ALPHA_DEG), xytext=(7, -2),
                    textcoords="offset points", fontsize=6, va="top",
                    color="#2e8b57")
    for size, values in fcc_cost.items():
        ax_top.plot([size], [60.0], "x", color="#c0392b", ms=6)
        right_edge = size > 25.0
        ax_top.annotate(
            f"+{float(np.mean(values)):.1f} kBT", xy=(size, 60.0),
            xytext=(-7 if right_edge else 7, 0), textcoords="offset points",
            fontsize=6, ha="right" if right_edge else "left", va="center",
            color="#c0392b",
        )
    ax_top.plot([], [], "x", color="#c0392b", ms=6,
                label=r"$\alpha$ = 60$^\circ$ (fcc-like), model cost")
    ax_top.set_ylim(55.0, 95.0)
    ax_top.set_ylabel(r"$\alpha^*$ ($^\circ$)")
    ax_top.set_title(title("f"))
    ax_top.grid(alpha=0.25, lw=0.4)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_bottom.plot(sizes, a_star, "o-", color="black", ms=4, lw=1.2)
    ax_bottom.plot(column(adg, "L_nm"), column(adg, "a_star_nm"), "s--",
                   color="0.55", ms=3.5, lw=0.9, mfc="none")
    ax_bottom.plot([16.0], [MEASURED_16NM["d"]], "^", color=measured_colour,
                   ms=7, markeredgecolor="black", markeredgewidth=0.5)
    ax_bottom.set_xlabel("particle edge L (nm)")
    ax_bottom.set_ylabel(r"$a^*$ (nm)")
    ax_bottom.grid(alpha=0.25, lw=0.4)
    # The alpha panel is too crowded for a legend; the a panel's upper left
    # is empty because a* grows with L.
    handles, labels = ax_top.get_legend_handles_labels()
    ax_bottom.legend(handles, labels, frameon=False, loc="upper left",
                     fontsize=5.5)


def main():
    global SI
    parser = argparse.ArgumentParser()
    # SI version: no super-title and no internal series names in the panels.
    parser.add_argument("--si", action="store_true")
    SI = int(parser.parse_args().si)
    report = json.loads((OUTPUT_DIR / "R_report.json").read_text("utf-8"))
    fitted = report["fitted"]
    verification = report["verification"]
    f_original = json.loads(
        (OUTPUT_DIR / "F_report.json").read_text("utf-8")
    )["original"]
    calibration = {
        (row["vdw_rule"], row["brush_model"]): row
        for row in read_csv("B3_brush_height_calibration.csv")
    }
    adg_height = float(
        calibration[("exact", "derjaguin")]["calibrated_brush_height_nm"]
    )
    adg_gap = float(
        next(row for row in read_csv("B6_size_series_prediction.csv")
             if float(row["L_nm"]) == 16.0)["gap_star_nm"]
    )
    probe_header = (OUTPUT_DIR / "R0_repulsion_family_probe.log").read_text(
        "utf-8").splitlines()[1]
    attraction_slope = float(
        re.search(r"dE/dalpha = ([-+0-9.e]+)", probe_header).group(1)
    )

    fig = plt.figure(figsize=(11.0, 6.9))
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.42)
    panel_laws(fig.add_subplot(grid[0, 0]), fitted, adg_height, adg_gap)
    panel_probe(fig.add_subplot(grid[0, 1]), fitted, attraction_slope)
    panel_vdw_convergence(fig.add_subplot(grid[0, 2]))
    panel_alpha(fig.add_subplot(grid[1, 0]), f_original, verification)
    panel_valley(fig.add_subplot(grid[1, 1]), verification)
    panel_transfer(fig, grid[1, 2], verification)
    if not SI:
        fig.suptitle(
            "Converged van der Waals attraction and a soft ligand repulsion "
            "fitted so that the model with Ms = 285 kA/m reproduces the "
            "16 nm superlattice\n"
            rf"$P(D) = P_{{\mathrm{{ref}}}}\,e^{{-(D - D_{{\mathrm{{ref}}}})"
            rf"/\lambda}}$, "
            f"P_ref = {fitted['P_ref_Pa'] / 1e3:.1f} kPa at "
            f"{fitted['reference_gap_nm']} nm, "
            rf"$\lambda$ = {fitted['lambda_nm']:.3f} nm",
            fontsize=9.5,
        )
    for suffix in ("png", "pdf"):
        name = "R_repulsion_fit_summary" + ("_SI" if SI else "")
        path = OUTPUT_DIR / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
