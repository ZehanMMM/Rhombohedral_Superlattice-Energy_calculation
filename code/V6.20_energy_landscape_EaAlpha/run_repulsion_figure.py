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
    python run_repulsion_figure.py --workers 12 # the same on 12 processes
    python run_repulsion_figure.py --coarse     # 16 x 16, ~15 min
    python run_repulsion_figure.py --fine --workers 14
                                        # 61 x 59 grid
    python run_repulsion_figure.py --size 18    # transfer to L = 18 nm
    python run_repulsion_figure.py --replot     # redraw from the saved data
    python run_repulsion_figure.py --replot --manuscript --panels
                                        # panels A, B, C as separate files
    python run_repulsion_figure.py --replot --jacs [--panels]
                                        # printed JACS size: PDF + 600 dpi TIFF
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
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Text3D
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

# Drawing sizes.  DEFAULT is the on-screen figure.  JACS is the figure at its
# printed size: JACS Appendix 2 asks for double-column graphics of at most
# 504 pt (7 in) width, lettering of at least 4.5 pt, lines of at least 0.5 pt,
# Helvetica or Arial, and information that does not rely on colour alone.
DEFAULT_STYLE = {
    "curve_lw": 1.8, "total_lw": 2.4, "star_ms": 14, "star_ms_c": 16,
    "exp_ms": 10, "guide_lw": 1.6, "contour_lw": 0.3, "annot_fs": 8,
    "annot_offset": (14, 22), "legend_fs": 8,
    "guide_styles": ("--", "--", "--"), "min_lw": 0.0, "text_fs_a": None,
    "zoom_a": None, "integer_ticks": False,
}
JACS_STYLE = {
    "curve_lw": 1.1, "total_lw": 1.6, "star_ms": 9, "star_ms_c": 10,
    "exp_ms": 6.5, "guide_lw": 1.0, "contour_lw": 0.5, "annot_fs": 6.5,
    # Right of the minimum and below the total curve, clear of the others.
    "annot_offset": (38, -1), "legend_fs": 6.5,
    "guide_styles": ("-.", "--", ":"), "min_lw": 0.55, "text_fs_a": 7.0,
    "zoom_a": 1.85, "integer_ticks": True,
}
JACS_WIDTH_IN = 7.0
JACS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
}


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


# Parallel landscape: every alpha row (and every profile point) is
# independent, so they are spread over worker processes.  Each worker builds
# the model once from the fitted parameters; the results are the same as the
# serial loop, point by point.
_WORKER_CACHE = {}


def _worker_model(fitted, size):
    key = (json.dumps(fitted, sort_keys=True), float(size))
    if _WORKER_CACHE.get("key") != key:
        _WORKER_CACHE["key"] = key
        _WORKER_CACHE["model"] = rf.steric_model(fitted, L_nm=size)
    return _WORKER_CACHE["model"]


def _landscape_row(task):
    fitted, size, alpha, lattice = task
    model = _worker_model(fitted, size)
    total = np.full(len(lattice), np.nan)
    interaction = np.full(len(lattice), np.nan)
    for j, a_nm in enumerate(lattice):
        _, components = relaxed(model, a_nm, alpha)
        if components is not None:
            total[j] = components["Total"]
            interaction[j] = components["Interaction"]
    return total, interaction


def _profile_point(task):
    fitted, size, alpha, lattice, row = task
    model = _worker_model(fitted, size)
    rows = alpha_profile(model, np.array([alpha]), lattice, row[None, :])
    return rows[0] if len(rows) else None


def parallel_landscape(fitted, size, alphas, lattice, workers):
    from concurrent.futures import ProcessPoolExecutor

    started = time.perf_counter()
    tasks = [(fitted, size, float(alpha), lattice) for alpha in alphas]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_landscape_row, tasks))
    print(f"      {len(alphas)} rows on {workers} workers "
          f"({time.perf_counter() - started:.0f} s)", flush=True)
    total = np.array([row[0] for row in rows])
    interaction = np.array([row[1] for row in rows])
    return total, interaction


def parallel_profile(fitted, size, alphas, lattice, interaction, workers):
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(fitted, size, float(alpha), lattice, interaction[i])
             for i, alpha in enumerate(alphas)
             if np.any(np.isfinite(interaction[i]))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for row in pool.map(_profile_point, tasks)
                if row is not None]
    return np.array(rows)


def draw_b(ax_b, profile, star, alphas, experiment, alpha_range,
           manuscript=False, style=DEFAULT_STYLE):
    """Panel B: energy components against alpha, gap and twist relaxed.

    The manuscript version drops the explanatory title suffix, the "(fitted)"
    tag and the in-panel annotations; the caption carries that information.
    """
    ax_b.axhline(0.0, color="0.5", lw=0.7)
    ax_b.plot(profile[:, 0], profile[:, 4], color="#6a5acd",
              lw=style["curve_lw"], label="Dipole")
    # All four curves are solid; colour tells them apart.
    ax_b.plot(profile[:, 0], profile[:, 5], color="#e8663c",
              lw=style["curve_lw"], label="VdW")
    ax_b.plot(profile[:, 0], profile[:, 6], color="#2e8b57",
              lw=style["curve_lw"],
              label="Steric" if manuscript else "Steric (fitted)")
    ax_b.plot(profile[:, 0], profile[:, 7], color="black",
              lw=style["total_lw"], label="Dipole + VdW + Steric")
    if experiment is not None:
        ax_b.axvline(experiment["alpha"], color="0.45", lw=1.0, ls=":",
                     label=r"$\alpha_{\mathrm{exp}}$")
    if experiment is not None and not manuscript:
        # The 60 deg experiments sit on the axis edge, so state the cost.
        cost = float(np.interp(experiment["alpha"], profile[:, 0],
                               profile[:, 7])) - star["interaction"]
        ax_b.text(
            0.98, 0.04,
            r"$E(\alpha_{\mathrm{exp}}) - E$* = " + f"{cost:+.3f} "
            + r"$k_BT$",
            transform=ax_b.transAxes, ha="right", fontsize=8,
        )
    if alpha_range is not None:
        ax_b.axvspan(*alpha_range, color="#5fd35f", alpha=0.18, lw=0,
                     label=r"measured $\alpha$ range (SI S14)")
    if alpha_range is not None and not manuscript:
        inside = alpha_range[0] <= star["alpha"] <= alpha_range[1]
        ax_b.text(
            0.98, 0.04,
            r"$\alpha$* " + ("inside" if inside else "outside")
            + f" the measured {alpha_range[0]:.0f}"
            + f"-{alpha_range[1]:.0f}°",
            transform=ax_b.transAxes, ha="right", fontsize=8,
        )
    # Manuscript: the star is keyed in the legend as alpha_sim, next to
    # alpha_exp, instead of an arrow with the value.
    ax_b.plot([star["alpha"]], [star["interaction"]], marker="*",
              ms=style["star_ms"], color="#f1c40f", markeredgecolor="black",
              markeredgewidth=0.6, linestyle="none", zorder=5,
              label=r"$\alpha_{\mathrm{sim}}$" if manuscript else None)
    if not manuscript:
        ax_b.annotate(
            r"$\alpha$* = " + f"{star['alpha']:.2f}°",
            xy=(star["alpha"], star["interaction"]),
            xytext=style["annot_offset"], textcoords="offset points",
            va="center" if style["annot_offset"][1] < 5 else "baseline",
            fontsize=style["annot_fs"],
            arrowprops=dict(arrowstyle="->", lw=0.6),
        )
    ax_b.set_xlabel(r"$\alpha$ (°)")
    ax_b.set_ylabel(r"Energy per particle ($k_BT$)")
    ax_b.set_title(r"$E$ vs $\alpha$" if manuscript
                   else r"$E$ vs $\alpha$, gap relaxed at every $\alpha$")
    ax_b.set_xlim(alphas[0], alphas[-1])
    # Head room above the curves for the legend; energies grow with L.  The
    # manuscript legend has a third row (alpha_sim) and an opaque white box,
    # so the alpha_exp line does not run through the labels.
    curves = profile[:, 4:8]
    span = float(np.nanmax(curves) - np.nanmin(curves))
    headroom = 0.45 if manuscript else 0.32
    ax_b.set_ylim(top=float(np.nanmax(curves)) + headroom * span)
    ax_b.grid(alpha=0.25)
    if style["integer_ticks"]:
        ax_b.yaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=6, integer=True,
                                   steps=[1, 2, 4, 5, 10]))
    if manuscript:
        ax_b.legend(ncol=2, loc="upper left", frameon=True,
                    facecolor="white", edgecolor="none", framealpha=1.0)
    else:
        ax_b.legend(ncol=2, frameon=False, loc="upper left")


def draw_c(figure, ax_c, profile, star, alphas, lattice, total, size,
           experiment, alpha_range, manuscript=False, style=DEFAULT_STYLE):
    """Panel C: the free E(a, alpha) landscape, twist relaxed at every node.

    The manuscript version drops the title suffix, the colour-window note on
    the colour bar and the ", relaxed gap" legend tag.
    """
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
        colors="white", linewidths=style["contour_lw"], linestyles="dashed",
        alpha=0.55,
    )
    # The three guide lines differ in dash pattern as well as colour in the
    # JACS style, so they stay distinguishable without colour.
    valley_ls, a_ls, alpha_ls = style["guide_styles"]
    if manuscript:
        # Manuscript: the relaxed-a valley as a black dashed
        # line, and no a* / alpha* guide lines (the star and legend give them).
        ax_c.plot(profile[:, 1], profile[:, 0], color="black",
                  lw=style["guide_lw"], ls="--",
                  label=r"$a_{\mathrm{relaxed}}$ at $\alpha$")
    else:
        ax_c.plot(profile[:, 1], profile[:, 0], color="#4d7fd1",
                  lw=style["guide_lw"], ls=valley_ls,
                  label=r"$a$*($\alpha$), relaxed gap")
        ax_c.axvline(star["a"], color="#e8a33d", lw=style["guide_lw"],
                     ls=a_ls, label=r"$a$ = " + f"{star['a']:.2f} nm")
        ax_c.axhline(star["alpha"], color="#4fc3d9", lw=style["guide_lw"],
                     ls=alpha_ls,
                     label=r"$\alpha$ = " + f"{star['alpha']:.1f}°")
    ax_c.plot([star["a"]], [star["alpha"]], marker="*",
              ms=style["star_ms_c"], color="#f1c40f",
              markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
              label=f"Sim ({star['a']:.2f} nm, {star['alpha']:.1f}"
              + "°)")
    if experiment is not None:
        ax_c.plot([experiment["d"]], [experiment["alpha"]], marker="^",
                  ms=style["exp_ms"], color="#5fd35f",
                  markeredgecolor="black", markeredgewidth=0.6,
                  linestyle="none", zorder=6,
                  label=f"Exp ({experiment['d']:.2f} nm, "
                  f"{experiment['alpha']:.1f}°)")
    if alpha_range is not None:
        for edge in alpha_range:
            ax_c.axhline(edge, color="#2e8b57", lw=1.2, ls=":")
        ax_c.plot([], [], color="#2e8b57", lw=1.2, ls=":",
                  label=f"measured {alpha_range[0]:.0f}-"
                  f"{alpha_range[1]:.0f}° (SI S14)")
    ax_c.set_xlabel(r"$a$ (nm)")
    ax_c.set_ylabel(r"$\alpha$ (°)")
    if manuscript:
        ax_c.set_title(r"$E(a, \alpha)$ landscape")
    else:
        ax_c.set_title(
            r"$E(a, \alpha)$ landscape, gap free "
            + ("(fitted repulsion)" if size == FIT_SIZE_NM
               else "(repulsion fitted at 16 nm, frozen)")
        )
    ax_c.set_xlim(lattice[0], lattice[-1])
    ax_c.set_ylim(alphas[0], alphas[-1])
    bar = figure.colorbar(mesh, ax=ax_c, pad=0.02)
    # "Total" is the energy per particle: Zeeman + anisotropy of one cube plus
    # half of its dipolar, vdW and steric pair energies (E = -153.96 kBT at
    # the minimum), the same quantity as E in SI S13.
    bar.set_label(r"$E_{\mathrm{total}}$ per particle ($k_BT$)" if manuscript
                  else r"$E_{\mathrm{total}} / k_BT$ (clipped at min + "
                  + f"{window:.0f})")
    if style["integer_ticks"] or manuscript:
        # Even integers inside the plotted window (a locator can place a tick
        # on the extend arrow, where it overlaps the top label), printed with
        # a true minus sign.
        step = 2 if window <= 20 else (5 if window <= 50 else 10)
        ticks = np.arange(np.ceil(low / step) * step, high + 1e-9, step)
        bar.set_ticks(ticks)
        bar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda value, _: f"{value:.0f}".replace("-", "−")))
    # Inside the axes at the upper right, which is the high-energy corner of
    # every landscape (large a and large alpha), on an opaque white box.
    ax_c.legend(loc="upper right", frameon=True, facecolor="white",
                edgecolor="0.6", framealpha=1.0, fontsize=style["legend_fs"])


def drop_twist_label(ax_a):
    """Manuscript panel A: the caption line gives a and alpha only."""
    for text in ax_a.texts:
        label = text.get_text()
        cut = label.find(r", $\gamma$")
        if cut >= 0:
            text.set_text(label[:cut])


def jacs_panel_a(ax_a, style):
    """Bring the shared panel A drawing to the JACS minimum sizes.

    Also fills the panel better (zoom) and sets the labels in Arial glyphs:
    a true minus sign and a text degree sign instead of mathtext's ring.
    """
    if style["text_fs_a"] is not None:
        for text in ax_a.texts:
            text.set_fontsize(style["text_fs_a"])
            text.set_text(text.get_text().replace(r"$^\circ$", "°")
                          .replace("-", "−"))
    if style["zoom_a"] is not None:
        ax_a.set_box_aspect((1.15, 1.0, 1.0), zoom=style["zoom_a"])
        # The zoom moves the field label (beyond the arrow tip) into the
        # next panel; bring it in under the tip.
        for text in [t for t in ax_a.texts if isinstance(t, Text3D)]:
            x, y, z = text.get_position_3d()
            text.set_position_3d((0.78 * x, y, 1.9 * z))
    for collection in ax_a.collections:
        widths = np.atleast_1d(collection.get_linewidth())
        collection.set_linewidth(np.maximum(widths, style["min_lw"]))


def save(figure, stem, extra_artists=None, jacs=False):
    # JACS: vector PDF, a 600 dpi TIFF (above the 300 dpi minimum for colour
    # art) for upload, and a 300 dpi PNG for embedding in the manuscript.
    formats = ((("pdf", {}),
                ("tif", {"dpi": 600,
                         "pil_kwargs": {"compression": "tiff_lzw"}}),
                ("png", {"dpi": 300}))
               if jacs else (("png", {}), ("pdf", {})))
    for suffix, options in formats:
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight",
                       bbox_extra_artists=extra_artists, **options)
        if suffix == "tif":
            # Agg writes RGBA; production wants a flat RGB image.
            from PIL import Image

            with Image.open(path) as image:
                flat = Image.new("RGB", image.size, "white")
                flat.paste(image, mask=image.getchannel("A"))
            flat.save(path, compression="tiff_lzw", dpi=(600, 600))
        print(f"  wrote {path.name}")
    plt.close(figure)


def figure_inputs(fitted, size):
    model = rf.steric_model(fitted, L_nm=size)
    reference = measured(size)
    return model, reference.get("point"), reference.get("range")


def variant_stem(size, manuscript, jacs):
    stem = f"R_fitted_repulsion_figure{tag(size)}"
    if jacs:
        return stem + "_JACS"
    return stem + ("_manuscript" if manuscript else "")


def make_panels(fitted, star, profile, alphas, lattice, total,
                size=FIT_SIZE_NM, manuscript=False, jacs=False):
    """Panels A, B and C as three separate figures, without panel letters.

    In the JACS style each panel is drawn at the size it has in the
    double-column figure, so the panels assemble into a 7 in figure at 100 %.
    """
    model, experiment, alpha_range = figure_inputs(fitted, size)
    stem = variant_stem(size, manuscript, jacs)
    style = JACS_STYLE if jacs else DEFAULT_STYLE
    sizes = (((2.9, 2.5), (4.0, 2.5), (JACS_WIDTH_IN, 2.9)) if jacs
             else ((6.2, 4.4), (6.0, 4.4), (9.5, 5.0)))

    # The field label is a 3D text beyond the axes.  A 3D text reports its
    # projected position only while it is being drawn, so the tight bounding
    # box measures it in the wrong place and crops it.  Re-place it as a
    # figure text at its projected position, which the bounding box sees.
    figure = plt.figure(figsize=sizes[0])
    ax_a = figure.add_axes((0.0, 0.0, 0.78, 1.0), projection="3d")
    panel_a(ax_a, model, star["a"], star["alpha"], star["gamma"])
    if manuscript:
        drop_twist_label(ax_a)
    jacs_panel_a(ax_a, style)
    figure.canvas.draw()
    for text in [t for t in ax_a.texts if isinstance(t, Text3D)]:
        x, y, z = text.get_position_3d()
        px, py, _ = proj3d.proj_transform(x, y, z, ax_a.get_proj())
        fx, fy = figure.transFigure.inverted().transform(
            ax_a.transData.transform((px, py)))
        figure.text(fx, fy, text.get_text(), color=text.get_color(),
                    fontsize=text.get_fontsize(),
                    fontweight=text.get_fontweight())
        text.remove()
    save(figure, stem + "_A", jacs=jacs)

    figure, ax_b = plt.subplots(figsize=sizes[1])
    draw_b(ax_b, profile, star, alphas, experiment, alpha_range, manuscript,
           style)
    save(figure, stem + "_B", jacs=jacs)

    figure, ax_c = plt.subplots(figsize=sizes[2])
    draw_c(figure, ax_c, profile, star, alphas, lattice, total, size,
           experiment, alpha_range, manuscript, style)
    save(figure, stem + "_C", jacs=jacs)


def make_figure(fitted, star, profile, alphas, lattice, total,
                size=FIT_SIZE_NM, manuscript=False, jacs=False):
    model, experiment, alpha_range = figure_inputs(fitted, size)
    style = JACS_STYLE if jacs else DEFAULT_STYLE
    if jacs:
        # Double column, 7 in wide; with the caption it stays far below the
        # 9.167 in maximum depth.
        figure = plt.figure(figsize=(JACS_WIDTH_IN, 5.9))
        grid = figure.add_gridspec(
            2, 2, height_ratios=(1.0, 1.1), width_ratios=(1.0, 1.35),
            hspace=0.42, wspace=0.38,
        )
        letter_size, letter_x = 10, -0.10
    else:
        figure = plt.figure(figsize=(11.0, 8.4))
        grid = figure.add_gridspec(
            2, 2, height_ratios=(1.0, 1.15), width_ratios=(1.0, 1.25),
            hspace=0.28, wspace=0.30,
        )
        letter_size, letter_x = 14, -0.08

    ax_a = figure.add_subplot(grid[0, 0], projection="3d")
    panel_a(ax_a, model, star["a"], star["alpha"], star["gamma"])
    if manuscript:
        drop_twist_label(ax_a)
    jacs_panel_a(ax_a, style)
    ax_b = figure.add_subplot(grid[0, 1])
    draw_b(ax_b, profile, star, alphas, experiment, alpha_range, manuscript,
           style)
    ax_c = figure.add_subplot(grid[1, :])
    draw_c(figure, ax_c, profile, star, alphas, lattice, total, size,
           experiment, alpha_range, manuscript, style)

    for axis, letter in ((ax_b, "B"), (ax_c, "C")):
        axis.text(letter_x, 1.03, letter, transform=axis.transAxes,
                  fontsize=letter_size, fontweight="bold")
    ax_a.text2D(-0.02, 1.03, "A", transform=ax_a.transAxes,
                fontsize=letter_size, fontweight="bold")
    if not (manuscript or jacs):
        figure.suptitle(
            f"V6.20, L = {size:g} nm, converged van der Waals, "
            f"Ms = {fitted['Ms_Apm'] / 1e3:.0f} kA/m, "
            + ("with the fitted steric repulsion\n" if size == FIT_SIZE_NM
               else "steric repulsion fitted at 16 nm and frozen\n")
            + steric_label(fitted),
            fontsize=10,
        )
    save(figure, variant_stem(size, manuscript, jacs), jacs=jacs)


def steric_label(fitted):
    """One line naming the steric law and its parameters."""
    return (
        r"$P(D) = P_{\mathrm{ref}}\, e^{-(D - D_{\mathrm{ref}})/\lambda}$, "
        f"P_ref = {fitted['P_ref_Pa'] / 1e3:.1f} kPa at "
        f"{fitted['reference_gap_nm']} nm, "
        r"$\lambda$ = " + f"{fitted['lambda_nm']:.3f} nm"
    )


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
    # 0.5 deg x 0.1 nm grid.
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--size", type=float, default=FIT_SIZE_NM)
    # Without the diagnostic super-title, for the manuscript figure.
    parser.add_argument("--manuscript", action="store_true")
    # Panels A, B and C as three separate files instead of one figure.
    parser.add_argument("--panels", action="store_true")
    # Printed size for JACS (implies --manuscript): 7 in wide, Arial,
    # lettering and lines above the journal minima, PDF + 600 dpi TIFF.
    parser.add_argument("--jacs", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the landscape and profile")
    # Read and write another result set (default: brush_vdw).
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is not None:
        global OUTPUT_DIR
        OUTPUT_DIR = args.output_dir.resolve()
    size = float(args.size)
    suffix = tag(size)
    report = json.loads((OUTPUT_DIR / "R_report.json").read_text("utf-8"))
    fitted = dict(report["fitted"])
    # The report carries Ms in its model block.
    fitted.setdefault("Ms_Apm", report["model"].get("Ms_Apm",
                                                   rf.MS_SAMPLE_APM))
    builder = make_panels if args.panels else make_figure
    manuscript = args.manuscript or args.jacs

    def draw(*positional):
        with mpl.rc_context(JACS_RC if args.jacs else {}):
            builder(*positional, manuscript=manuscript, jacs=args.jacs)

    if args.replot:
        alphas, lattice, total, profile, star = load_saved(size)
        draw(fitted, star, profile, alphas, lattice, total, size)
        return
    model = rf.steric_model(fitted, L_nm=size)
    alpha_points, lattice_points = (
        (16, 16) if args.coarse else (61, 59) if args.fine else (31, 30)
    )
    alphas = np.linspace(60.0, 90.0, alpha_points)
    lattice = lattice_window(size, lattice_points)
    print(f"L = {size:g} nm, lattice window {lattice[0]:.2f}-"
          f"{lattice[-1]:.2f} nm")

    print("panel C landscape")
    if args.workers > 1:
        total, interaction = parallel_landscape(fitted, size, alphas,
                                                lattice, args.workers)
    else:
        total, interaction = landscape(model, alphas, lattice)
    print("panel B profile")
    if args.workers > 1:
        profile = parallel_profile(fitted, size, alphas, lattice,
                                   interaction, args.workers)
    else:
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

    draw(fitted, star, profile, alphas, lattice, total, size)


if __name__ == "__main__":
    main()
