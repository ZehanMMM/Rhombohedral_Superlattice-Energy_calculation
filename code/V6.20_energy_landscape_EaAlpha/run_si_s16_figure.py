"""SI Figure S16: the final S13 model only (Ms = 285 kA/m, fitted soft steric).

(A) Dipolar, vdW, steric and total interaction energy per particle against the
    nearest surface separation D at alpha = 74.0 deg (R3_energy_versus_gap.csv).
(B) vdW energy per particle against alpha: in the configuration that minimises
    E_int at each alpha (the Figure 4B curve, R_alpha_profile.csv), and on its
    own at a fixed nearest separation of 3.3 nm with untwisted cubes
    (gamma = 0), computed here and written to S16_vdw_fixed_gap.csv.

Usage:
    python run_si_s16_figure.py
        [--data-dir ../../outputs/V6.20_energy_landscape_EaAlpha/brush_vdw]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import v620_repulsion_fit as rf
from run_repulsion_figure import JACS_RC, JACS_WIDTH_IN

DEFAULT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "V6.20_energy_landscape_EaAlpha" / "brush_vdw"
)
FIXED_GAP_NM = 3.3
COLOURS = {"Dipole": "#6a5acd", "VdW": "#e8663c", "Steric": "#2e8b57"}


def read(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def fixed_gap_scan(model, alphas, gap_nm):
    rows = []
    for alpha in alphas:
        a_nm = model.a_for_exact_gap(alpha, 0.0, gap_nm)
        c = model.energy_components(a_nm, alpha, 0.0)
        rows.append((alpha, a_nm, rf.vdw_sharp_gap_nm(model, a_nm, alpha, 0.0),
                     c["VdW"]))
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    out = args.data_dir.resolve()

    report = json.loads((out / "R_report.json").read_text("utf-8"))
    fitted = dict(report["fitted"])
    fitted.setdefault("Ms_Apm", report["model"].get("Ms_Apm", rf.MS_SAMPLE_APM))
    model = rf.steric_model(fitted, L_nm=16.0)

    gap = read(out / "R3_energy_versus_gap.csv")
    profile = read(out / "R_alpha_profile.csv")
    alphas = np.arange(60.0, 90.0 + 1e-9, 1.0)
    scan = fixed_gap_scan(model, alphas, FIXED_GAP_NM)
    with (out / "S16_vdw_fixed_gap.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["alpha_deg", "a_nm", "sharp_gap_nm", "VdW6_kBT"])
        writer.writerows([[f"{v:.6f}" for v in row] for row in scan])
    print(f"fixed-gap scan: min sharp-cube gap {scan[:, 2].min():.2f} nm; "
          f"vdW {scan[0, 3]:.2f} (60 deg) -> {scan[-1, 3]:.2f} kBT (90 deg)")

    with mpl.rc_context(JACS_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(JACS_WIDTH_IN, 2.6))
        fig.subplots_adjust(wspace=0.32)

        d = gap["gap_nm"]
        for key, label in (("Dipole_kBT", "Dipole"), ("VdW6_kBT", "VdW"),
                           ("Steric_kBT", "Steric")):
            ax_a.plot(d, gap[key], color=COLOURS[label], lw=1.1, label=label)
        ax_a.plot(d, gap["Interaction_kBT"], color="black", lw=1.6,
                  label="Dipole + VdW + Steric")
        ax_a.axvline(fitted["gap_at_target_nm"], color="0.45", lw=1.0, ls=":",
                     label=r"$D_{\mathrm{eq}}$")
        ax_a.axhline(0.0, color="0.5", lw=0.7)
        ax_a.set_xlim(2.0, 7.0)
        ax_a.set_ylim(-8.0, 6.0)
        ax_a.set_xlabel(r"$D$ (nm)")
        ax_a.set_ylabel(r"Energy per particle ($k_BT$)")
        ax_a.legend(loc="upper right", frameon=True, facecolor="white",
                    edgecolor="none", framealpha=1.0, ncol=2)

        ax_b.plot(profile["alpha_deg"], profile["VdW6_kBT"], color=COLOURS["VdW"],
                  lw=1.1, label=r"$a$, $\gamma$ relaxed")
        ax_b.plot(scan[:, 0], scan[:, 3], color=COLOURS["VdW"], lw=1.1, ls="--",
                  label=r"$D$ = 3.3 nm, $\gamma$ = 0°")
        ax_b.set_xlim(60.0, 90.0)
        ax_b.set_xlabel(r"$\alpha$ (°)")
        ax_b.set_ylabel(r"VdW energy per particle ($k_BT$)")
        ax_b.legend(loc="upper right", frameon=True, facecolor="white",
                    edgecolor="none", framealpha=1.0)

        for ax, letter in ((ax_a, "A"), (ax_b, "B")):
            ax.grid(alpha=0.25)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _: f"{v:g}".replace("-", "−")))
            ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=10,
                    fontweight="bold")

        stem = out / "R_SI_S16_final"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
        fig.savefig(stem.with_suffix(".tif"), bbox_inches="tight", dpi=600,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    from PIL import Image
    with Image.open(stem.with_suffix(".tif")) as image:
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.getchannel("A"))
    flat.save(stem.with_suffix(".tif"), compression="tiff_lzw", dpi=(600, 600))
    print(f"wrote {stem.name}.pdf/.png/.tif")


if __name__ == "__main__":
    main()
