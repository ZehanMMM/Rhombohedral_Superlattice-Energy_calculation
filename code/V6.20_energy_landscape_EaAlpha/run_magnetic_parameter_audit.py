"""Which magnetic parameters does V6.20 use, and which of them matter?

V6.20 carries four magnetic inputs: the saturation magnetisation ``Ms``, the
applied field ``B``, the cubic anisotropy constant ``K_ani`` and the
temperature.  Three of the four turn out to be structurally inert:

* ``B`` enters only as ``-Ms V B``, which is independent of (a, alpha, gamma);
* ``K_ani`` enters through the moment direction in the particle frame, and that
  direction is fixed: the field lies along the cube body diagonal, the moment
  follows the field, and gamma is a rotation *about that same axis*, so the
  moment sits permanently on the [111] easy axis and
  ``E_ani = -K V / 3`` exactly, for every structure;
* the temperature only sets the kBT unit; the minimisation is at zero
  temperature.

So ``Ms`` is the only magnetic parameter that can move the fitted lattice, and
it does so through ``E_dd ~ Ms^2``.  This script writes out the audit and scans
alpha* against ``Ms``, because the angle selection is a competition between the
dipolar and van der Waals terms whose margin is a fraction of a kBT.

E1  The parameter table, with derived scales and the self-consistency checks.
E2  alpha*, gamma* and a* against Ms, at the converged attraction.
E3  The same against the ratio that actually controls the selection, Ms^2 / A.

Outputs go to ``outputs/V6.20_energy_landscape_EaAlpha/brush_vdw``.

    python run_magnetic_parameter_audit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, mu_0, physical_constants

import v620_brush_vdw as bv
from v620_brush_fitting import fixed_gap_alpha_scan

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
BOHR_MAGNETON = physical_constants["Bohr magneton"][0]

# The comparison values are the repository's own sourced table,
# versions/V11_literature_constrained/PARAMETER_SOURCES.md, which cites
# Moreno et al., Sci. Rep. 10, 2722 (2020), DOI 10.1038/s41598-020-58976-7
# for both Ms and K1 of highly crystalline magnetite at room temperature.
LITERATURE = {
    "Ms_Apm": 460_000.0,
    "K1_Jpm3": -11_500.0,
    "source": "V11 PARAMETER_SOURCES.md / Moreno et al. 2020",
}
MS_SCAN_Apm = (285_000.0, 320_000.0, 360_000.0, 400_000.0, 460_000.0,
               480_000.0)


def write_csv(name, header, rows):
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return path


def section_e1():
    print("E1  the magnetic parameter table")
    model = bv.BrushRhombohedralSL(REF["L_nm"], bv.BrushPhys(
        brush_model="off"
    ))
    volume = model.volume
    kT = model.kT

    direction = model.orientation(REF["gamma_deg"]).T @ model.mu_hat
    body_diagonal = float(
        abs(direction @ np.ones(3) / np.sqrt(3.0))
    )
    print(
        f"    moment direction in the particle frame = "
        f"{np.round(direction, 6)}, |cos to [111]| = {body_diagonal:.9f}"
    )
    if abs(body_diagonal - 1.0) > 1e-9:
        raise RuntimeError("The moment is not on the cube body diagonal.")

    rows = []
    for label, ms in (
        ("V6.20 (55 A m2/kg x 5.18e3 kg/m3)", model.Ms),
        ("literature (Moreno et al. 2020)", LITERATURE["Ms_Apm"]),
        ("bulk magnetite at 300 K, common value", 480_000.0),
    ):
        moment = ms * volume
        rows.append(
            [
                "Ms",
                label,
                f"{ms:.6g}",
                "A/m",
                f"{ms / model.Ms:.4f}",
                f"moment {moment:.4e} A m2 = {moment / BOHR_MAGNETON:.4e} muB;"
                f" Zeeman at 0.5 T {-moment * model.B_field / kT:.1f} kBT;"
                f" mu0 m^2 / 4 pi a^3 at a = 20.31 nm ="
                f" {(mu_0 / (4 * np.pi)) * moment**2 / (20.31e-9) ** 3 / kT:.3f} kBT",
            ]
        )
    for label, k in (
        ("V6.20 notebook K_ani", model.K_ani),
        ("V6.20 v620_two_nc_temperature.py (12 x ZFC/FC barrier)", 2.563299e5),
        ("literature |K1| (Moreno et al. 2020)", abs(LITERATURE["K1_Jpm3"])),
    ):
        anisotropy_field = 2.0 * k / (mu_0 * model.Ms)
        rows.append(
            [
                "K_cubic",
                label,
                f"{k:.6g}",
                "J/m^3",
                f"{k / model.K_ani:.4f}",
                f"K V = {k * volume / kT:.2f} kBT;"
                f" <111> to <110> barrier K V / 12 ="
                f" {k * volume / 12 / kT:.3f} kBT;"
                f" mu0 H_K = {mu_0 * anisotropy_field:.3f} T;"
                f" Zeeman/(K V) = {model.mu_magnitude * model.B_field / (k * volume):.2f}",
            ]
        )
    rows.append(
        [
            "B",
            "V6.20 notebook (assembly field, experimental setting)",
            f"{model.B_field:.6g}",
            "T",
            "1.0000",
            "enters only as -Ms V B, a constant offset of "
            f"{-model.mu_magnitude * model.B_field / kT:.1f} kBT; "
            "no effect on alpha*, gamma* or a*",
        ]
    )
    rows.append(
        [
            "B",
            "V6.20 v620_two_nc_temperature.py",
            "0",
            "T",
            "0",
            "that module has no field at all; it is a zero-field pair model",
        ]
    )
    rows.append(
        [
            "T",
            "V6.20 notebook",
            f"{model.temperature:.6g}",
            "K",
            "1.0000",
            "sets the kBT unit only; the minimisation is at zero temperature",
        ]
    )
    rows.append(
        [
            "E_ani",
            "value implied by the fixed moment direction",
            f"{-model.K_ani * volume / 3.0 / kT:.6f}",
            "kBT",
            "",
            "E_ani = -K V / 3 for every (a, alpha, gamma): the moment sits on "
            "the [111] easy axis and gamma rotates about the moment itself",
        ]
    )
    write_csv(
        "E1_magnetic_parameters.csv",
        ["parameter", "variant", "value", "unit", "ratio_to_V620", "notes"],
        rows,
    )
    return rows


def section_e2():
    print("E2  alpha* against Ms, converged attraction, constant-gap ensemble")
    gap = bv.BrushRhombohedralSL(
        REF["L_nm"], bv.BrushPhys(brush_model="off")
    ).nearest_exact_gap(REF["a_nm"], REF["alpha_deg"], REF["gamma_deg"])
    rows = []
    for ms in MS_SCAN_Apm:
        for scheme, tag in (("continuum", "exact"), ("voxel", "N=4")):
            phys = bv.BrushPhys(
                brush_model="off",
                vdw_scheme=scheme,
                voxel_count=4,
                vdw_diagnostics=False,
                Ms=ms,
            )
            fit = fixed_gap_alpha_scan(
                bv.BrushRhombohedralSL(REF["L_nm"], phys), gap, ALPHAS, GAMMAS
            )
            star = fit["components_star"]
            rows.append(
                [
                    f"{ms:.6g}",
                    tag,
                    f"{fit['alpha_star']:.4f}",
                    f"{fit['gamma_star']:.4f}",
                    f"{fit['a_star']:.5f}",
                    f"{star['Dipole']:.5f}",
                    f"{star['VdW']:.5f}",
                    f"{star['Interaction']:.5f}",
                    f"{ms**2 / 2.0e-20 * 1e-20 / 1e6:.4f}",
                ]
            )
            print(
                f"    Ms={ms:7.0f} A/m  vdW={tag:5s}  "
                f"alpha*={fit['alpha_star']:7.3f}  "
                f"a*={fit['a_star']:8.4f}  "
                f"Dip={star['Dipole']:8.4f}  VdW={star['VdW']:8.4f} kBT",
                flush=True,
            )
    write_csv(
        "E2_alpha_versus_Ms.csv",
        [
            "Ms_Apm",
            "vdw_rule",
            "alpha_star_deg",
            "gamma_star_deg",
            "a_star_nm",
            "Dipole_kBT",
            "VdW6_kBT",
            "Interaction_kBT",
            "Ms_squared_over_hamaker_1e6",
        ],
        rows,
    )
    return rows


def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")
    section_e1()
    rows = section_e2()
    exact = {float(row[0]): float(row[2]) for row in rows if row[1] == "exact"}
    report = {
        "structurally_active_magnetic_parameters": ["Ms"],
        "structurally_inert": {
            "B": "constant Zeeman offset",
            "K_ani": "constant -K V / 3, moment locked on the [111] easy axis",
            "T": "kBT unit only",
        },
        "V620_Ms_Apm": 285_000.0,
        "literature": LITERATURE,
        "alpha_star_versus_Ms_exact_vdw_deg": exact,
        "experimental_alpha_deg": bv.EXP_DATA[16.0]["alpha"],
        "experimental_a_nm": bv.EXP_DATA[16.0]["d"],
    }
    (OUTPUT_DIR / "E_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  wrote E_report.json")


if __name__ == "__main__":
    main()
