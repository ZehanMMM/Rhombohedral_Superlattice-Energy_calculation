"""R0: which repulsion families can move the angle at all.

Before fitting, each candidate law is put on the experimental state
(a, alpha) = (20.31 nm, 74.0 deg) with the attraction's own relaxed twist, its
amplitude is solved from the force balance dE/dD = 0, and the remaining angle
derivative dE/dalpha is reported.  The attraction alone gives
dE/dalpha = -0.067 kBT/deg (it pulls towards larger alpha); a law can fix the
angle only if, at force balance, this derivative crosses zero somewhere in its
shape parameter.

    exponential law on Derjaguin areas    shape = decay length lambda (nm)
    exponential law on facing facets only shape = decay length lambda (nm)
    Alexander-de Gennes brush, Derjaguin  shape = grafting spacing s (nm),
                                          amplitude = brush height L0 (nm)

The derivatives are taken at fixed gap in (alpha, D) coordinates with gamma
re-minimised at every node (v620_repulsion_fit.stationarity).  This is a
screening step only; the fit itself (run_repulsion_fit.py) uses the more
robust valley-vertex condition.

    python run_repulsion_family_probe.py        # ~20 min
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scipy.optimize import brentq

import v620_brush_vdw as bv
import v620_repulsion_fit as rf

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "brush_vdw"
)
BASE = dict(vdw_scheme="continuum", vdw_diagnostics=False, Ms=285_000.0)
DECAY_LENGTHS_NM = (0.05, 0.15, 0.3, 0.6, 1.2)
GRAFTING_SPACINGS_NM = (0.4, 0.63, 1.0, 1.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUT_DIR / "R0_repulsion_family_probe.log",
    )
    args = parser.parse_args()
    lines = []

    def say(text):
        print(text, flush=True)
        lines.append(text)

    attraction = bv.BrushRhombohedralSL(
        16.0, bv.BrushPhys(brush_model="off", **BASE)
    )
    target_gap = rf.target_gap_nm(attraction)
    point = rf.stationarity(attraction, rf.TARGET_ALPHA_DEG, target_gap)
    say("target gap D_t = %.6f nm (gamma* = %.3f, a = %.4f)"
        % (target_gap, point.gamma_deg, point.a_nm))
    say("attraction only: dE/dD = %+.4f kBT/nm, dE/dalpha = %+.5f kBT/deg"
        % (point.dE_dD, point.dE_dalpha))

    def family(label, make, bracket, shapes):
        say("--- %s" % label)
        for shape in shapes:
            started = time.perf_counter()

            def force(amplitude):
                return rf.stationarity(
                    make(amplitude, shape), rf.TARGET_ALPHA_DEG, target_gap
                ).dE_dD

            low, high = bracket(shape)
            try:
                amplitude = brentq(force, low, high, xtol=1e-4 * low,
                                   rtol=1e-6)
            except ValueError as error:
                say("   shape=%g: no force balance in bracket (%s)"
                    % (shape, error))
                continue
            state = rf.stationarity(
                make(amplitude, shape), rf.TARGET_ALPHA_DEG, target_gap
            )
            say("   shape=%-6g amp=%.5g  dE/dD=%+.2e  dE/dalpha=%+.5f  "
                "gamma*=%.3f  Steric=%.4f kBT  [%.0fs]"
                % (shape, amplitude, state.dE_dD, state.dE_dalpha,
                   state.gamma_deg, state.components["Steric"],
                   time.perf_counter() - started))

    def exponential(area_rule):
        def make(pressure_Pa, decay_nm):
            return bv.BrushRhombohedralSL(16.0, bv.BrushPhys(
                brush_model=area_rule, brush_law="exponential",
                steric_pressure_Pa=pressure_Pa, steric_decay_nm=decay_nm,
                steric_reference_gap_nm=target_gap, **BASE,
            ))
        return make

    def adg(height_nm, spacing_nm):
        return bv.BrushRhombohedralSL(16.0, bv.BrushPhys(
            brush_model="derjaguin", brush_law="adg",
            brush_height_nm=height_nm, grafting_spacing_nm=spacing_nm, **BASE,
        ))

    family("exponential, Derjaguin area rule (shape = decay length nm; "
           "amp = P at D_t, Pa)", exponential("derjaguin"),
           lambda decay: (1e2, 1e7), DECAY_LENGTHS_NM)
    family("exponential, facet-only area rule", exponential("facet"),
           lambda decay: (1e2, 1e7), DECAY_LENGTHS_NM)
    family("AdG, Derjaguin (shape = grafting spacing nm; amp = brush "
           "height L0 nm)", adg,
           lambda spacing: (0.5 * target_gap + 1e-3,
                            0.5 * target_gap + 0.6),
           GRAFTING_SPACINGS_NM)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
