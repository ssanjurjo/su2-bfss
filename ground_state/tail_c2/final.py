"""The tail coefficient |C|^2 with its uncertainty.

Anchor: the direct far-field fit of the best state Q4L22G16V25_r0 (10 x 10 transverse mesh with quartic elements in a and b,
elements of at most 0.25 along the valley, pruning tolerance 1e-16, L_c = 22, sixteen Gauss points per element across the
valley), with the corrections as a series in R^-3 (R^-3 alone and R^-3, R^-6) on the windows lo 7..10, hi 17..19.  Every other
refinement is measured in the same box as the difference of two direct fits: the transverse mesh 10 -> 12 and the pruning
tolerance are added as shifts; the elements along the valley (0.5, 0.35, 0.25) enter the uncertainty with the full change
between 0.5 and 0.25, which is the largest effect observed (a conservative choice).  The result is quoted with the
uncertainty rounded up to one digit.
Every entry of the uncertainty is a measured difference or is estimated from one, except a fixed allowance of 0.02 for the
quadrature of the profile on the slices; nothing is extrapolated in E_h.  The
prediction enters only the last lines.

    python c2_table.py --ref Q4L22G16V25_r0 --json table.json; python final.py table.json

Seconds; expected output data/tail_c2/final.txt (and data/tail_c2/table.json for the table)."""
import json, sys, numpy as np
from math import gamma, pi
import os
from c2_table import load, fit, paths, PRED          # the loader, the fit, the profiles and the prediction
T = json.load(open(sys.argv[1])); rows = {r["tag"]: r for r in T["rows"]}
d = lambda t: rows[t]["dN"]; E = lambda t: rows[t]["E"]
def series_fits(tag):
    R, u, Eh, Rw = load(paths[tag])
    return np.array([fit(R, u, Eh, lo, hi, ps)[0] for ps in ((3,), (3, 6)) for lo in (7, 8, 9, 10) for hi in (17, 18, 19)]), Eh
ANCHOR = "Q4L22G16V25_r0"                                 # the best state
fits, Eh = series_fits(ANCHOR); anchor = fits.mean(); anc_err = (fits.max() - fits.min()) / 2
print(f"anchor {ANCHOR} (E_h {Eh:.6f}): {anchor:.3f}  (R^-3 series fits {fits.min():.3f} .. {fits.max():.3f})")
entries = []
def add(label, val, err, src): entries.append((label, val, err, src)); print(f"  {label:48s} {val:+7.3f} +- {err:5.3f}   ({src})")
PRED = 893025 / (32 * pi ** 4)
def report():
    tot = anchor + sum(e[1] for e in entries); err = np.sqrt(sum(e[2] ** 2 for e in entries))
    print(f"\n|C|^2 = {tot:.3f} +- {err:.3f}   (relative {err / tot * 100:.2f}%)")
    print(f"prediction 893025/(32 pi^4) = {PRED:.3f}: difference {tot - PRED:+.3f} = {(tot - PRED) / PRED * 100:+.3f}% = {(tot - PRED) / err:+.2f} times the uncertainty")
    q = np.ceil(err * 10 - 1e-9) / 10
    print(f"quoted, with the uncertainty rounded up: |C|^2 = {tot:.1f} +- {q:.1f}; the prediction lies {abs(round(tot, 1) - PRED) / q:.1f} of it away")
def budget():                                 # every refinement is a pair of direct fits in the box L_c = 22
    from scipy.optimize import minimize_scalar
    ref, v35, n12, e0, e1, t24, g12 = "Q4L22G16_r0", "Q4L22G16V35_r0", "Q4N12L22G16_r0", "Q4L22G16E_r0", "Q4L22G16E_r1", "Q4L22G16T24_r0", "Q4L22G12_r0"
    F = {t: series_fits(t) for t in (ref, v35, n12, e0, e1, t24, g12)}; m = lambda t: F[t][0].mean(); Ef = lambda t: F[t][1]
    print("\nelements along the valley (direct fits): " + ", ".join(f"h {h}: E_h {e:.6f}, |C|^2 {c:.3f}" for h, e, c in ((0.5, Ef(ref), m(ref)), (0.35, Ef(v35), m(v35)), (0.25, Eh, anchor))))
    print("\nshifts (differences of direct far-field fits, L_c = 22):")
    last = anchor - m(v35); full = anchor - m(ref); rich = {q: last * 0.25 ** q / (0.35 ** q - 0.25 ** q) for q in (2, 3, 4)}
    add("elements along the valley (full change 0.5 -> 0.25)", 0.0, abs(full), f"0.5 -> 0.35: {m(v35) - m(ref):+.3f}, 0.35 -> 0.25: {last:+.3f}; expected remainder below 0.25: "
        + ", ".join(f"{v:+.3f} (h^{q})" for q, v in rich.items()))
    s = m(n12) - m(ref); add("transverse mesh 10 -> 12 (valley h 0.5)", s, max(abs(s), 0.005), f"dE_h {Ef(n12) - Ef(ref):+.6f}")
    slopes = {lab: (d(a) - d(b)) / (E(a) - E(b)) for lab, a, b in (("h 10->12 cubic", "N12P16B15_r1", "P16B15_r1"), ("p 3->4 n=10", "Q4P16B15_r0", "P16B15_r1"),
              ("h 10->12 quartic", "Q4N12P16B15_r0", "Q4P16B15_r0"), ("p 3->4 n=12", "Q4N12P16B15_r0", "N12P16B15_r1"))}
    Ev = (Ef(v35) - Eh) * 0.25 ** 2 / (0.35 ** 2 - 0.25 ** 2); Etr = max(0.0, Eh - Ev); slope = np.mean(list(slopes.values()))
    print(f"    E_h budget of the anchor: E_h {Eh:.6f} = valley {Ev:.6f} (h^2 from the last step; the three meshes give h^1.8) + remainder <= {Etr:.6f}; "
          "transverse slopes d|C|^2/dE_h (L_c = 15): " + ", ".join(f"{k} {v:+.0f}" for k, v in slopes.items()))
    add("transverse remainder", 0.0, max(-slope * Etr, abs(s) / 2), f"E_h budget: {Etr:.6f} x slope {slope:.0f}")
    steps = {"cubic L15 n=10": d("P16B15_r1") - d("P16B15_r0"), "cubic L15 n=12": d("N12P16B15_r1") - d("N12P16B15_r0"), "cubic L22": d("L22P16B22_r1") - d("L22P16B22_r0"),
             "quartic L22, 5 -> 10 channels": m(ref) - m(e0), "quartic L22, one round": m(e1) - m(e0)}
    add("channel rounds to saturation", 0.0, 2.0 * max(abs(v) for v in steps.values()), ", ".join(f"{k} {v:+.3f}" for k, v in steps.items()))
    t22 = m(t24) - m(e0); t15 = d("T24B15_r0") - d("P16B15_r0")
    add("tolerance 1e-16 -> 1e-24 (valley h 0.5)", t22, max(0.001, abs(t22 - t15)), f"dE_h {Ef(t24) - Ef(e0):+.6f}; cubic L_c = 15 window: {t15:+.3f}")
    R, u, _, _ = load(paths[ANCHOR]); free = []; lam = []
    for ps in ((3,), (3, 6)):
        for lo in (7, 8, 9, 10):
            for hi in (17, 18, 19):
                o = minimize_scalar(lambda x: fit(R, u, x * Eh, lo, hi, ps)[2], bounds=(0.5, 2.5), method="bounded", options=dict(xatol=1e-6))
                free.append(fit(R, u, o.x * Eh, lo, hi, ps)[0]); lam.append(o.x)
    ends = {hi: np.mean([fit(R, u, Eh, lo, hi, ps)[0] for ps in ((3,), (3, 6)) for lo in (7, 8, 9, 10)]) for hi in (13, 15)}
    add("far-field model of the anchor", 0.0, max(anc_err, abs(np.mean(free) - anchor)),
        f"half range of the series fits {anc_err:.3f}; k^2 free: {np.mean(free):.3f} ({min(free):.3f} .. {max(free):.3f}), k^2/E_h = {np.mean(lam):.3f} ({min(lam):.3f} .. {max(lam):.3f}); "
        + ", ".join(f"windows ending at {hi}: {v:.3f}" for hi, v in ends.items()))
    q = m(ref) - m(g12); add("quadrature across the valley, 12 -> 16 points (valley h 0.5)", 0.0, max(abs(q), 0.001), f"direct fits differ by {q:+.4f}")
    add("profile slice quadrature", 0.0, 0.02, f"int P dR = {rows[ANCHOR]['norm']:.5f} for the anchor; 8 against 24 points on the slices of the quartic long box: 3e-6 in P")
    report()


budget()
