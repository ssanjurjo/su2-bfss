"""Tail coefficient |C|^2 from the radial profiles of data/profiles, with one fixed protocol.
  (1) Direct far-field fit of a long-box state REF: u = sqrt(P) is fitted to
          sqrt(R) [A J_{-11/2}(kR) (1 + sum_p D_p R^-p) + B J_{11/2}(kR)],    k^2 = E_h,
      on the windows [7..10, 17..19], with the corrections R^-3 and R^-3, R^-6; A gives |C|^2, and the mean over the 24
      fits is C_ref.
  (2) Every state: the pointwise coefficient C2_pt(R) = P(R)/(NORM S(R)^2), S the order-11/2 Bessel solution at the state's
      own k^2 = E_h with its own fitted reflection B/A (window [9, R_wall - 2.4] for c <= 15, [12, R_wall - 3.4] for long
      boxes); dN = mean over R in [6, 9.5] of C2_pt(state) - C2_pt(LE10B15), where the state LE10B15 is a common baseline
      that drops out of every difference; C2 = C_ref + dN - dN(REF).
Also listed: the two-term Bessel fit on [10, 13] for the boxes c <= 15, and C2 with three other windows for dN.
final.py turns the table into the final value.
Usage: python c2_table.py [--json out.json] [--ref TAG]   (--ref: the long-box state of step (1); default Q4L22G16V25_r0,
    the best state; seconds)"""
import numpy as np, os, re, sys, json, glob
from math import gamma, pi
from scipy.special import jv
HERE = os.path.dirname(os.path.abspath(__file__))
SHIP = os.path.join(os.environ.get("BFSS_DATA", os.path.join(HERE, "..", "..", "data")), "profiles")
NORM = 8 / 9 * 2 * pi ** 4.5 / gamma(4.5); NU = 5.5; PRED = 893025 / (32 * pi ** 4)
def load(path):
    z = np.load(path); R, P, E = z["R"], z["P"], float(z["E"])
    return R, np.sqrt(np.maximum(P, 1e-300)), E, R[P > 1e-12 * P.max()].max() + 0.05
def lead(k): return (k / 2) ** -NU / gamma(1 - NU)
def fit(R, u, E, lo, hi, ps=()):
    m = (R >= lo) & (R <= hi); k = np.sqrt(E); r = R[m]; sm, sp = np.sqrt(r) * jv(-NU, k * r), np.sqrt(r) * jv(NU, k * r)
    X = np.stack([sm, sp] + [sm * r ** -p for p in ps], 1) / u[m][:, None]; sc = np.linalg.norm(X, axis=0)
    co = np.linalg.lstsq(X / sc, np.ones(m.sum()), rcond=None)[0] / sc      # scaled columns: at small k the reflected wave is otherwise below the cutoff of lstsq
    return (co[0] * lead(k)) ** 2 / NORM, co, np.sqrt(np.mean((X @ co - 1) ** 2))
def c2pt(path):
    R, u, E, Rw = load(path); k = np.sqrt(E); lo, hi = (12, Rw - 3.4) if Rw > 18 else (9, Rw - 2.4)
    _, co, _ = fit(R, u, E, lo, hi); S = np.sqrt(R) * (jv(-NU, k * R) + co[1] / co[0] * jv(NU, k * R)) / lead(k)
    return R, u ** 2 / (NORM * S ** 2)
paths = {os.path.basename(p)[9:-7]: p for p in sorted(glob.glob(SHIP + "/solve_ch_*_PR.npz"))}
def main():
    Rr, Cr = c2pt(paths["LE10B15"])                        # common baseline of the window amplitudes dN
    REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "Q4L22G16V25_r0"
    R, u, E, Rw = load(paths[REF]); asym = {}
    SERIES = ((3,), (3, 6))                      # corrections as a series in R^-3 (the expansion parameter of the valley)
    allv = {}
    for ps in SERIES:
        allv[ps] = [fit(R, u, E, lo, hi, ps)[0] for lo in (7, 8, 9, 10) for hi in (17, 18, 19)]; asym[ps] = np.mean(allv[ps])
    Cref = np.mean([asym[ps] for ps in SERIES])
    ser = np.concatenate([allv[ps] for ps in SERIES])
    print(f"direct far-field fit of {REF}: " + ", ".join(f"p={ps}: {asym[ps]:.3f}" for ps in SERIES))
    print(f"  R^-3 series fits: mean {ser.mean():.3f}, range {ser.min():.3f} .. {ser.max():.3f}")
    def dN(path, lo=6, hi=9.5):
        Rt, C = c2pt(path); nn = min(len(C), len(Cr)); s_ = (Rr[:nn] >= lo) & (Rr[:nn] <= hi); d = (C[:nn] - Cr[:nn])[s_]
        return float(d.mean()), float(d.max() - d.min())
    dref = dN(paths[REF])[0]
    rows = []
    for tag, p in paths.items():
        R, u, E, Rw = load(p)
        if Rw < 14: continue
        d, spr = dN(p); w = fit(R, u, E, 10, 13)[0] if Rw < 18 else float("nan")
        alt = [Cref + dN(p, a, b)[0] - dN(paths[REF], a, b)[0] for (a, b) in ((5, 8), (7, 10.5), (8, 11))]
        rows.append(dict(tag=tag, E=E, Rw=float(Rw), C2=float(Cref + d - dref), dN=d, spread=spr, plain_10_13=float(w), alt_windows=[float(x) for x in alt],
                         norm=float((getattr(np, "trapezoid", None) or np.trapz)(u * u, R))))
    print(f"C_ref (direct far-field fit of {REF}) = {Cref:.3f}; prediction (not used) {PRED:.3f}")
    print(f"{'state':18s} {'E_h':>9s} {'|C|^2':>8s} {'dN':>7s} {'spread':>6s} {'[10,13]':>8s}  alt dN windows [5,8] [7,10.5] [8,11]   norm")
    for r in sorted(rows, key=lambda r: r["tag"]):
        print(f"{r['tag']:18s} {r['E']:9.6f} {r['C2']:8.3f} {r['dN']:+7.3f} {r['spread']:6.3f} {r['plain_10_13']:8.2f}  " + " ".join(f"{x:8.3f}" for x in r["alt_windows"]) + f"  {r['norm']:.5f}")
    if "--json" in sys.argv: json.dump(dict(C_ref=float(Cref), ref=REF, rows=[{k: (None if v != v else v) for k, v in r.items()} for r in rows]), open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)

if __name__ == "__main__":
    main()
