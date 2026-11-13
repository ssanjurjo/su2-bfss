"""Moments of the bound state, in the two normalizations, from the per-state files of observables.py.

The moments are those of the best state Q4L22G16V25_r0 (10 x 10 transverse mesh, quartic elements in a and b, elements of at
most 0.25 along the valley, pruning tolerance 1e-16, L_c = 22, sixteen Gauss points across the valley, E_h = 0.0003), each
completed beyond the middle matching radius R_m = 10 with the zero-energy tail.  The state with elements twice as long along
the valley (Q4L22G16_r0, E_h = 0.0009) is printed next to it: the valley mesh was the largest part of its E_h.  The error
combines in quadrature
  transverse elements: the slope d<O>/dE_h of the last transverse refinement (cubic -> quartic, P16B15_r1 -> Q4P16B15_r0)
                       times the part of E_h not accounted for by the valley mesh (E_h - 1/3 of the change of E_h when the
                       elements along the valley are halved);
  valley mesh:         the full change when the elements along the valley are halved (Q4L22G16_r0 -> Q4L22G16V25_r0), a
                       conservative choice: it is the largest effect observed;
  channels:            twice the change in one more enrichment round (N12P16B15_r0 -> r1);
  pruning tolerance:   1e-16 -> 1e-24 (P16B15_r0 -> T24B15_r0);
  matching radius:     the range over R_m = 8, 10, 12;
  virial relation:     for V_bos only, its difference from the kinetic energy <T> of the same state (data/virial), since the
                       exact state has <T> = <V_bos> and the table gives <K> through this relation.

Then the values in the normalization of the matrix-bootstrap literature (g = 1, so lengths x 2^{1/3} and energies x 2^{-2/3}; for SU(2)
X^I X^I = (t1/4) 1, so <Tr X^{2n}> = <Tr (X^I X^I)^n> = 2^{1-4n/3} <t1^n> and <K> = 2^{-2/3} <T>, with <T> = <V_bos> by the
virial theorem), and in terms of the matrices at g = 2.
Usage: python moments.py [--dir DATA/observables_lowtol]   (seconds)"""
import glob, json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
def opt(name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default
DATA = os.environ.get("BFSS_DATA", os.path.join(HERE, "..", "data"))
folder = opt("--dir", os.path.join(DATA, "observables_lowtol"))
LP = (16 * np.pi ** 2) ** (-1 / 6)
NAMES = ["t1", "t1^2", "t1^3", "t1^4", "V_bos", "t2", "t3"]


def load(f):
    o = pickle.load(open(f, "rb")); Rms = o["Rms"]; mid = Rms[len(Rms) // 2]
    def at(R):
        v = dict(zip(o["names"], o["completed"][R])); v["V_bos"] = (v["t1^2"] - v["t2"]) / 2; return v
    tag = os.path.basename(f)[4:-4]
    return dict(tag=tag, E=o["E_h"], v=at(mid), all=[at(R) for R in Rms])


def fmt(v, e):
    dig = min(4, max(0, int(np.ceil(-np.log10(e))))) if e > 0 else 2
    return f"{v:.{dig}f}" + (f"({int(round(e * 10 ** dig))})" if e > 0 else "")


S = {r["tag"]: r for r in map(load, sorted(glob.glob(os.path.join(folder, "obs_*.pkl"))))}
print(f"{'state':16s} {'E_h':>9s} " + " ".join(f"{n:>11s}" for n in NAMES))
for r in sorted(S.values(), key=lambda r: -r["E"]):
    print(f"{r['tag']:16s} {r['E']:9.6f} " + " ".join(f"{r['v'][n]:11.4f}" for n in NAMES))
best, h05, q15, p0, p1, t24 = S["Q4L22G16V25_r0"], S["Q4L22G16_r0"], S["Q4P16B15_r0"], S["P16B15_r0"], S["P16B15_r1"], S["T24B15_r0"]
n12a, n12b = S["N12P16B15_r0"], S["N12P16B15_r1"]
E_valley = (h05["E"] - best["E"]) / 3              # valley-mesh part of E_h that remains (E_h falls as h^2 or slightly slower along the valley)
E_tr = max(0.0, best["E"] - E_valley)              # what may remain in the transverse elements
est = {}
VIR = json.load(open(os.path.join(DATA, "virial", f"virial_{best['tag']}.json")))      # <T>, <V_B>, <V_F> of the best state (virial.py)
print(f"\nbest state {best['tag']} (E_h = {best['E']:.6f}); valley-mesh part of E_h {E_valley:.6f}, transverse remainder <= {E_tr:.6f}")
print(f"{'observable':10s} {'(h = 0.5)':>11s} {'(L_c=15)':>11s} {'valley':>9s} {'value':>11s} | {'transv.':>8s} {'valley':>8s} {'channels':>8s} {'tol.':>8s} {'R_m':>8s} {'error':>8s}  quoted")
for n in NAMES:
    shift = best["v"][n] - h05["v"][n]; v = best["v"][n]
    e_tr = abs((q15["v"][n] - p1["v"][n]) / (q15["E"] - p1["E"])) * E_tr
    e_val = abs(shift)
    e_ch = 2 * abs(n12b["v"][n] - n12a["v"][n])
    e_tol = abs(t24["v"][n] - p0["v"][n])
    rng = max(a[n] for a in best["all"]) - min(a[n] for a in best["all"])
    e_vir = abs(VIR["T"] - VIR["V_B"]) if n == "V_bos" else 0.0
    err = float(np.sqrt(e_tr ** 2 + e_val ** 2 + e_ch ** 2 + e_tol ** 2 + rng ** 2 + e_vir ** 2))
    est[n] = (v, err)
    print(f"{n:10s} {h05['v'][n]:11.4f} {q15['v'][n]:11.4f} {shift:+9.4f} {v:11.4f} | {e_tr:8.4f} {e_val:8.4f} {e_ch:8.4f} {e_tol:8.4f} {rng:8.4f} {err:8.4f}  {fmt(v, err)}")


# normalization of the matrix-bootstrap literature (g = 1)
conv = {"t1": 2 ** (-1 / 3), "t1^2": 2 ** (-5 / 3), "t1^3": 2.0 ** -3, "t1^4": 2 ** (-13 / 3), "V_bos": 2 ** (-2 / 3)}
label = {"t1": "<Tr X^2>", "t1^2": "<Tr X^4>", "t1^3": "<Tr X^6>", "t1^4": "<Tr X^8>", "V_bos": "<K>"}
print("\nmatrix-bootstrap normalization (g = 1):")
for n, c in conv.items():
    print(f"  {label[n]:9s} = {fmt(est[n][0] * c, est[n][1] * c):>12s}")
print(f"rms hyperradius sqrt(<t1>) = {np.sqrt(est['t1'][0]):.4f} = {np.sqrt(est['t1'][0]) / LP:.3f} l_p")

# the same moments in terms of the matrices, in the units g = 2
conv2 = {"t1": 2.0 ** -1, "t1^2": 2.0 ** -3, "t1^3": 2.0 ** -5, "t1^4": 2.0 ** -7, "V_bos": 1.0}
label2 = {"t1": "<Tr X^2>", "t1^2": "<Tr X^4>", "t1^3": "<Tr X^6>", "t1^4": "<Tr X^8>", "V_bos": "<V_B>"}
print("\nin terms of the matrices, g = 2 (<Tr X^2n> = <Tr (X^I X^I)^n> = 2^(1-2n) <t1^n>; <V_B> = -<Tr [X^I,X^J]^2>):")
for n, c in conv2.items():
    print(f"  {label2[n]:18s} = {fmt(est[n][0] * c, est[n][1] * c):>12s}")

bad = []
def check(name, got, ref, ok, src="expected"):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: computed {got}, {src} {ref}")
    if not ok: bad.append(name)
for n, ref in (("t1", "10.1729(7)"), ("t1^2", "109.14(2)"), ("t1^3", "1244.9(4)"), ("t1^4", "16140(10)"),
               ("V_bos", "27.164(3)"), ("t2", "54.82(2)"), ("t3", "407.2(3)")):
    check(f"g = 2, <{n}>", fmt(*est[n]), ref, fmt(*est[n]) == ref)
for n, ref in (("t1", "8.0742(6)"), ("t1^2", "34.378(6)"), ("t1^3", "155.61(5)"), ("t1^4", "800.6(5)"), ("V_bos", "17.112(2)")):
    got = fmt(est[n][0] * conv[n], est[n][1] * conv[n])
    check(f"g = 1, {label[n]}", got, ref, got == ref)
for n, ref in (("t1", "5.0865(4)"), ("t1^2", "13.643(3)"), ("t1^3", "38.90(1)"), ("t1^4", "126.09(8)"), ("V_bos", "27.164(3)")):
    got = fmt(est[n][0] * conv2[n], est[n][1] * conv2[n])
    check(f"g = 2, {label2[n]}", got, ref, got == ref)
check("rms size in Planck lengths", f"{np.sqrt(est['t1'][0]) / LP:.2f}", "7.4", abs(np.sqrt(est["t1"][0]) / LP - 7.4) < 0.05)
print(f"\n{len(bad)} failures" + (f": {bad}" if bad else ""))
sys.exit(1 if bad else 0)
