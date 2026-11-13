"""The figure (full width, two panels), for the best state (fig_style.REF = Q4L22G16V25_r0: 10 x 10 transverse mesh with
quartic elements in a and b, elements of at most 0.25 along the valley, pruning tolerance 1e-16, wall at r1 = 22.3, E_h = 0.0003).
(a) The density P(r1) of the largest singular value, from the core to the wall on a log scale, against the zero-energy tail
    7560 R^-10 and its continuation to the energy of the discrete state (order-11/2 Bessel factor at k^2 = E_h with the
    reflection off the wall); top axis in Planck lengths.
(b) The local tail coefficient |C|^2(R) = R^10 P(R) / (8/9 Omega_8 F(R)), F the Bessel factor at k^2 = E_h, i.e. the value of |C|^2
    that the density at R would give.  Dotted: the fit u = (A s_- + B s_+)(1 + a R^-3) on 7 <= R <= 19.  Band: the result
    286.4 +- 0.2 (../data/tail_c2/final.txt), the mean of the far-field fits with a conservative uncertainty, the full change
    under the refinement of the elements along the valley, rounded up; line: the prediction 893025/(32 pi^4) = 286.49.  The x axis of (b) is uniform in 1/R, so that R = infinity
    is the right edge.  The local coefficient carries a ripple with the period (about 0.25) of the finite elements along the
    valley; it is shown as a running mean over that length (the fits use the raw profile)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *
def element_mean(y, h=0.05, w=0.5):
    """running mean over one element length w along the valley (trapezoidal window on the uniform grid h); trims w/2 at both ends"""
    n = int(round(w / h)); k = np.ones(n + 1); k[0] = k[-1] = 0.5; k /= n
    if n % 2: k = np.ones(n) / n                     # an odd number of points per element: the plain mean over one element
    return np.convolve(y, k, mode="valid"), (len(k) - 1) // 2
def local(tag, lo, hi):
    """R, the local coefficient, the fitted asymptote and correction, the energy and the wall of a state"""
    R, P, E, Rw = profile(tag); C2, beta, a = far_fit(R, P, E, lo, hi, p=3.0)
    return R, R ** 10 * P / (NORM * bessel_factor(R, E, beta)), C2, a, E, Rw
fig, (A, B) = plt.subplots(1, 2, figsize=(FULL, 2.75), gridspec_kw={"wspace": 0.24, "left": 0.075, "right": 0.995, "bottom": 0.15, "top": 0.86})

# ---------------- (a) the density of r1 from the core to the wall
R, P, E, Rw = profile(REF); C2, beta, a = far_fit(R, P, E, 7.0, 19.0, p=3.0)
m = (R >= 0.55) & (R <= Rw - 0.05)
A.axvspan(Rw, 25, **BAND)
A.semilogy(R[m], P[m], "-", **STATE, zorder=2, label=r"solved state")
Rt = np.linspace(2.4, 25, 400)
A.semilogy(Rt, 7560 * Rt ** -10.0, "--", **EXACT, zorder=3, label=r"$7560\,R^{-10}$")
Rb = np.linspace(2.8, Rw - 0.005, 800)
A.semilogy(Rb, 7560 * Rb ** -10.0 * bessel_factor(Rb, E, beta), ":", **FIT, zorder=4,
           label=r"$7560\,R^{-10}$ at $k^2=E_h$")
A.set_xlim(0, 24); A.set_ylim(1e-12, 1e2); A.set_xlabel(r"$r_1$"); A.set_ylabel(r"$\mathcal P(r_1)$", labelpad=2)
A.set_xticks([0, 5, 10, 15, 20])
A.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e-12, 1e-9, 1e-6, 1e-3, 1]))
A.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([10.0 ** k for k in range(-12, 1)]))
A.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
top_lp(A, [0, 10, 20, 30, 40, 50])
A.text(Rw + 0.85, 1e-5, "boundary region", rotation=90, ha="center", va="center", fontsize=8, color="#5a5a5a")
A.legend(loc="upper right", bbox_to_anchor=(0.93, 1.0), borderaxespad=0.5, handlelength=2.3)
panel(A, "(a)")

# ---------------- (b) the local tail coefficient, on an axis uniform in 1/R
B.axhspan(C2_NUM - C2_ERR, C2_NUM + C2_ERR, **BAND, zorder=0)
B.axhline(C2_PRED, **EXACT, zorder=1)
RL, locL, C2L, aL, EL, RwL = local(REF, 7.0, 19.0)
m = (RL >= 5.6) & (RL <= RwL - 2.55)
y, t = element_mean(locL[m], w=HVAL); Rm = RL[m][t:len(RL[m]) - t]
B.plot(1 / Rm, y, "-", **STATE, zorder=2, label=r"solved state")
xi = np.linspace(0, 1 / 6.0, 400); Rf = 1 / np.maximum(xi, 1e-9); fit = C2L * (1 + aL * Rf ** -3.0) ** 2
B.plot(xi, fit, ":", **FIT, zorder=3, label=r"fit with $R^{-3}$")
print(f"{REF}: E_h = {EL:.5f}, wall {RwL:.2f}, |C|^2 (R^-3 fit on [7,19]) = {C2L:.3f}, D_3 = {aL:.3f}")
print("local coefficient over the asymptote of the fit: " + ", ".join(f"R = {r}: {100 * (locL[np.argmin(abs(RL - r))] / C2L - 1):+.2f}% (fit {100 * ((1 + aL * r ** -3.0) ** 2 - 1):+.2f}%)" for r in (7, 10, 15)))
Rt = [6, 7, 8, 10, 15, 20]
B.set_xticks([1 / r for r in Rt] + [0]); B.set_xticklabels([str(r) for r in Rt] + [r"$\infty$"])
B.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([1 / r for r in (9, 12, 30, 50)]))
B.set_xlim(1 / 5.8, 0); B.set_ylim(284.9, 292.0); B.set_xlabel(r"$R=r_1$"); B.set_ylabel(r"$|C|^2(R)$", labelpad=3)
B.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1)); B.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
hs = B.get_legend_handles_labels()
leg = B.legend(*hs, loc="lower left", borderaxespad=0.45, handlelength=2.0, labelspacing=0.3); B.add_artist(leg)
pr = matplotlib.lines.Line2D([], [], **EXACT); bd = matplotlib.patches.Patch(**BAND)
B.legend([pr, bd], [r"predicted, $286.49$", rf"this work, ${C2_NUM:.1f}\pm{C2_ERR:.1f}$"], loc="upper right",
         borderaxespad=0.45, handlelength=2.0, labelspacing=0.3)
panel(B, "(b)")
save(fig, "fig_density")
