"""Style and shared functions of the figure (fig_density.py).
Computer Modern through LaTeX, inward ticks on all four sides, no titles or grids, a restrained palette; widths for two-column
REVTeX (column 3.4 in, page 7.0 in).  Units g = 2, R_- = 1, where l_p = 0.430.
Data: data/profile_<tag>.npz, the density of the largest singular value on the slices r1 = R (a copy of
../data/profiles/solve_ch_<tag>_PR.npz, written by ground_state/tail_fit.py --save), for the best state
REF = Q4L22G16V25_r0 (10 x 10 transverse mesh with quartic elements in a and b, elements of at most 0.25 along the valley,
pruning 1e-16, box 22, sixteen Gauss points across the valley, E_h = 0.0003).
Run: python figures/fig_density.py; set BFSS_FIGURE_OUT for the output folder (default: this folder). LaTeX is required."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import gamma, pi
from scipy.special import jv
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
INK, BLUE, RED, GREY = "#1a1a1a", "#1f4e79", "#b03a2e", "#8c8c8c"
plt.rcParams.update({
    "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath,amssymb}", "font.family": "serif",
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "lines.linewidth": 1.1, "legend.frameon": False, "legend.handlelength": 2.2, "axes.grid": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
COL, FULL = 3.4, 7.0
LP = 0.430                                          # Planck length in the units g = 2, R_- = 1 (l_p^6 = 1/16 pi^2)
NU = 5.5; OMEGA8 = 2 * pi ** 4.5 / gamma(4.5); NORM = 8 / 9 * OMEGA8      # P(R) = NORM |C|^2 R^-10, Omega_8 = 32 pi^4/105
C2_PRED = 893025 / (32 * pi ** 4)                   # 286.49, i.e. P(R) = 7560 R^-10
C2_NUM, C2_ERR = 286.4, 0.2                         # this work: far-field fit of REF, with the conservative uncertainty rounded up (../data/tail_c2/final.txt)
REF = "Q4L22G16V25_r0"                              # the best state
HVAL = 0.25                                         # its largest element along the valley
# line styles
STATE = dict(color=BLUE, lw=2.4, alpha=0.5, solid_capstyle="butt")     # the solved state: wide translucent line ("-")
EXACT = dict(color=INK, lw=0.8, dashes=(4, 2.5))                        # an analytic expectation: thin dashed line ("--")
FIT = dict(color=RED, lw=1.3)                                           # a fit to the solved state: dotted line (":")
BAND = dict(color=GREY, alpha=0.18, lw=0)                               # a shaded region (the wall of the box, the band of a result)
# palette: BLUE (the state), INK (expectations, axes), RED (fits), GREY (shading)
def panel(ax, s, x=0.035, y=0.955, **kw): ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, **kw)
def top_lp(ax, ticks, label=r"$r_1/l_p$"):
    """secondary x axis in Planck lengths for an axis in the units of the code (g = 2)"""
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / LP, lambda y: y * LP))
    sec.set_xticks(ticks); sec.set_xlabel(label, labelpad=4); sec.tick_params(direction="in", which="both")
    sec.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.tick_params(top=False, which="both"); return sec
def profile(tag):
    z = np.load(os.path.join(DATA, f"profile_{tag}.npz")); R, P, E = z["R"], z["P"], float(z["E"])
    return R, P, E, R[P > 1e-12 * P.max()].max() + 0.05            # last point: the Dirichlet wall of the box
def far_fit(R, P, E, lo, hi, p=3.2):
    """u = sqrt(P) = A s_- + B s_+ + A a s_- R^-p, s_-+ = sqrt(R) J_{-+11/2}(kR), k^2 = E_h, fitted on lo <= R <= hi:
    returns |C|^2 (from A), beta = B/A (reflection off the wall) and a"""
    m = (R >= lo) & (R <= hi); r, u = R[m], np.sqrt(P[m]); k = np.sqrt(E)
    sm, sp = np.sqrt(r) * jv(-NU, k * r), np.sqrt(r) * jv(NU, k * r)
    X = np.stack([sm, sp, sm * r ** -p], 1) / u[:, None]; sc = np.linalg.norm(X, axis=0); co = np.linalg.lstsq(X / sc, np.ones_like(u), rcond=None)[0] / sc
    return (co[0] * (k / 2) ** -NU / gamma(1 - NU)) ** 2 / NORM, co[1] / co[0], co[2] / co[0]
def bessel_factor(R, E, beta):
    """F(R) = [S(R) R^5]^2, S = sqrt(R)[J_{-11/2}(kR) + beta J_{11/2}(kR)] (k/2)^{11/2} Gamma(-9/2): the zero-energy tail times F
    is the far field at the energy E_h of the discrete state, with the reflection off the wall (F -> 1 as kR -> 0)"""
    k = np.sqrt(E); S = np.sqrt(R) * (jv(-NU, k * R) + beta * jv(NU, k * R)) * (k / 2) ** NU * gamma(1 - NU)
    return (S * R ** 5) ** 2
def save(fig, name):
    output = os.environ.get("BFSS_FIGURE_OUT", HERE)
    os.makedirs(output, exist_ok=True)
    fig.savefig(os.path.join(output, f"{name}.pdf"))
    plt.close(fig); print("wrote", os.path.join(output, f"{name}.pdf"))
