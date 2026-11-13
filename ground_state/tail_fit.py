"""Radial profile of a solve_ch solution and its fit to the zero-energy tail.
P(R) is the marginal density of the largest singular value r1 = R, evaluated exactly on the slice r1 = R (2D Gauss quadrature in
(a, b) = (r3, r2 - r3), c = R - a - b) and normalised with the FEM mass matrix.  In the Born-Oppenheimer region u(R) = sqrt(P(R))
obeys  -u'' + 30 u / R^2 = k^2 u  (l = 2 in nine dimensions), with k^2 = E_h, the eigenvalue of the discrete state, so
    u(R) = A sqrt(R) J_{-11/2}(kR) + B sqrt(R) J_{11/2}(kR)     -> A (k/2)^{-11/2}/Gamma(-9/2) R^-5 + ... as k -> 0,
and the zero-energy tail P = (8/9) Omega_8 |C|^2 R^-10 gives |C|^2 = (A (k/2)^{-11/2} / Gamma(-9/2))^2 / ((8/9) Omega_8).
The B term absorbs the reflection off the Dirichlet wall of the box.  Also reported: the E = 0 fit u = A0 R^-5 + B0 R^6 and a fit
with k^2 free.  The fits behind the quoted value are made on the saved profile by tail_c2/c2_table.py and
tail_c2/final.py.
Element orders are read from the state file; the state must be read with the Gauss orders it was computed with
(FEM_NGAUSS_AB, FEM_NGAUSS_C of fem3d, six points by default); the slices use 8 x 8 points per element.
Usage: python tail_fit.py solve_ch_<tag>.pkl [Rlo Rhi] [--save] [--outdir DIR] [--virial --ops fspace_ops.pkl]
    --save writes solve_ch_<tag>_PR.npz (R, P, E_h) into DIR (default: the current directory); 30 to 60 minutes and 3 to 6 GB
    for a long-box state with cubic elements.  The values of R are computed one after the other, which takes a few tens
    of core-hours for the best state."""
import numpy as np, scipy.sparse as sp, pickle, sys, itertools, os
from math import gamma, pi
from scipy.special import jv, iv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
import fem3d
from fem3d import Grid, env_valley, jac
argv = sys.argv[1:]
if not argv or argv[0] in ('-h', '--help'):
    print(__doc__.strip()); sys.exit(0 if argv else 2)
def opt(name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default
outdir = opt("--outdir", "."); ops_file = opt("--ops", os.environ.get("BFSS_FSPACE_OPS", "fspace_ops.pkl"))
os.makedirs(outdir, exist_ok=True)
OMEGA8 = 2 * pi ** 4.5 / gamma(4.5); NORM = 8 / 9 * OMEGA8; NU = 5.5
d = pickle.load(open(argv[0], "rb")); a, b, c = d["grid"]; nch = d["nch"]; keep = d["keep"]
fem3d.check_quadrature(d, os.path.basename(argv[0]))
Phi0 = d["Phi"].astype(complex) if "Phi" in d else np.einsum("adc,ac->ad", d["U"], d["X"].reshape(len(keep), nch)); E_h = d["ev"][0]
G = Grid(a, b, c, qs=d.get("qs")); K, M, P = G.assemble(terms={"Vbos"}, env=env_valley(1.0))
s = 1 / np.sqrt(M.diagonal().real[keep]); Ns = len(keep); D = Phi0.shape[1]
full = np.zeros((G.N + 1, D), complex); full[keep] = Phi0 * s[:, None]   # last row: Dirichlet 0
nrm = np.vdot(full[:-1], M @ full[:-1]).real
na, nb, nc = G.shape; gidx = np.full(na * nb * nc, G.N); gidx[G.free] = np.arange(G.N)
qa, qb, qc = G.qs; La, Lb, Lc = (fem3d.lagrange1d(x) for x in (qa, qb, qc))
ea, eb, ec = G.ea, G.eb, G.ec; ha, hb, hc = np.diff(ea), np.diff(eb), np.diff(ec)
corners = list(itertools.product(range(qa + 1), range(qb + 1), range(qc + 1))); NQ = 8; GA, GW = np.polynomial.legendre.leggauss(NQ)
cp = np.array(corners)

def P_of(R):
    tot = 0.0
    for i in range(len(ea) - 1):
        for j in range(len(eb) - 1):
            if ea[i] + eb[j] > R: continue
            u = np.repeat((1 + GA) / 2, NQ); v = np.tile((1 + GA) / 2, NQ); w2 = np.repeat(GW, NQ) * np.tile(GW, NQ) / 4
            A_ = ea[i] + u * ha[i]; B_ = eb[j] + v * hb[j]; C_ = R - A_ - B_
            ok = (C_ >= 0) & (C_ < ec[-1])
            if not ok.any(): continue
            A_, B_, C_, u, v, w2 = A_[ok], B_[ok], C_[ok], u[ok], v[ok], w2[ok]
            k = np.clip(np.searchsorted(ec, C_, side="right") - 1, 0, len(ec) - 2); w = (C_ - ec[k]) / hc[k]
            ids = gidx[((qa * i + cp[:, 0])[None, :] * nb + (qb * j + cp[:, 1])[None, :]) * nc + (qc * k[:, None] + cp[:, 2][None, :])]
            Nv = np.stack([La[p][0](u) * Lb[qq][0](v) * Lc[t][0](w) for (p, qq, t) in corners], axis=1)
            psi = np.einsum("pc,pcd->pd", Nv, full[ids])
            r3 = A_; r2 = A_ + B_; r1 = np.full_like(A_, R)
            g = env_valley(1.0)(r1, r2, r3)[0]
            tot += np.sum(jac(r1, r2, r3) * g**2 * ha[i] * hb[j] * w2 * np.einsum("pd,pd->p", psi.conj(), psi).real)
    return tot / nrm

args = [x for x in argv[1:] if not x.startswith("--")]
Rmax = ec[-1] + 1.0
TRAP = getattr(np, 'trapezoid', None) or np.trapz
Rs = np.arange(0.05, Rmax, 0.05); Ps = np.array([P_of(R) for R in Rs])
print(f"{argv[0]}: E_h = {E_h:.5f}; box c <= {ec[-1]}; int P(R) dR = {TRAP(Ps, Rs):.5f} (should be 1)")
print("   R     P(R)        R^10 P / ((8/9) Omega_8)")
for R, p in zip(Rs, Ps):
    if abs(R * 4 - round(R * 4)) < 1e-9 and R >= 1.5 and p > 0: print(f"  {R:5.2f}  {p:10.3e}  {R**10 * p / NORM:10.3e}")
u_ = np.sqrt(np.maximum(Ps, 0))

def basis(k2, R):
    if abs(k2) < 1e-12: return R ** -5.0, R ** 6.0, 1.0
    if k2 > 0:
        k = np.sqrt(k2); return np.sqrt(R) * jv(-NU, k * R), np.sqrt(R) * jv(NU, k * R), (k / 2) ** -NU / gamma(1 - NU)
    kap = np.sqrt(-k2); return np.sqrt(R) * iv(-NU, kap * R), np.sqrt(R) * iv(NU, kap * R), (kap / 2) ** -NU / gamma(1 - NU)

def fit(k2, lo, hi):
    m = (Rs >= lo) & (Rs <= hi); R = Rs[m]; y = u_[m]; f1, f2, lead = basis(k2, R)
    Amat = np.stack([f1, f2], 1) / y[:, None]; sc = np.linalg.norm(Amat, axis=0)       # relative residuals; columns scaled, since at
    coef = np.linalg.lstsq(Amat / sc, np.ones_like(y), rcond=None)[0] / sc            # small k the second one is below the cutoff of lstsq
    res = np.sqrt(np.mean((Amat @ coef - 1) ** 2)); C2 = (coef[0] * lead) ** 2 / NORM
    return C2, coef, res

wins = [(float(args[0]), float(args[1]))] if len(args) >= 2 else [(2.5, ec[-1] - 0.5), (3.0, ec[-1] - 0.3), (3.0, ec[-1]), (3.5, ec[-1])]
for lo, hi in wins:
    C2e, _, re = fit(E_h, lo, hi); C20, _, r0 = fit(0.0, lo, hi)
    k2s = np.linspace(-0.5, 1.0, 301); rs = [fit(k2, lo, hi)[2] for k2 in k2s]; kb = k2s[int(np.argmin(rs))]; C2b, _, rb = fit(kb, lo, hi)
    print(f"  window [{lo:.2f},{hi:.2f}]: |C|^2 = {C2e:9.1f} (k^2 = E_h, rel. rms {re:.1e});  {C20:9.1f} (k^2 = 0, {r0:.1e});  "
          f"{C2b:9.1f} (best k^2 = {kb:+.3f}, {rb:.1e})")
if "--virial" in argv:
    # <T> = <V_bos>, <V_Y> = -2 <V_bos> for the exact E = 0 state (dilatation virial theorem)
    ops = pickle.load(open(ops_file, "rb"))
    Kf, Mf_, Pf = G.assemble(env=env_valley(1.0)); F = full[:-1]
    def expv(S, Mt=None):
        phi = S @ F
        if Mt is not None: phi = (Mt @ phi.T).T
        return np.real(np.vdot(F, phi)) / nrm
    kin = expv(Kf); vb = expv(Pf["Vbos"]); yuk = sum(expv(Pf[("Y", i)], sp.csr_matrix(ops["Y"][i])) for i in range(3))
    ang = sum(expv(Pf[(kk, pr)], sp.csr_matrix(ops[kk][pr])) for pr in [(0, 1), (0, 2), (1, 2)] for kk in ("KS2", "KD2"))
    ang += sum(expv(Pf[("O", j)], sp.csr_matrix(ops["O"][j])) for j in range(3))
    T_ = kin + ang
    print(f"  virial: <T> = {T_:.4f} (radial {kin:.4f}, angular {ang:.4f}), <V_bos> = {vb:.4f}, <V_Y> = {yuk:.4f}, <H> = {T_ + vb + yuk:.5f}; "
          f"T/V_bos = {T_ / vb:.4f} (exact 1), V_Y/V_bos = {yuk / vb:.4f} (exact -2)")
if "--save" in argv:
    fn = os.path.join(outdir, os.path.basename(argv[0]).replace(".pkl", "_PR.npz")); np.savez(fn, R=Rs, P=Ps, E=E_h); print("wrote", fn)
