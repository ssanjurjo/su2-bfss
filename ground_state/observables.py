"""Non-protected observables of a solve_ch singlet state (SU(2) BFSS, units g = 2): Gram moments inside the box and completed beyond
R_m by the zero-energy tail P(R) = (8/9) Omega_8 |C|^2 R^-10 (t1 ~ R^2 + 8/R, t2 ~ R^4, t3 ~ R^6 there), the bosonic potential
<V_bos> = (<t1^2> - <t2>)/2 (= <T> = -<V_Y>/2 for the exact state by the virial theorem), and the radial densities P(r1), P(rho),
rho^2 = X.X = r1^2 + r2^2 + r3^2.  Element orders are read from the state file.  moments.py combines the files of the states
into the tables of moments.
The state must be read with the Gauss orders it was computed with (FEM_NGAUSS_AB, FEM_NGAUSS_C of fem3d; they set the scaling
of its coefficients and its norm); OBS_NQ (default 6) is the number of Gauss points per direction of the moment integrals.
Beyond the matching radius the moments use the tail with the predicted coefficient |C|^2 = 893025/(32 pi^4) = 286.49,
unless the second argument gives another value.  With the fitted 286.4 the largest change is -0.25 in <t1^4> = 16140(10).
Usage: python observables.py solve_ch_<tag>.pkl [C2=286.5] [Rm list, e.g. 8,10,12] [--outdir DIR]  -> DIR/obs_<tag>.pkl
    (+ a printed table; from a minute and 3 GB for a state with cubic elements in the box of length 15 to half an hour and
    about 20 GB for the best state with OBS_NQ=12)"""
import numpy as np, pickle, sys, os, itertools
from math import gamma, pi
argv = sys.argv[1:]
if not argv or argv[0] in ('-h', '--help'):
    print(__doc__.strip()); sys.exit(0 if argv else 2)
outdir = "."
if "--outdir" in argv:
    i = argv.index("--outdir"); outdir = argv[i + 1]; del argv[i:i + 2]
os.makedirs(outdir, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
d = pickle.load(open(argv[0], "rb"))
qs = d.get("qs") or (2, 2, 2)
import fem3d
from fem3d import Grid, env_valley, jac
fem3d.check_quadrature(d, os.path.basename(argv[0]))
from scipy.integrate import quad
OMEGA8 = 2 * pi ** 4.5 / gamma(4.5); NORM = 8 / 9 * OMEGA8
C2 = float(argv[1]) if len(argv) > 1 else 893025 / (32 * pi ** 4)
a, b, c = d["grid"]; nch = d["nch"]; keep = d["keep"]; cmax = float(c[-1])
Rms = [float(x) for x in argv[2].split(",")] if len(argv) > 2 else ([8.0, 10.0, 12.0] if cmax >= 14 else [6.0, 7.0, 8.0])
Phi0 = d["Phi"].astype(complex)
G = Grid(a, b, c, qs=qs); K, M, P = G.assemble(terms={"Vbos"}, env=env_valley(1.0))
s = 1 / np.sqrt(M.diagonal().real[keep]); D = Phi0.shape[1]
full = np.zeros((G.N + 1, D), complex); full[keep] = Phi0 * s[:, None]
nrm = np.vdot(full[:-1], M @ full[:-1]).real
na, nb, nc = G.shape; gidx = np.full(na * nb * nc, G.N); gidx[G.free] = np.arange(G.N)
qa, qb, qc = G.qs; Ls = [fem3d.lagrange1d(x) for x in (qa, qb, qc)]
ea, eb, ec = G.ea, G.eb, G.ec; ha, hb, hc = np.diff(ea), np.diff(eb), np.diff(ec)
corners = np.array(list(itertools.product(range(qa + 1), range(qb + 1), range(qc + 1)))); NQ = int(os.environ.get("OBS_NQ", "6"))
GA, GW = np.polynomial.legendre.leggauss(NQ); u3 = np.array(list(itertools.product((1 + GA) / 2, repeat=3))); w3 = np.prod(np.array(list(itertools.product(GW, repeat=3))), 1) / 8
Nv = np.stack([Ls[0][p][0](u3[:, 0]) * Ls[1][q][0](u3[:, 1]) * Ls[2][t][0](u3[:, 2]) for (p, q, t) in corners], 1)
names = ["t1", "t1^2", "t2", "t3", "t1^3", "t1t2", "t1^4", "t2^2", "t1^2t2", "t1t3"]
def words(t1, t2, t3): return np.stack([t1, t1**2, t2, t3, t1**3, t1 * t2, t1**4, t2**2, t1**2 * t2, t1 * t3])
tailf = [lambda R: R**2 + 8 / R, lambda R: (R**2 + 8 / R)**2, lambda R: R**4, lambda R: R**6, lambda R: (R**2 + 8 / R)**3,
         lambda R: (R**2 + 8 / R) * R**4, lambda R: (R**2 + 8 / R)**4, lambda R: R**8, lambda R: (R**2 + 8 / R)**2 * R**4, lambda R: (R**2 + 8 / R) * R**6]
edges = np.arange(0, 30.01, 0.25); h_r1 = np.zeros(len(edges) - 1); h_rho = np.zeros(len(edges) - 1)
allm = np.zeros(len(names)); nall = 0.0; core = {Rm: np.zeros(len(names)) for Rm in Rms}; ncore = {Rm: 0.0 for Rm in Rms}
r1_all = []; wt_all = []
for i in range(len(ea) - 1):
    for j in range(len(eb) - 1):
        for k in range(len(ec) - 1):
            ids = gidx[((qa * i + corners[:, 0]) * nb + (qb * j + corners[:, 1])) * nc + (qc * k + corners[:, 2])]
            psi = Nv @ full[ids]
            r3 = ea[i] + u3[:, 0] * ha[i]; r2 = r3 + eb[j] + u3[:, 1] * hb[j]; r1 = r2 + ec[k] + u3[:, 2] * hc[k]
            g = env_valley(1.0)(r1, r2, r3)[0]
            wt = jac(r1, r2, r3) * g**2 * ha[i] * hb[j] * hc[k] * w3 * np.einsum("pd,pd->p", psi.conj(), psi).real / nrm
            t1 = r1**2 + r2**2 + r3**2; t2 = r1**4 + r2**4 + r3**4; t3 = r1**6 + r2**6 + r3**6
            vals = words(t1, t2, t3); allm += vals @ wt; nall += wt.sum()
            for Rm in Rms:
                m = r1 < Rm; core[Rm] += vals[:, m] @ wt[m]; ncore[Rm] += wt[m].sum()
            h_r1 += np.histogram(r1, edges, weights=wt)[0]; h_rho += np.histogram(np.sqrt(t1), edges, weights=wt)[0]
out = dict(file=argv[0], E_h=float(d["ev"][0]), nch=nch, grid=(a, b, c), qs=tuple(qs), cmax=cmax, C2=C2, names=names,
           box=allm / nall, norm_box=nall, Rms=Rms, completed={}, tailpart={}, ncore=ncore, ntail={}, edges=edges,
           P_r1=h_r1 / 0.25, P_rho=h_rho / 0.25)
print(f"{argv[0]}: E_h = {out['E_h']:.5f}, nch {nch}, orders {tuple(qs)}, box c <= {cmax:g}; norm inside the box {nall:.5f}")
print(f"   {'moment':8s} {'box':>12s} " + " ".join(f"{'R_m=%g' % Rm:>12s}" for Rm in Rms))
for Rm in Rms:
    tail = np.array([quad(lambda R: f(R) * NORM * C2 * R**-10, Rm, np.inf)[0] for f in tailf]); ntail = NORM * C2 * Rm**-9 / 9
    out["completed"][Rm] = (core[Rm] + tail) / (ncore[Rm] + ntail); out["tailpart"][Rm] = tail / (ncore[Rm] + ntail); out["ntail"][Rm] = ntail
for nm, bx, row in zip(names, out["box"], zip(*[out["completed"][Rm] for Rm in Rms])):
    print(f"   {nm:8s} {bx:12.4f} " + " ".join(f"{v:12.4f}" for v in row))
vb = [(out["completed"][Rm][1] - out["completed"][Rm][2]) / 2 for Rm in Rms]
print(f"   {'V_bos':8s} {(out['box'][1] - out['box'][2]) / 2:12.4f} " + " ".join(f"{v:12.4f}" for v in vb))
tag = os.path.basename(argv[0]).replace("solve_ch_", "").replace(".pkl", "")
pickle.dump(out, open(os.path.join(outdir, f"obs_{tag}.pkl"), "wb"))
