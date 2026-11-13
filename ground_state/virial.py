"""Virial terms of a saved solve_ch state (SU(2) BFSS, units g = 2): <T> = radial + angular kinetic energy, the bosonic
potential <V_B> and the fermionic (Yukawa) term <V_F>.  For the exact zero-energy state <T> = <V_B> = -<V_F>/2, and the three
terms add up to the energy of the state, which is checked here against the stored eigenvalue E_h: the script stops with an
error, and writes nothing, if they differ by more than --tol (default 1e-5).  The terms are evaluated on
the node fermion vectors of the saved state (single precision, so that they carry about seven digits).  For the best state they are
<T> = 27.166, <V_B> = 27.164, <V_F> = -54.329 (data/virial).  The state must be read with the quadrature it was computed with (FEM_NGAUSS_AB=16 for the best state).
Usage: python virial.py solve_ch_<tag>.pkl [--ops fspace_ops.pkl] [--outdir DIR] [--tol 1e-5]   -> DIR/virial_<tag>.json and a
    printed line (ten minutes and 7 GB for the quartic state of box length 15; under two hours and about 20 GB for the best state)"""
import numpy as np, scipy.sparse as sp, pickle, sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
from fem3d import Grid, env_valley, prune_scale, check_quadrature

argv = sys.argv[1:]
if not argv or argv[0] in ('-h', '--help'):
    print(__doc__.strip()); sys.exit(0 if argv else 2)
def opt(name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default
ops_file = opt("--ops", os.environ.get("BFSS_FSPACE_OPS", "fspace_ops.pkl")); outdir = opt("--outdir", "."); tol = float(opt("--tol", "1e-5"))
os.makedirs(outdir, exist_ok=True)
t0 = time.time()
d = pickle.load(open(argv[0], "rb")); a, b, c = d["grid"]; qs = d.get("qs") or (2, 2, 2)
check_quadrature(d, os.path.basename(argv[0]))
Psi = d["Phi"].astype(complex)                                  # (kept nodes, fermion space): Phi[alpha] = U_alpha x_alpha
ops = pickle.load(open(ops_file, "rb"))
keys = [("Y", i) for i in range(3)] + [(k, p) for p in [(0, 1), (0, 2), (1, 2)] for k in ("KS2", "KD2")] + [("O", j) for j in range(3)]
Mt = [sp.csr_matrix(ops["Y"][k[1]] if k[0] == "Y" else (ops["O"][k[1]] if k[0] == "O" else ops[k[0]][k[1]])) for k in keys]
G = Grid(a, b, c, qs=qs); K, M, P = G.assemble(env=env_valley(1.0))
keep, K, M, P = prune_scale(K, M, P, keep=d["keep"])            # the basis of the run: same kept nodes, unit mass diagonal
assert K.shape[0] == Psi.shape[0], (K.shape, Psi.shape)
print(f"{os.path.basename(argv[0])}: orders {tuple(qs)}, {K.shape[0]} spatial functions, operators assembled ({time.time() - t0:.0f}s)", flush=True)
def expv(S, Mf=None):
    phi = S @ Psi
    if Mf is not None: phi = (Mf @ phi.T).T
    return float(np.real(np.vdot(Psi, phi)))
nrm = expv(M)
rad = expv(K) / nrm; vb = expv(P["Vbos"]) / nrm
ang = sum(expv(P[k], Mt[t]) for t, k in enumerate(keys) if k[0] != "Y") / nrm
yuk = sum(expv(P[k], Mt[t]) for t, k in enumerate(keys) if k[0] == "Y") / nrm
T = rad + ang; E = T + vb + yuk; Eh = float(d["ev"][0])
out = dict(file=os.path.basename(argv[0]), E_h=Eh, T=T, T_radial=rad, T_angular=ang, V_B=vb, V_F=yuk, sum=E,
           T_over_VB=T / vb, VF_over_VB=yuk / vb)
print(f"    virial: <T> = {T:.4f} (radial {rad:.4f} + angular {ang:.4f}), <V_B> = {vb:.4f}, <V_F> = {yuk:.4f}, sum {E:.5f} "
      f"(stored E_h {Eh:.5f}); T/V_B = {T / vb:.5f}, V_F/V_B = {yuk / vb:.5f} ({time.time() - t0:.0f}s)", flush=True)
if not abs(E - Eh) <= tol:
    sys.exit(f"FAIL  the three terms add up to {E:.7f} and the stored eigenvalue is {Eh:.7f}: they differ by more than {tol:g} (--tol); nothing written")
tag = os.path.basename(argv[0]).replace("solve_ch_", "").replace(".pkl", "")
json.dump(out, open(os.path.join(outdir, f"virial_{tag}.json"), "w"), indent=1)
