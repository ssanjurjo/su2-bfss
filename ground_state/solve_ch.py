"""Zero-energy state of SU(2) BFSS in the reduced problem, by the Galerkin method with finite elements.

Trial space: (g N_alpha) (x) u_{alpha,c}, where N_alpha are the Lagrange elements of fem3d with the valley envelope
g = exp(-(r1 r2^2 + r1 r3^2 + r2 r3^2)/2), and u_{alpha,c}, c < nch, are local fermion channels at node alpha in the
2066-dimensional space of reduction/fspace.py: first the lowest eigenvectors of the local operator
A_alpha = sum_t <f_t>_alpha M_t + <V_bos>_alpha (<.>_alpha the average over the node with weight J g^2 N_alpha), then, in
each enrichment round, two new vectors per node built from the residual of the current state (the block-preconditioned
residual and the projected residual).  Nodes on the walls r1 = r2 and r2 = r3 keep their channels in the kernel of K_S^2
of that wall.  --seed FILE (repeatable) starts from saved states: each is interpolated to the new grid and added as a
channel at every node, so that one solve (--enrich 0) or one more round suffices; a seed is read with the Gauss orders it
records, and is refused if it was computed with another operators file.  --ptol is the pruning tolerance of the
basis.  The lowest eigenpairs are found by shift-invert iteration: SciPy SuperLU for moderate problems, a CHOLMOD Cholesky
factor with block subspace iteration (Julia, jl/eig_chol.jl) for large ones.  Up to quadrature errors the lowest
eigenvalue E_h is an upper bound on the lowest level of the box; it does not bound the error of the tail or of the moments.

Elements and quadrature are set through the environment (fem3d): FEM_ORDER_AB, FEM_ORDER_C, FEM_NGAUSS, and FEM_NGAUSS_AB,
FEM_NGAUSS_C for the Gauss points across and along the valley.  Far along the valley the envelope is much narrower than a
transverse element, and six Gauss points per element do not integrate the products of quartic polynomials with it (matrix
elements at r1 = 16 off by up to 16%; 1.6% for cubic ones); the quartic states use sixteen points in a and b (twelve give
the same state to 1e-8).  A state computed with a non-default quadrature must be post-processed with the same
FEM_NGAUSS_AB and FEM_NGAUSS_C (tail_fit.py, observables.py, virial.py), since the norm of the saved coefficients is that
of its mass matrix; a saved state records its Gauss orders, and those scripts refuse to read it with others.

The best state (10 x 10 mesh across the valley with quartic elements in a and b, elements of at most 0.25
along the valley, box c <= 22, pruning tolerance 1e-16, five channels per node, 443294 unknowns; E_h = 0.00031, next
levels 0.1742, 0.3345, 0.5322):
    FEM_ORDER_AB=4 FEM_NGAUSS_AB=16 python solve_ch.py 10 10 0 3.0 3.0 22.0 4 --h0 0.25 --hmaxc 0.25 --ptol 1e-16 \
        --enrich 0 --nev 4 --seed solve_ch_Q4L22G16_r0.pkl --out _Q4L22G16V25
(four hours on 80 cores, about 200 GB).  data/README.md gives the commands of its seed and of the other states.

Output: solve_ch<tag>.pkl (lowest eigenvalues ev, node fermion vectors Phi of the lowest state, the element grid and
orders, the Gauss orders, the number of channels, the residuals of the eigensolver and the checksum of the operators file)
and solve_ch<tag>_r<n>.pkl for every enrichment round.
Usage: python solve_ch.py na nb nc amax bmax cmax nch [--nev 4] [--out tag] [--outdir .] [--h0 0.08] [--h0c 0.15] [--qc 1.4]
       [--hmaxc H] [--enrich N] [--no-round-states] [--nproc 16] [--ops fspace_ops.pkl] [--ptol 1e-12] [--seed FILE ...]
Environment: BFSS_FSPACE_OPS (default for --ops), EIGSOLVER (julia or superlu, default by size), JULIA (julia executable),
FEM_ORDER_AB, FEM_ORDER_C, FEM_NGAUSS (element orders and Gauss points of fem3d), JL_MAXIT and JL_TOL1 (iteration limit and
residual tolerance of jl/eig_chol.jl, defaults 80 and 1e-10), JL_ACCEPT (default 1e-8): if the iteration limit is reached,
the run stops with an error unless the relative residual of the lowest state is below it."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, pickle, sys, time, os, hashlib
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
EIG = {}                                                       # diagnostics of the last eigensolve, saved with the state
import fem3d
from fem3d import Grid, geometric, env_valley, prune_scale
_G = {}
def nd_order(G, keep, leaf=64):
    """geometric nested dissection of the spatial node grid (separators: planes of element-vertex nodes), returned as an
    ordering of the pruned node indices 0..len(keep)-1"""
    na, nb, nc = G.shape; qs = getattr(G, "qs", (G.order,) * 3)
    gi = np.nonzero(G.free)[0][keep]; ia, ib, ic = gi // (nb * nc), (gi // nc) % nb, gi % nc
    pos = {}
    for n_, (x, y, z) in enumerate(zip(ia, ib, ic)): pos[(x, y, z)] = n_
    out = []
    def rec(lo, hi):
        ext = [hi[d] - lo[d] for d in range(3)]
        cnt = np.prod([e + 1 for e in ext])
        if cnt <= leaf or max(ext) < 2 * max(qs):
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    for z in range(lo[2], hi[2] + 1):
                        if (x, y, z) in pos: out.append(pos[(x, y, z)])
            return
        d = int(np.argmax(ext)); q = qs[d]; mid = (lo[d] + hi[d]) // 2; mid = max(lo[d] + q, min(hi[d] - q, q * round(mid / q)))
        l1 = list(lo); h1 = list(hi); h1[d] = mid - 1; l2 = list(lo); l2[d] = mid + 1; ls = list(lo); hs = list(hi); ls[d] = hs[d] = mid
        rec(l1, h1); rec(l2, hi); rec(ls, hs)
    rec([0, 0, 0], [na - 1, nb - 1, nc - 1])
    out = np.array(out); assert len(out) == len(keep) and len(set(out.tolist())) == len(keep)
    return out

def _enrich_node(al):
    """Davidson correction for node al: z = (B + shift)^-1 r_perp with B the full-fermion-space diagonal block of H - E at the node
    (shifted to stay >= 1; its lowest eigenvalue is known from the local channel solve), r_perp the residual with the current channels
    projected out.  Returns two orthonormal new directions (z, r_perp), flags telling which are usable, and |r_perp|."""
    Mt = _G["Mt"]; U = _G["Uenr"][al]; r = _G["Renr"][al]; E = _G["Eenr"]
    ok_u = _G["valid"][al]; Uv = U[:, ok_u]                    # only the valid channels span the current space
    Z = _G.get("Zw", {}).get(_G["wall"][al]) if "wall" in _G else None
    proj = (lambda v: Z @ (Z.conj().T @ v)) if Z is not None else (lambda v: v)
    orth = lambda v: (lambda w: w - Uv @ (Uv.conj().T @ w))(v - Uv @ (Uv.conj().T @ v))
    D = U.shape[0]; out = np.zeros((D, 2), complex); flags = np.zeros(2, bool)
    rp = orth(proj(r)); nr = np.linalg.norm(rp)
    def clean(v, others):
        if not np.all(np.isfinite(v)): return None
        for o in others: v = v - o * np.vdot(o, v)
        v = orth(proj(v)); n_ = np.linalg.norm(v)
        return v / n_ if (np.isfinite(n_) and n_ > 1e-6 * max(1.0, nr)) else None
    if not np.isfinite(nr) or nr < 1e-12: return al, out, flags, nr
    B = sum(cf * M for cf, M in zip(_G["coefs"][al], Mt)) + (_G["H0diag"][al] - E) * sp.identity(D)
    B = ((B + B.conj().T) / 2).tocsc()
    if Z is not None: B = sp.csc_matrix(Z.conj().T @ (B @ Z)); rz = Z.conj().T @ rp
    else: rz = rp
    lmin = _G["Aloc0"][al] + _G["H0diag"][al] - E               # lowest eigenvalue of B (from the local channel solve)
    shift = max(0.0, 1.0 - lmin)
    try:
        z = spla.spsolve((B + shift * sp.identity(B.shape[0])).tocsc(), rz)
        if Z is not None: z = Z @ z
    except Exception:
        z = rp
    v1 = clean(z, [])
    if v1 is None: v1 = clean(rp, [])
    if v1 is not None: out[:, 0] = v1; flags[0] = True
    v2 = clean(rp, [v1] if v1 is not None else [])
    if v2 is not None and flags[0]: out[:, 1] = v2; flags[1] = True
    return al, out, flags, nr

def _chan(args):
    alpha, coef = args; Mt = _G["Mt"]; nch = _G["nch"]
    A = sum(cf * M for cf, M in zip(coef, Mt)); A = ((A + A.conj().T) / 2).tocsr()
    Z = _G.get("Zw", {}).get(_G["wall"][alpha]) if "wall" in _G else None
    if Z is not None:                      # node on a coincidence wall: channels in ker K_S^2 of that wall (exact wall behaviour)
        Ar = Z.conj().T @ (A @ Z); ev, V = np.linalg.eigh((Ar + Ar.conj().T) / 2); return alpha, ev[:nch], Z @ V[:, :nch]
    try:
        ev, V = spla.eigsh(A, k=nch, which="SA", tol=1e-10, maxiter=20000)
        o = np.argsort(ev); return alpha, ev[o], V[:, o]
    except Exception:
        ev, V = np.linalg.eigh(A.toarray()); return alpha, ev[:nch], V[:, :nch]

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 7 or args[0] in ("-h", "--help"):
        print(__doc__.strip()); sys.exit(0 if args and args[0] in ("-h", "--help") else 2)
    def opt(name, default, cast):
        if name in args:
            i = args.index(name); v = cast(args[i + 1]); del args[i:i + 2]; return v
        return default
    nev = opt("--nev", 4, int); tag = opt("--out", "", str); h0 = opt("--h0", 0.08, float); nproc = opt("--nproc", 16, int)
    outdir = opt("--outdir", ".", str); h0c = opt("--h0c", 0.15, float); qc = opt("--qc", 1.4, float); hmaxc = opt("--hmaxc", None, float)
    rounds = opt("--enrich", 0, int); ops_file = opt("--ops", os.environ.get("BFSS_FSPACE_OPS", "fspace_ops.pkl"), str)
    save_rounds = "--no-round-states" not in args
    os.makedirs(outdir, exist_ok=True)                         # before hours of work, not after them
    seeds = []
    while "--seed" in args:
        i_ = args.index("--seed"); seeds.append(args[i_ + 1]); del args[i_:i_ + 2]
    ptol = opt("--ptol", 1e-12, float)
    if not save_rounds: args.remove("--no-round-states")
    na, nb, nc = map(int, args[:3]); amax, bmax, cmax = map(float, args[3:6]); nch = int(args[6])
    t0 = time.time()
    for f_ in [ops_file] + seeds:
        if not os.path.exists(f_): raise SystemExit(f"{f_} not found (operators: --ops or BFSS_FSPACE_OPS; seeds: --seed)")
    ops = pickle.load(open(ops_file, "rb")); D = ops["dim"]
    ops_sha = hashlib.sha256(open(ops_file, "rb").read()).hexdigest()
    keys = [("Y", i) for i in range(3)] + [(k, p) for p in [(0, 1), (0, 2), (1, 2)] for k in ("KS2", "KD2")] + [("O", j) for j in range(3)]
    Mt = [ops["Y"][k[1]] if k[0] == "Y" else (ops["O"][k[1]] if k[0] == "O" else ops[k[0]][k[1]]) for k in keys]
    Mt = [sp.csr_matrix(m) for m in Mt]
    a = geometric(na, amax, h0); b = geometric(nb, bmax, h0)
    c = geometric(nc, cmax, h0c) if hmaxc is None else fem3d.graded(cmax, h0c, qc, hmaxc)
    nc = len(c) - 1
    G = Grid(a, b, c); K, M, P = G.assemble(env=env_valley(1.0)); d_unscaled = M.diagonal().real.copy()
    keep, K, M, P = prune_scale(K, M, P, tol=ptol); s_new = 1 / np.sqrt(d_unscaled[keep])
    _gi = np.nonzero(G.free)[0][keep]; _xa = G.a[_gi // (G.shape[1] * G.shape[2])]; _xb = G.b[(_gi // G.shape[2]) % G.shape[1]]; _xc = G.c[_gi % G.shape[2]]
    print(f"pruning tol {ptol:g}: kept {len(keep)} of {G.N} nodes; kept-node extents a <= {_xa.max():.3f}, b <= {_xb.max():.3f}, a + b <= {(_xa + _xb).max():.3f}, c <= {_xc.max():.3f}; "
          f"a-grid {np.round(np.diff(a), 3).tolist()}", flush=True)
    Ns = K.shape[0]; H0 = (K + P["Vbos"]).tocsr(); Pt = [P[k].tocsr() for k in keys]
    print(f"grid {na}x{nb}x{nc} box ({amax},{bmax},{cmax}): {Ns} spatial functions ({time.time()-t0:.0f}s)", flush=True)
    # local channel bases from the node-averaged operator (diagonal of the scaled potential matrices; M has unit diagonal)
    coefs = [[Pt[t][al, al] for t in range(len(keys))] for al in range(Ns)]
    vb = np.array([P["Vbos"][al, al] for al in range(Ns)])
    _G.update(Mt=Mt, nch=nch)
    # coincidence walls b = 0 (r2 = r3) and c = 0 (r1 = r2): J ~ distance, K_S^2 / (2 distance^2) makes the Galerkin integrals of
    # wall-node functions log-divergent unless their fermion part lies in ker K_S^2 (the exact solution vanishes there like
    # distance^{|kappa|/2}, kappa = 2, 4, ...).  Restrict the wall-node channels to that kernel.
    ker = lambda M: (lambda w, V: V[:, w < 1e-8])(*np.linalg.eigh(((M + M.conj().T) / 2).toarray()))
    K01, K12 = ops["KS2"][(0, 1)], ops["KS2"][(1, 2)]
    Zw = {"c": ker(K01), "b": ker(K12), "bc": ker(K01 + K12)}
    naG, nbG, ncG = G.shape; gi = np.nonzero(G.free)[0][keep]; ib = (gi // ncG) % nbG; ic = gi % ncG
    wall = [("b" if ib[al] == 0 else "") + ("c" if ic[al] == 0 else "") or None for al in range(Ns)]
    _G.update(Zw=Zw, wall=wall)
    print(f"wall-kernel channels: ker dims b {Zw['b'].shape[1]}, c {Zw['c'].shape[1]}, bc {Zw['bc'].shape[1]}; "
          f"wall nodes b {sum(w == 'b' for w in wall)}, c {sum(w == 'c' for w in wall)}, bc {sum(w == 'bc' for w in wall)}", flush=True)
    with Pool(nproc) as pool: res = pool.map(_chan, list(enumerate(coefs)), chunksize=8)
    U = [None] * Ns; E_loc = np.zeros((Ns, nch))
    for al, ev, V in res: U[al] = V; E_loc[al] = ev + vb[al]
    print(f"local channels ({nch} per node) done ({time.time()-t0:.0f}s); lowest local level range {E_loc[:, 0].min():.2f} .. {E_loc[:, 0].max():.2f}", flush=True)
    # ---- Galerkin solve for a given channel set (Ublk: (Ns, D, nch_cur); the first nch_base are local eigenvectors) ----
    H0 = H0.tocsc(); Mcsc = M.tocsc(); Ptc = [P_.tocsc() for P_ in Pt]
    def galerkin(Ublk):
        nchc = Ublk.shape[2]; base = np.arange(nchc)
        # (node, channel) pairs whose enrichment vector was unusable get zero coefficients
        act2 = valid.copy(); active = act2.ravel(); newidx = np.cumsum(active) - 1; n_act = int(active.sum())
        if n_act < Ns * nchc: print(f"{n_act} of {Ns * nchc} (node, channel) unknowns active", flush=True)
        def col_blocks(be):
            als = np.unique(np.concatenate([H0.indices[H0.indptr[be]:H0.indptr[be + 1]], Mcsc.indices[Mcsc.indptr[be]:Mcsc.indptr[be + 1]]]
                                           + [P_.indices[P_.indptr[be]:P_.indptr[be + 1]] for P_ in Ptc]))
            als = als[als <= be]                               # upper triangle by node (the matrices are Hermitian)
            jb = np.nonzero(act2[be])[0]; Ub = Ublk[be][:, jb]  # active channels of the column node
            Ua = Ublk[als].conj()                              # (m, D, nchc)
            ov = np.einsum("mdi,dj->mij", Ua, Ub, optimize=True)
            colv = lambda A_: A_[:, be].toarray().ravel()[als]
            h = colv(H0)[:, None, None] * ov; m_ = colv(Mcsc)[:, None, None] * ov
            for t in range(len(keys)):
                pv = colv(Ptc[t]); nz = pv != 0
                if nz.any(): h[nz] += pv[nz][:, None, None] * np.einsum("mdi,dj->mij", Ua[nz], Mt[t] @ Ub, optimize=True)
            r = np.repeat(als[:, None] * nchc + base[None, :], len(jb), axis=1).ravel(); cidx = np.tile(be * nchc + jb, len(als) * nchc)
            up = (r <= cidx) & active[r] & active[cidx]        # scalar upper triangle (drops the lower half of the diagonal block)
            return newidx[r[up]].astype(np.int32), newidx[cidx[up]].astype(np.int32), h.ravel()[up], m_.ravel()[up], len(als)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, nproc)) as ex: out = list(ex.map(col_blocks, range(Ns)))
        rows = [o[0] for o in out]; cols = [o[1] for o in out]; hv = [o[2] for o in out]; mv = [o[3] for o in out]; npairs = sum(o[4] for o in out); del out
        print(f"blocks assembled for {npairs} spatial pairs ({time.time()-t0:.0f}s)", flush=True)
        n = n_act
        RR = np.concatenate(rows); CC = np.concatenate(cols); del rows, cols
        Hu = sp.csc_matrix((np.concatenate(hv), (RR, CC)), shape=(n, n)); del hv
        Mu = sp.csc_matrix((np.concatenate(mv), (RR, CC)), shape=(n, n)); del mv, RR, CC
        Hu.setdiag(Hu.diagonal().real); Mu.setdiag(Mu.diagonal().real)
        full_ = lambda U_: (U_ + U_.conj().T - sp.diags(U_.diagonal())).tocsr()
        print(f"channel Galerkin matrices (upper triangles): {n} unknowns, nnz {Hu.nnz} ({time.time()-t0:.0f}s)", flush=True)
        sig = -1.0
        if os.environ.get("EIGSOLVER", "julia" if (Hu.nnz > 2e7 or n_act < Ns * nchc) else "superlu") == "julia":
            # large problems: CHOLMOD (Julia stdlib) Cholesky of H - sigma M and block subspace iteration (eig_chol.jl); the
            # SuperLU factorisation runs out of memory beyond ~5e7 nonzeros
            import subprocess, tempfile, shutil
            tmp = tempfile.mkdtemp(prefix="bfss_ch_")
            Hu.sort_indices(); Mu.sort_indices()
            np.savez(os.path.join(tmp, "in.npz"), n=n, nev=nev, sigma=sig, upper=1,
                     h_indptr=Hu.indptr.astype(np.int64), h_indices=Hu.indices.astype(np.int64), hre=Hu.data.real, him=Hu.data.imag,
                     m_indptr=Mu.indptr.astype(np.int64), m_indices=Mu.indices.astype(np.int64), mre=Mu.data.real, mim=Mu.data.imag)
            del Hu, Mu
            jl = os.environ.get("JULIA", "julia")
            r_ = subprocess.run([jl, f"--project={os.path.join(HERE, 'jl')}", os.path.join(HERE, "jl", "eig_chol.jl"),
                                 os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")], capture_output=True, text=True)
            print("\n".join(l for l in r_.stdout.splitlines() if "CHOLMOD" in l or l.startswith("converged")), flush=True)
            if r_.returncode != 0: print(r_.stderr[-3000:], flush=True); raise RuntimeError("eig_chol.jl failed")
            o_ = np.load(os.path.join(tmp, "out.npz")); ev = o_["ev"]; X = o_["xre"] + 1j * o_["xim"]
            # relative residuals of the eigenpairs: the run is accepted if eig_chol.jl converged (JL_TOL1 for the lowest state,
            # 1e-6 for the others) or, at its iteration limit, if the lowest state is below JL_ACCEPT
            res = np.atleast_1d(o_["res"]).astype(float); tol1 = float(os.environ.get("JL_TOL1", "1e-10")); acc = float(os.environ.get("JL_ACCEPT", "1e-8"))
            conv = bool(res[0] < tol1 and res.max() < 1e-6)
            EIG.clear(); EIG.update(solver="julia", residuals=[float(x) for x in res], converged=conv, tol=tol1, accept=acc, maxit=int(os.environ.get("JL_MAXIT", "80")))
            if not conv:
                if not (res[0] < acc and res.max() < 1e-6):
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise RuntimeError(f"eig_chol.jl did not converge: relative residuals {res} (accepted below JL_ACCEPT = {acc:g} for the lowest state and 1e-6 for the others)")
                print(f"eigensolver at its iteration limit: relative residuals {res}, accepted (JL_ACCEPT = {acc:g})", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            Hc = full_(Hu); Mcm = full_(Mu); del Hu, Mu
            OP = (Hc - sig * Mcm).tocsc()
            if n_act == Ns * nchc:                   # node-level nested dissection, expanded to the channel blocks
                pn = nd_order(G, keep); perm = (pn[:, None] * nchc + base[None, :]).ravel(); iperm = np.empty_like(perm); iperm[perm] = np.arange(n)
                lu = spla.splu(OP[perm][:, perm].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
                solve_ = lambda v: lu.solve(v[perm])[iperm]
            else:
                lu = spla.splu(OP, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)); solve_ = lu.solve
            print(f"LU: nnz(L) + nnz(U) = {lu.L.nnz + lu.U.nnz:.3e} ({(lu.L.nnz + lu.U.nnz) * 16 / 2**30:.1f} GB) ({time.time()-t0:.0f}s)", flush=True)
            OPinv = spla.LinearOperator(OP.shape, matvec=solve_, dtype=complex)
            ev, X = spla.eigsh(Hc, k=nev, M=Mcm, sigma=sig, which="LM", OPinv=OPinv)      # ARPACK raises if it does not converge
            EIG.clear(); EIG.update(solver="superlu", residuals=None, converged=True, tol=0.0, accept=None, maxit=None)
        o = np.argsort(ev); ev = ev[o]; X = X[:, o]
        if n_act < Ns * nchc:                                  # back to the full (node, channel) layout
            Xf = np.zeros((Ns * nchc, X.shape[1]), complex); Xf[active] = X; X = Xf
        return ev, X
    # ---- residual-driven channel enrichment (Davidson-type): at every node add the block-preconditioned residual of the
    # current lowest state, (H_aa - E)^-1 (1 - U_a U_a^+) R_a, and the projected residual itself, then solve again ----
    def enrich_vectors(Ublk, Psi, E):
        R = (H0 - E * M) @ Psi                                 # (Ns, D): residual of the full (spatial x fermion) problem
        for t in range(len(keys)): R = R + (Mt[t] @ (Pt[t] @ Psi).T).T
        _G.update(Uenr=Ublk, Renr=R, Eenr=E, H0diag=H0.diagonal().real, coefs=coefs, Aloc0=E_loc[:, 0] - vb, valid=valid)
        with Pool(nproc) as pool: res_ = pool.map(_enrich_node, range(Ns), chunksize=16)
        Z = np.zeros((Ns, Ublk.shape[1], 2), complex); rn = np.zeros(Ns); fl = np.zeros((Ns, 2), bool)
        for al, z, f_, nr in res_: Z[al] = z; rn[al] = nr; fl[al] = f_
        print(f"enrichment: residual norm {np.sqrt((rn ** 2).sum()):.3e} (largest node {rn.max():.3e}); "
              f"{int((~fl).sum())} of {2 * Ns} new directions unusable", flush=True)
        return Z, fl
    Ublk = np.stack(U)                                         # (Ns, D, nch)
    valid = np.ones((Ns, nch), bool)                           # usable channels (enrichment can fail at a node)
    if seeds:
        from fem3d import lagrange1d
        seedvecs = []
        def loc(x, e):
            k_ = np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2); return k_, (x - e[k_]) / np.diff(e)[k_], x <= e[-1] + 1e-12
        cache_ = {}
        for sf in seeds:
            d0 = pickle.load(open(sf, "rb")); a0, b0, c0 = d0["grid"]; keep0 = d0["keep"]
            # a seed is read in the basis it was written in: the same operators file, and the Gauss orders of its own run
            # (they fix the norm of its coefficients); a seed that records neither is read with those of this run
            if d0.get("ops_sha256") and d0["ops_sha256"] != ops_sha:
                raise SystemExit(f"seed {sf} was computed with another operators file (sha256 {d0['ops_sha256'][:12]}..., this run "
                                 f"{ops_sha[:12]}...): its fermion vectors are in another basis")
            q0f = d0.get("fem") or fem3d.quadrature()
            if d0.get("fem") and q0f != fem3d.quadrature(): print(f"seed {os.path.basename(sf)}: read with its own Gauss orders ({q0f['ngauss_ab']}, {q0f['ngauss_c']}); this run uses ({fem3d.NGAUSS_AB}, {fem3d.NGAUSS_C})", flush=True)
            if not d0.get("fem"): print(f"seed {os.path.basename(sf)} records neither its Gauss orders nor its operators file: read with those of this run", flush=True)
            key_ = (np.asarray(a0).tobytes(), np.asarray(b0).tobytes(), np.asarray(c0).tobytes(), tuple(d0.get("qs") or ()), q0f["ngauss_ab"], q0f["ngauss_c"])
            if key_ not in cache_:
                with fem3d.gauss(q0f["ngauss_ab"], q0f["ngauss_c"]):
                    G0 = Grid(a0, b0, c0, qs=d0.get("qs")); _, M0, _ = G0.assemble(terms={"Vbos"}, env=env_valley(1.0))
                cache_[key_] = (G0, M0.diagonal().real.copy()); del M0
            G0, d0diag = cache_[key_]; s0 = 1 / np.sqrt(d0diag[keep0])
            full0 = np.zeros((G0.N + 1, D), complex); full0[keep0] = d0["Phi"].astype(complex) * s0[:, None]
            na0, nb0, nc0 = G0.shape; gidx0 = np.full(na0 * nb0 * nc0, G0.N); gidx0[G0.free] = np.arange(G0.N)
            q0 = G0.qs; L0 = [lagrange1d(q_) for q_ in q0]
            ka, ua, ina = loc(_xa, G0.ea); kb, ub, inb = loc(_xb, G0.eb); kc, uc, inc = loc(_xc, G0.ec); inside = ina & inb & inc
            val = np.zeros((Ns, D), complex)
            for p_ in range(q0[0] + 1):
                wa = L0[0][p_][0](ua)
                for q_ in range(q0[1] + 1):
                    wb = L0[1][q_][0](ub)
                    for t_ in range(q0[2] + 1):
                        wc = L0[2][t_][0](uc)
                        ids = gidx0[((q0[0] * ka + p_) * nb0 + (q0[1] * kb + q_)) * nc0 + (q0[2] * kc + t_)]
                        val += (wa * wb * wc * inside)[:, None] * full0[ids]
            del full0
            sv = val / s_new[:, None]; del val                     # coefficients in the scaled basis of the new grid
            HPsi = H0 @ sv
            for t in range(len(keys)): HPsi = HPsi + (Mt[t] @ (Pt[t] @ sv).T).T
            print(f"seed {os.path.basename(sf)} (E_h there {d0['ev'][0]:.5f}): Rayleigh quotient of its interpolant on this grid "
                  f"{np.real(np.vdot(sv, HPsi)) / np.real(np.vdot(sv, M @ sv)):.5f}; {int((~inside).sum())} nodes outside its box ({time.time()-t0:.0f}s)", flush=True)
            del HPsi; seedvecs.append(sv)
        nseed = len(seedvecs); Unew = np.zeros((Ns, D, nseed + nch), complex); valid = np.zeros((Ns, nseed + nch), bool)
        for al in range(Ns):
            Zk = Zw.get(wall[al]) if wall[al] else None
            cols = [(Zk @ (Zk.conj().T @ sv[al])) if Zk is not None else sv[al] for sv in seedvecs] + [Ublk[al][:, j] for j in range(nch)]
            for j, v in enumerate(cols):
                v = v.copy(); n0 = np.linalg.norm(v)
                if not (n0 > 0 and np.isfinite(n0)): continue
                for _ in range(2):
                    for i in range(j):
                        if valid[al, i]: v -= Unew[al, :, i] * np.vdot(Unew[al, :, i], v)
                n1 = np.linalg.norm(v)
                if np.isfinite(n1) and n1 > 1e-6 * n0: Unew[al, :, j] = v / n1; valid[al, j] = True
        print(f"seed channels: {nseed} seeds + {nch} local; {int((~valid[:, :nseed]).sum())} seed and {int((~valid[:, nseed:]).sum())} local directions unusable ({time.time()-t0:.0f}s)", flush=True)
        Ublk = Unew; del Unew, seedvecs
    for rnd in range(rounds + 1):
        ev, X = galerkin(Ublk); nchc = Ublk.shape[2]
        print(f"### round {rnd}: {nchc} channels per node: lowest eigenvalues {np.round(ev, 6)} ({time.time()-t0:.0f}s)", flush=True)
        if save_rounds:
            Phi_ = np.einsum("adc,ac->ad", Ublk, X[:, 0].reshape(Ns, nchc)).astype(np.complex64)
            fn_ = os.path.join(outdir, f"solve_ch{tag}_r{rnd}.pkl")
            pickle.dump(dict(ev=ev, Phi=Phi_, keep=keep, grid=(a, b, c), nch=nchc, qs=getattr(G, "qs", None), shape=G.shape, N=G.N,
                             round=rnd, ptol=ptol, seeds=seeds, fem=fem3d.quadrature(), eig=dict(EIG), ops_sha256=ops_sha), open(fn_ + ".tmp", "wb"), protocol=4); os.replace(fn_ + ".tmp", fn_); del Phi_
        if rnd == rounds: break
        Psi = np.einsum("adc,ac->ad", Ublk, X[:, 0].reshape(Ns, nchc)); Psi /= np.sqrt(np.real(np.vdot(Psi, M @ Psi)))
        Znew, fl = enrich_vectors(Ublk, Psi, ev[0])
        Ublk = np.concatenate([Ublk, Znew], axis=2); valid = np.concatenate([valid, fl], axis=1)
    nch_final = Ublk.shape[2]
    print(f"### grid {na}x{nb}x{nc} box ({amax},{bmax},{cmax}), nch {nch_final}: lowest eigenvalues {np.round(ev, 5)} ({time.time()-t0:.0f}s)", flush=True)
    # compact output: the node fermion vectors Phi[alpha] = U_alpha x_alpha of the lowest state (single precision), not the channels
    Phi = np.einsum("adc,ac->ad", Ublk, X[:, 0].reshape(Ns, nch_final)).astype(np.complex64)
    pickle.dump(dict(ev=ev, Phi=Phi, keep=keep, grid=(a, b, c), nch=nch_final, qs=getattr(G, "qs", None), shape=G.shape, N=G.N,
                     round=rounds, ptol=ptol, seeds=seeds, fem=fem3d.quadrature(), eig=dict(EIG), ops_sha256=ops_sha),
                open(os.path.join(outdir, f"solve_ch{tag}.pkl"), "wb"), protocol=4)
    try:
        # virial check: for the exact zero-energy state <T> = <V_bos> and <V_Yukawa> = -2 <V_bos> (T = radial + angular kinetic).
        # Psi[alpha] = U_alpha x_alpha is the state in the (spatial node) x (fermion space) basis; Hc is the restriction of
        # (K + V_bos) x 1 + sum_t P_t x M_t, so the expectations are those of the full operators in Psi.
        Psi = np.einsum("adc,ac->ad", Ublk, X[:, 0].reshape(Ns, nch_final))
        def expv(S, Mf=None):
            phi = S @ Psi
            if Mf is not None: phi = (Mf @ phi.T).T
            return np.real(np.vdot(Psi, phi))
        nrm = expv(M); ex = dict(kin=expv(K) / nrm, Vbos=expv(P["Vbos"]) / nrm,
                                 ang=sum(expv(Pt[t], Mt[t]) for t in range(len(keys)) if keys[t][0] != "Y") / nrm,
                                 yuk=sum(expv(Pt[t], Mt[t]) for t in range(len(keys)) if keys[t][0] == "Y") / nrm)
        T_ = ex["kin"] + ex["ang"]
        print(f"    virial: <T> = {T_:.4f} (radial {ex['kin']:.4f} + angular {ex['ang']:.4f}), <V_bos> = {ex['Vbos']:.4f}, <V_Y> = {ex['yuk']:.4f}, "
              f"sum {T_ + ex['Vbos'] + ex['yuk']:.4f}; exact state: T/V_bos = 1 (here {T_/ex['Vbos']:.4f}), V_Y/V_bos = -2 (here {ex['yuk']/ex['Vbos']:.4f})", flush=True)
    except Exception as e_:
        print(f"    virial check failed: {e_!r}", flush=True)
