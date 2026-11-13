"""Build the reduced fermion space of the SU(2) singlet problem and the fermionic operators of the reduced Hamiltonian.

The 24 complex fermion modes carry 7898 states invariant under the Spin(6) stabiliser
(skew Howe duality), and the Z2 x Z2 stabiliser reduces them to 2066, the components of the singlet
wavefunction.  Output (pickle, about 20 MB): an orthonormal basis of the Spin(6) x Z2^2 - invariant fermion
states and on it
    Y[i]      (i = 0,1,2)   Yukawa bilinears, H_ferm(Sigma) = sum_i r_i Y_i
    KS2[ij], KD2[ij]         squares of the generators of the S_ij and D_ij directions
    O[j]      (j = 0,1,2)   sum_{a=3..8} K_{O,aj}^2
    N4                       fermion layer k (number of fermions / 4)
The finite-element solver (ground_state/solve_ch.py) reads this file.
Usage: python fspace.py [--out fspace_ops.pkl] [--nproc 8]   (10 minutes; about 40 GB peak with 8 processes)"""
import numpy as np, itertools, pickle, time, sys, os, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]
import scipy.sparse as sp, scipy.linalg as sla
from fermions import single_particle, quad_to_c, A_yukawa, A_KS, A_KD, A_KO, NM

def popcount(x):
    x = np.asarray(x, np.int64).copy(); c = np.zeros_like(x)
    while np.any(x): c += x & 1; x >>= 1
    return c
def sgn_below(m, k): return np.where(popcount(m & ((1 << k) - 1)) % 2, -1.0, 1.0)
def op_c(m, l):        # returns (keep mask, new m, sign)
    keep = ((m >> l) & 1) == 1; mm = m[keep]; return keep, mm & ~(1 << l), sgn_below(mm, l)
def op_cd(m, k):
    keep = ((m >> k) & 1) == 0; mm = m[keep]; return keep, mm | (1 << k), sgn_below(mm, k)
def op_cdc(m, k, l):   # c^+_k c_l
    k1, m1, s1 = op_c(m, l); k2, m2, s2 = op_cd(m1, k)
    idx = np.nonzero(k1)[0][k2]; return idx, m2, s1[k2] * s2
def op_cc(m, k, l):    # c_k c_l
    k1, m1, s1 = op_c(m, l); k2, m2, s2 = op_c(m1, k)
    idx = np.nonzero(k1)[0][k2]; return idx, m2, s1[k2] * s2
def op_cdcd(m, k, l):  # c^+_k c^+_l
    k1, m1, s1 = op_cd(m, l); k2, m2, s2 = op_cd(m1, k)
    idx = np.nonzero(k1)[0][k2]; return idx, m2, s1[k2] * s2

def weight_zero_states(k):
    subsets = list(itertools.combinations(range(6), k)); out = []
    bits = [[sum(1 << (4 * a + u) for a in S) for S in subsets] for u in range(4)]
    for combo in itertools.product(*bits): out.append(combo[0] | combo[1] | combo[2] | combo[3])
    return np.array(sorted(out), np.int64)
def gl6_weight(m):
    return np.stack([sum(((m >> (4 * a + u)) & 1) for u in range(4)) for a in range(6)], axis=1)

def invariant_blocks(k):
    m = weight_zero_states(k); n = gl6_weight(m)
    key = n @ (5 ** np.arange(6)); order = np.argsort(key, kind="stable"); m, n, key = m[order], n[order], key[order]
    starts = np.nonzero(np.r_[True, key[1:] != key[:-1]])[0]; ends = np.r_[starts[1:], len(m)]
    blocks = []
    for s0, e0 in zip(starts, ends):
        ms = m[s0:e0]; nb = len(ms); mats = []
        for u in range(3):
            rows_m, cols, vals = [], [], []
            for a in range(6):
                idx, m2, sg = op_cdc(ms, 4 * a + u, 4 * a + u + 1)
                rows_m.append(m2); cols.append(idx); vals.append(sg)
            rm = np.concatenate(rows_m); cc = np.concatenate(cols); vv = np.concatenate(vals)
            if len(rm) == 0: continue
            tgt, inv = np.unique(rm, return_inverse=True)
            M = np.zeros((len(tgt), nb)); np.add.at(M, (inv, cc), vv); mats.append(M)
        if mats:
            Mall = np.vstack(mats); U_, S, Vt = np.linalg.svd(Mall, full_matrices=True)
            rk = int((S > 1e-10 * max(S[0], 1.0)).sum()); V = Vt[rk:].T
        else: V = np.eye(nb)
        if V.shape[1]: blocks.append(dict(k=k, n=tuple(n[s0]), masks=ms, V=V))
    return blocks

class InvSpace:
    """invariant states of all layers; column index = global invariant index"""
    def __init__(self, blocks):
        self.blocks = blocks; off = 0
        for b in blocks: b["off"] = off; off += b["V"].shape[1]
        self.dim = off
        allm = np.concatenate([b["masks"] for b in blocks]); self.order = np.argsort(allm); self.sorted_masks = allm[self.order]
        self.row_block = np.concatenate([np.full(len(b["masks"]), i) for i, b in enumerate(blocks)])[self.order]
        self.row_pos = np.concatenate([np.arange(len(b["masks"])) for b in blocks])[self.order]
    def locate(self, m):
        p = np.searchsorted(self.sorted_masks, m); p = np.minimum(p, len(self.sorted_masks) - 1)
        ok = self.sorted_masks[p] == m; return ok, self.row_block[p], self.row_pos[p]
    def number_conserving(self, H):
        """matrix of sum_kl H_kl c^+_k c_l on the invariant space (H must preserve SU(4) invariance)"""
        out = np.zeros((self.dim, self.dim), complex)
        nzk, nzl = np.nonzero(np.abs(H) > 1e-14)
        for bi, b in enumerate(self.blocks):
            ms = b["masks"]
            for kk, ll in zip(nzk, nzl):
                idx, m2, sg = op_cdc(ms, kk, ll)
                if len(idx) == 0: continue
                ok, tb, tp = self.locate(m2)
                if not ok.all(): raise ValueError("operator leaves the weight-zero space")
                for t in np.unique(tb):
                    sel = tb == t; B = self.blocks[t]
                    S = np.zeros((len(B["masks"]), len(ms)), complex); np.add.at(S, (tp[sel], idx[sel]), H[kk, ll] * sg[sel])
                    out[B["off"]:B["off"] + B["V"].shape[1], b["off"]:b["off"] + b["V"].shape[1]] += B["V"].T @ S @ b["V"]
        return out
    def pairing_image(self, P, R):
        """sparse matrix X (intermediate Fock states x invariants) of the operator sum P_kl c_k c_l + R_kl c^+_k c^+_l"""
        rows_m, cols, vals = [], [], []
        terms = [(op_cc, P), (op_cdcd, R)]
        for bi, b in enumerate(self.blocks):
            ms = b["masks"]; V = b["V"]
            for fn, Mt in terms:
                nzk, nzl = np.nonzero(np.abs(Mt) > 1e-14)
                img_m, img_i, img_v = [], [], []
                for kk, ll in zip(nzk, nzl):
                    idx, m2, sg = fn(ms, kk, ll)
                    if len(idx): img_m.append(m2); img_i.append(idx); img_v.append(Mt[kk, ll] * sg)
                if not img_m: continue
                im = np.concatenate(img_m); ii = np.concatenate(img_i); iv = np.concatenate(img_v)
                tgt, inv = np.unique(im, return_inverse=True)
                S = sp.csr_matrix((iv, (inv, ii)), shape=(len(tgt), len(ms)))
                X = S @ V                                          # (len(tgt) x n_inv of the block), dense
                r_, c_ = np.nonzero(np.abs(X) > 1e-15)
                rows_m.append(tgt[r_]); cols.append(b["off"] + c_); vals.append(X[r_, c_])
        rm = np.concatenate(rows_m); cc = np.concatenate(cols); vv = np.concatenate(vals)
        tgt, inv = np.unique(rm, return_inverse=True)
        return sp.csr_matrix((vv, (inv, cc)), shape=(len(tgt), self.dim))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="reduced fermion space and operators")
    ap.add_argument("--out", default="fspace_ops.pkl"); ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time(); W, _ = single_particle()
    blocks = []
    for k in range(7):
        bl = invariant_blocks(k); blocks += bl
        print(f"k = {k}: {sum(len(b['masks']) for b in bl)} states, {len(bl)} blocks with invariants, {sum(b['V'].shape[1] for b in bl)} invariants ({time.time()-t0:.0f}s)", flush=True)
    S = InvSpace(blocks); print(f"invariant space: {S.dim} ({time.time()-t0:.0f}s)", flush=True)
    N4 = np.concatenate([np.full(b["V"].shape[1], b["k"]) for b in blocks])
    layers = [np.nonzero(N4 == k)[0] for k in range(7)]
    def nc(A):
        const, H, P, R = quad_to_c(A, W); assert max(abs(P).max(), abs(R).max()) < 1e-12
        M = S.number_conserving(H) + const * np.eye(S.dim)
        off = max(abs(M[np.ix_(layers[k], layers[kk])]).max(initial=0) for k in range(7) for kk in range(7) if k != kk)
        assert off < 1e-10, off
        return [M[np.ix_(layers[k], layers[k])] for k in range(7)]      # per-layer blocks
    Y = [nc(A_yukawa(i)) for i in range(3)]; print(f"Yukawa done ({time.time()-t0:.0f}s)", flush=True)
    pairs = [(0, 1), (0, 2), (1, 2)]
    KS = {p: nc(A_KS(*p)) for p in pairs}; KD = {p: nc(A_KD(*p)) for p in pairs}; print(f"K_S, K_D done ({time.time()-t0:.0f}s)", flush=True)
    # discrete stabiliser Z2^2 = {1, U01, U02, U01 U02}, U_ij = exp(-i pi K_S(ij)).  U01 is a pi rotation about the spin
    # quantisation axis: a scalar phase on each GL(6)-weight block.  U02 flips the spins: it maps block n to block pi(n).
    # Invariant basis: blocks with U01 = +1, symmetrised under U02 (each basis vector lives on at most two blocks).
    cols = []                                                    # list of sparse column vectors (global index, coefficient)
    dims = []
    for k in range(7):
        idx = layers[k]; U01 = sla.expm(-1j * np.pi * KS[(0, 1)][k]); U02 = sla.expm(-1j * np.pi * KS[(0, 2)][k])
        kb = [bi for bi, b in enumerate(blocks) if b["k"] == k]; pos = {}; loc = 0
        for bi in kb: ni = blocks[bi]["V"].shape[1]; pos[bi] = np.arange(loc, loc + ni); loc += ni
        keep = []
        for bi in kb:
            sl = pos[bi]; ph = U01[np.ix_(sl, sl)]
            assert np.allclose(ph, ph[0, 0] * np.eye(len(sl)), atol=1e-9) and abs(U01[sl].sum() - ph.sum()) < 1e-8 * len(sl)
            if abs(ph[0, 0] - 1) < 1e-8: keep.append(bi)
        done = set(); nk = 0
        for bi in keep:
            if bi in done: continue
            sl = pos[bi]; img = U02[:, sl]; tgt = [bj for bj in keep if abs(img[pos[bj]]).max() > 1e-9]
            assert len(tgt) == 1, tgt
            bj = tgt[0]; done.update({bi, bj})
            if bj == bi:
                ev, Q = np.linalg.eigh((U02[np.ix_(sl, sl)] + U02[np.ix_(sl, sl)].conj().T) / 2); Qp = Q[:, ev > 0.5]
                for q in Qp.T: cols.append((idx[sl], q)); nk += 1
            else:
                for m in range(len(sl)):
                    e = np.zeros(len(sl)); e[m] = 1.0
                    v2 = U02[np.ix_(pos[bj], sl)] @ e
                    cols.append((np.r_[idx[sl], idx[pos[bj]]], np.r_[e, v2] / np.sqrt(2))); nk += 1
        dims.append(nk)
    D = len(cols)
    Bsp = sp.csc_matrix((np.concatenate([c for _, c in cols]), (np.concatenate([i for i, _ in cols]), np.concatenate([np.full(len(i), n) for n, (i, _) in enumerate(cols)]))), shape=(S.dim, D))
    print(f"reduced fermion space: {D} ({dims}); basis nnz {Bsp.nnz} ({time.time()-t0:.0f}s)", flush=True)
    # check: orthonormal and invariant
    G = (Bsp.conj().T @ Bsp).toarray(); print(f"basis orthonormality {abs(G - np.eye(D)).max():.1e}", flush=True)
    layer_of_global = N4; posl = np.zeros(S.dim, int)
    for k in range(7): posl[layers[k]] = np.arange(len(layers[k]))
    def red_nc(blk):
        """restrict a layer-block-diagonal operator to the reduced basis (sparse)"""
        full = sp.block_diag([sp.csr_matrix(np.where(abs(blk[k]) > 1e-13, blk[k], 0)) for k in range(7)], format="csr")
        perm = np.concatenate(layers); Pm = sp.csr_matrix((np.ones(S.dim), (perm, np.arange(S.dim))), shape=(S.dim, S.dim))
        full = Pm @ full @ Pm.T                                   # back to the global ordering
        out = (Bsp.conj().T @ full @ Bsp).tocsr(); out.data[abs(out.data) < 1e-13] = 0; out.eliminate_zeros(); return out
    sq = lambda blk: [blk[k] @ blk[k] for k in range(7)]
    ops = dict(dim=D, layer_dims=dims, N4=np.real((Bsp.conj().T @ sp.diags(N4.astype(float)) @ Bsp).diagonal()), Y=[red_nc(y) for y in Y],
               KS2={p: red_nc(sq(KS[p])) for p in pairs}, KD2={p: red_nc(sq(KD[p])) for p in pairs})
    print(f"number-conserving operators restricted ({time.time()-t0:.0f}s); nnz: Y0 {ops['Y'][0].nnz}, KS2_01 {ops['KS2'][(0,1)].nnz}", flush=True)
    # quartic O_j = sum_a K_{O,aj}^2 = sum_a X_a^+ X_a with X_a the pairing image (sparse), restricted to the selected columns
    from multiprocessing import Pool
    def one(args):
        a, j = args; const, H, P, R = quad_to_c(A_KO(a, j), W); assert abs(H).max() < 1e-12
        X = S.pairing_image(P, R) @ Bsp
        return j, (X.conj().T @ X).tocsr() + const * sp.identity(D, format="csr")
    tasks = [(a, j) for j in range(3) for a in range(3, 9)]
    with Pool(min(len(tasks), args.nproc)) as pool: res = pool.map(one, tasks)
    O = [sum(M for jj, M in res if jj == j) for j in range(3)]
    ops["O"] = O; print(f"O_j done ({time.time()-t0:.0f}s); nnz O_0 {O[0].nnz}", flush=True)
    for name, M in [("Y0", ops["Y"][0]), ("KS2_01", ops["KS2"][(0, 1)]), ("KD2_12", ops["KD2"][(1, 2)]), ("O0", O[0])]:
        Md = M.toarray(); print(f"{name}: |M - M^+| = {abs(Md - Md.conj().T).max():.1e}, |M| = {abs(Md).max():.2f}, min eig {np.linalg.eigvalsh((Md + Md.conj().T) / 2)[0]:.4f}", flush=True)
    pickle.dump(ops, open(args.out, "wb")); print(f"saved {args.out} ({time.time()-t0:.0f}s)", flush=True)
    checks = [("Spin(6)-invariant fermion states", S.dim, 7898, "expected"), ("after Z2 x Z2", D, 2066, "expected"),
              ("layer dimensions k = 0..6", dims, [1, 36, 468, 1056, 468, 36, 1], "expected")]
    bad = 0
    for name, got, ref, src in checks:
        ok = got == ref; bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: computed {got}, {src} {ref}")
    sys.exit(1 if bad else 0)
