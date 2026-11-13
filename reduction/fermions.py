"""Fermion space of the reduced SU(2) problem.

At the singular-value point Sigma = diag(r1,r2,r3) the stabiliser of SO(9) x SO(3)_gauge is SO(6) x Z2^2, so the singlet
wavefunction takes values in the Spin(6) x Z2^2 - invariant part of the Fock space of the 48 Majorana fermions Theta_{alpha A}.
Complex structure J = gamma^0 gamma^1 gamma^2 (commutes with Spin(3) x Spin(6), J^2 = -1): its +i eigenspace is C^2 (x) C^4 for
Spin(3) x Spin(6) = SU(2) x SU(4).  Complex fermions c_{a,u}, a = 2A + sigma (colour A, spin sigma), u = 0..3 the SU(4) index;
Fock mode index m = 4a + u.  The SU(4)-invariant states are, by skew Howe duality, the GL(6) modules S_{(4^k)}(C^6), k = 0..6
(dimensions 1, 126, 1764, 4116, 1764, 126, 1): states with k fermions of each SU(4) index, annihilated by the SU(4) raising
operators sum_a c^+_{a,u} c_{a,u+1}.  They are built per GL(6) weight (n_a = occupation of a), by a dense kernel per block.

Conventions: Theta_m with {Theta_m, Theta_n} = delta, m = 16 A + alpha; gamma matrices from common/gamma.py.
Majorana bilinear sum_mn Theta_m A_mn Theta_n  ->  const + c^+ H c + c P c + c^+ R c^+  (quad_to_c).

Library for fspace.py.  Run directly it checks the SU(4) invariance of the model bilinears and counts the invariants of the
layers k <= kmax.  Usage: python fermions.py [kmax=2]  (seconds to minutes)."""
import numpy as np, itertools, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]
from gamma import gam                      # (9, 16, 16) real symmetric, {g_I, g_J} = 2 delta
NM = 24

# ------------------------------------------------------------------ single-particle basis
def single_particle():
    J = gam[0] @ gam[1] @ gam[2]
    assert np.allclose(J @ J, -np.eye(16)) and np.allclose(J, -J.T)
    w, V = np.linalg.eigh(-1j * J)               # eigenvalue +1  <->  J v = i v
    P = V[:, w > 0]; assert P.shape[1] == 8
    sz = 0.5j * gam[0] @ gam[1]; sx = 0.5j * gam[1] @ gam[2]; sy = 0.5j * gam[2] @ gam[0]
    if not np.allclose(sx @ sy - sy @ sx, 1j * sz): sy = -sy
    assert np.allclose(sx @ sy - sy @ sx, 1j * sz) and np.allclose(sy @ sz - sz @ sy, 1j * sx)
    # spin-up part of the +i space, any orthonormal basis; spin-down = S_- (spin-up)
    ev, U = np.linalg.eigh(P.conj().T @ sz @ P)
    up = P @ U[:, ev > 0]; assert up.shape[1] == 4 and np.allclose(ev[ev > 0], 0.5)
    Sm = sx - 1j * sy
    dn = Sm @ up; dn = dn / np.linalg.norm(dn, axis=0)
    v = np.zeros((2, 4, 16), complex); v[0] = up.T; v[1] = dn.T         # v[sigma, u, alpha]
    # W: Theta_{alpha A} = sum_k W[m, k] c_k + conj(W[m, k]) c_k^+,  m = 16A + alpha, k = 4(2A+sigma) + u
    W = np.zeros((48, NM), complex)
    for A in range(3):
        for s in range(2):
            for u in range(4):
                W[16 * A:16 * A + 16, 4 * (2 * A + s) + u] = v[s, u]
    # check completeness: Theta = W c + W* c^+ with {c, c^+} = 1 requires W W^+ + W* W^T = 1, W^T W = 0
    assert np.allclose(W @ W.conj().T + W.conj() @ W.T, np.eye(48)) and np.allclose(W.T @ W, 0) and np.allclose(W.conj().T @ W, np.eye(NM))
    return W, dict(sx=sx, sy=sy, sz=sz, J=J)

def quad_to_c(A, W):
    """sum_mn Theta_m A_mn Theta_n = const + sum_kl H_kl c^+_k c_l + sum_kl P_kl c_k c_l + sum_kl R_kl c^+_k c^+_l  (P, R antisymmetric)"""
    Wc = W.conj()
    const = np.trace(W.T @ A @ Wc)
    H = Wc.T @ A @ W - (W.T @ A @ Wc).T
    P = W.T @ A @ W; P = (P - P.T) / 2
    R = Wc.T @ A @ Wc; R = (R - R.T) / 2
    return const, H, P, R

# ------------------------------------------------------------------ Majorana bilinears of the model (as 48x48 A matrices)
def midx(A, al): return 16 * A + al
eps = np.zeros((3, 3, 3))
for a, b, c in itertools.permutations(range(3)): eps[a, b, c] = np.linalg.det(np.eye(3)[[a, b, c]])

def A_yukawa(i):
    """Y_i = sum_{B,C} i eps_{iBC} Theta_B gamma^i Theta_C   (H_ferm at Sigma = sum_i r_i Y_i)"""
    A = np.zeros((48, 48), complex)
    for B in range(3):
        for C in range(3):
            if eps[i, B, C]: A[16 * B:16 * B + 16, 16 * C:16 * C + 16] += 1j * eps[i, B, C] * gam[i]
    return A
def A_Jrot(s, t):
    """fermion part of the SO(9) generator: -(i/4) sum_A Theta_A gamma^{st} Theta_A"""
    A = np.zeros((48, 48), complex); g = gam[s] @ gam[t]
    for Acol in range(3): A[16 * Acol:16 * Acol + 16, 16 * Acol:16 * Acol + 16] += -0.25j * g
    return A
def A_Jgauge(Acol, Bcol):
    """fermion part of the gauge generator: -(i/2) sum_alpha (Theta_A Theta_B - Theta_B Theta_A)"""
    A = np.zeros((48, 48), complex)
    for al in range(16):
        A[midx(Acol, al), midx(Bcol, al)] += -0.5j; A[midx(Bcol, al), midx(Acol, al)] += 0.5j
    return A
# generators K_xi of the normal directions (T_xi = -i D_xi + K_xi annihilates singlets; the angular term is sum K^2 / h):
#   S_ij: K = -Jrot(i,j) - Jgauge(i,j);   D_ij: K = -Jrot(i,j) + Jgauge(i,j);   O_aj: K = -Jrot(a,j)
def A_KS(i, j): return -A_Jrot(i, j) - A_Jgauge(i, j)
def A_KD(i, j): return -A_Jrot(i, j) + A_Jgauge(i, j)
def A_KO(a, j): return -A_Jrot(a, j)

# ------------------------------------------------------------------ Fock states with SU(4) weight zero
def weight_zero_states(k):
    subsets = list(itertools.combinations(range(6), k))
    masks = []
    for Ss in itertools.product(subsets, repeat=4):
        m = 0
        for u, S in enumerate(Ss):
            for a in S: m |= 1 << (4 * a + u)
        masks.append(m)
    return np.array(sorted(masks), dtype=np.int64)

def gl6_weight(masks):
    n = np.zeros((len(masks), 6), np.int64)
    for a in range(6):
        for u in range(4): n[:, a] += (masks >> (4 * a + u)) & 1
    return n

def popcount(x):
    x = x.astype(np.int64); c = np.zeros_like(x)
    while np.any(x): c += x & 1; x = x >> 1
    return c

def apply_cdag_c(masks, coef, k, l):
    """c^+_k c_l on the states (masks, coef); returns (new masks, new coef) for the states where it acts"""
    occ_l = (masks >> l) & 1
    if k == l: return masks[occ_l == 1], coef[occ_l == 1]
    m1 = masks[occ_l == 1]; c1 = coef[occ_l == 1]
    s1 = popcount(m1 & ((1 << l) - 1)); m1 = m1 & ~(1 << l)
    occ_k = (m1 >> k) & 1; keep = occ_k == 0
    m2 = m1[keep]; c2 = c1[keep] * np.where(s1[keep] % 2, -1.0, 1.0)
    s2 = popcount(m2 & ((1 << k) - 1)); m2 = m2 | (1 << k)
    return m2, c2 * np.where(s2 % 2, -1.0, 1.0)

def apply_c(masks, coef, l):
    occ = (masks >> l) & 1; m1 = masks[occ == 1]; c1 = coef[occ == 1]
    s = popcount(m1 & ((1 << l) - 1)); return m1 & ~(1 << l), c1 * np.where(s % 2, -1.0, 1.0)
def apply_cdag(masks, coef, k):
    occ = (masks >> k) & 1; m1 = masks[occ == 0]; c1 = coef[occ == 0]
    s = popcount(m1 & ((1 << k) - 1)); return m1 | (1 << k), c1 * np.where(s % 2, -1.0, 1.0)

def invariant_basis(k, verbose=True):
    """list of blocks (gl6 weight n, masks (Fock states of the block), V (n_states x n_inv) orthonormal real kernel)"""
    t0 = time.time(); masks = weight_zero_states(k); n = gl6_weight(masks)
    keys = [tuple(r) for r in n]; blocks = {}
    for i, key in enumerate(keys): blocks.setdefault(key, []).append(i)
    out = []; tot = 0
    for key, idx in sorted(blocks.items()):
        ms = masks[idx]; nb = len(ms); rows = []
        # raising operators E_u = sum_a c^+_{a,u} c_{a,u+1}: images, stacked into one matrix
        for u in range(3):
            cols_img = {}
            for j in range(nb):
                acc = {}
                for a in range(6):
                    m2, c2 = apply_cdag_c(ms[j:j + 1], np.ones(1), 4 * a + u, 4 * a + u + 1)
                    for mm, cc in zip(m2, c2): acc[int(mm)] = acc.get(int(mm), 0.0) + cc
                cols_img[j] = acc
            tgt = sorted({mm for acc in cols_img.values() for mm in acc})
            pos = {mm: i for i, mm in enumerate(tgt)}
            M = np.zeros((len(tgt), nb))
            for j, acc in cols_img.items():
                for mm, cc in acc.items(): M[pos[mm], j] += cc
            rows.append(M)
        Mall = np.vstack(rows) if rows else np.zeros((0, nb))
        if Mall.shape[0] == 0: V = np.eye(nb)
        else:
            U_, S, Vt = np.linalg.svd(Mall, full_matrices=True)
            rk = int((S > 1e-10 * max(S[0], 1)).sum()); V = Vt[rk:].T
        if V.shape[1]: out.append((key, ms, V)); tot += V.shape[1]
    if verbose: print(f"k = {k}: {len(masks)} weight-zero states in {len(blocks)} GL(6)-weight blocks, {tot} SU(4) invariants ({time.time()-t0:.0f}s)", flush=True)
    return out

if __name__ == "__main__":
    W, ops = single_particle()
    # SU(4) invariance of the model bilinears that must be SU(4) invariant: no pairing parts, number-conserving part = M6 (x) 1_4
    def check(name, A):
        const, H, P, R = quad_to_c(A, W)
        Hr = H.reshape(6, 4, 6, 4)
        M6 = np.einsum("aubu->ab", Hr) / 4
        dev = np.abs(Hr - np.einsum("ab,uv->aubv", M6, np.eye(4))).max()
        print(f"{name:10s}: |pairing| = {max(abs(P).max(), abs(R).max()):.1e}, |H - M6 x 1| = {dev:.1e}, const = {const:.3f}", flush=True)
    for i in range(3): check(f"Y_{i}", A_yukawa(i))
    for i, j in [(0, 1), (0, 2), (1, 2)]: check(f"K_S{i}{j}", A_KS(i, j)); check(f"K_D{i}{j}", A_KD(i, j))
    const, H, P, R = quad_to_c(A_KO(3, 0), W)
    print(f"K_O(3,0): |number-conserving part| = {abs(H).max():.1e}, |pairing| = {max(abs(P).max(), abs(R).max()):.2f}")
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dims = {}
    for k in range(kmax + 1):
        bl = invariant_basis(k); dims[k] = sum(b[2].shape[1] for b in bl)
    print("invariant dimensions:", dims, "(expected 1, 126, 1764, 4116, 1764, 126, 1)")
