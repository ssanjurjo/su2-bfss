"""Real symmetric 16x16 gamma matrices of SO(9) (Majorana representation) and the identities the BFSS algebra needs.

The matrices are found by a deterministic search over tensor products of Pauli-type factors, so every import builds the
same set.  Usage: python gamma.py  (prints the checks; seconds)."""
import numpy as np, itertools
s0 = np.eye(2); sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]])
def kron(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms: out = np.kron(out, m)
    return out
# Build SO(8) chiral gammas 8x8 real via a standard construction, then SO(9) 16x16 real symmetric.
# Construction: real symmetric 16x16 gammas of Cl(9): use 4 Pauli tensor factors; real symmetric matrices
# among kron products of {1, sx, sz} (real symmetric) and (i sy) (real antisymmetric): a product of two i*sy factors is real symmetric.
# Choose 9 mutually anticommuting real symmetric 16x16 matrices.
E = np.eye(2); A = 1j * sy   # A is real antisymmetric: [[0,1],[-1,0]] up to sign
A = np.real(A)
cands = []
for combo in itertools.product([E, sx, sz, A], repeat=4):
    m = np.real(kron(*combo)) if not np.iscomplexobj(kron(*combo)) else np.real(kron(*combo))
    if np.allclose(m, m.T) and np.allclose(m @ m, np.eye(16)):
        cands.append(m)
# greedy search for 9 mutually anticommuting ones
def anticommute(a, b): return np.allclose(a @ b + b @ a, 0)
best = None
def search(chosen, start):
    global best
    if len(chosen) == 9:
        best = list(chosen); return True
    for i in range(start, len(cands)):
        if all(anticommute(cands[i], c) for c in chosen):
            if search(chosen + [cands[i]], i + 1): return True
    return False
search([], 0)
gam = np.array(best)             # shape (9,16,16), real symmetric, {g_I, g_J} = 2 delta
def gamma2(I, J): return 0.5 * (gam[I] @ gam[J] - gam[J] @ gam[I])
gam2 = np.array([[gamma2(I, J) for J in range(9)] for I in range(9)])   # antisymmetric real
def checks():
    for I in range(9):
        for J in range(9):
            assert np.allclose(gam[I] @ gam[J] + gam[J] @ gam[I], 2 * (I == J) * np.eye(16))
        assert np.allclose(gam[I], gam[I].T) and np.allclose(gam[I], gam[I].real)
    # Fierz identity peculiar to d=9 (and 2,3,5): gamma^I_{(ab} gamma^I_{c)d} = delta_{(ab} delta_{c)d}
    G = np.einsum('iab,icd->abcd', gam, gam)
    lhs = G + np.einsum('abcd->bcad', G) + np.einsum('abcd->cabd', G)
    D = np.einsum('ab,cd->abcd', np.eye(16), np.eye(16))
    rhs = D + np.einsum('abcd->bcad', D) + np.einsum('abcd->cabd', D)
    assert np.allclose(lhs, rhs), "Fierz identity failed"
    # gamma^{123} squares to +1 or -1 and commutes with gamma^{1,2,3}, anticommutes with gamma^{4..9}
    g123 = gam[0] @ gam[1] @ gam[2]
    assert np.allclose(g123 @ g123, -np.eye(16)) or np.allclose(g123 @ g123, np.eye(16))
    for I in range(3): assert np.allclose(g123 @ gam[I], gam[I] @ g123)
    for I in range(3, 9): assert np.allclose(g123 @ gam[I], -gam[I] @ g123)
    return True
if __name__ == "__main__":
    print("found", len(cands), "candidate real symmetric involutions; 9 anticommuting:", best is not None)
    print("checks:", checks())
    g123 = gam[0] @ gam[1] @ gam[2]
    print("(gamma^123)^2 = ", "+1" if np.allclose(g123 @ g123, np.eye(16)) else "-1", "; symmetric:", np.allclose(g123, g123.T))
    # count symmetric / antisymmetric among gamma^{I1..Ik}
    for k, name in [(0, "1"), (1, "g^I"), (2, "g^{IJ}"), (3, "g^{IJK}"), (4, "g^{IJKL}")]:
        idx = list(itertools.combinations(range(9), k))
        m = np.eye(16) if k == 0 else np.linalg.multi_dot([gam[i] for i in idx[0]]) if k > 1 else gam[idx[0][0]]
        print(f"{name}: symmetric={np.allclose(m, m.T)}, antisymmetric={np.allclose(m, -m.T)}")
