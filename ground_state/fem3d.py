"""Tensor-grid Lagrange finite elements for the reduced SU(2) problem on the chamber r1 >= r2 >= r3 >= 0.

Coordinates a = r3, b = r2 - r3, c = r1 - r2 (the chamber is the positive octant, walls are coordinate planes, d^3r = da db dc).
Weighted form with J = |(r1^2-r2^2)(r1^2-r3^2)(r2^2-r3^2)| (r1 r2 r3)^6:
    <psi|H|psi> = int J [ sum_i |d_{r_i} psi|^2 + psi^+ A(r) psi ],   <psi|psi> = int J |psi|^2,
    sum_i (d_{r_i} psi)^2 = psi_a^2 + 2 psi_b^2 + 2 psi_c^2 - 2 psi_a psi_b - 2 psi_b psi_c.
The walls a = 0, b = 0, c = 0 carry natural boundary conditions (J vanishes there); the outer faces are Dirichlet (a box).
Stiffness, mass and potentials are consistent (Galerkin, NGAUSS^3 Gauss points per element): the discrete eigenvalues are
Rayleigh-Ritz upper bounds.  (Lumping the J-weighted mass creates spurious low modes on the walls, where J vanishes.)  A(r) = V_bos(r) + sum_t f_t(r) M_t with fermion matrices M_t (fspace_ops.pkl).

Element orders and quadrature: the defaults are cubic elements in a, b, quadratic elements in c and six Gauss points per
direction (environment FEM_ORDER_AB, FEM_ORDER_C, FEM_NGAUSS override them; FEM_NGAUSS_AB and
FEM_NGAUSS_C set the Gauss order across and along the valley separately), and Grid(..., qs=...)
takes the orders stored in a saved state.  Library module, no command line."""
import numpy as np, scipy.sparse as sp, itertools
import os, contextlib
NGAUSS = int(os.environ.get("FEM_NGAUSS", "6"))
GA, GW = np.polynomial.legendre.leggauss(NGAUSS)
# separate Gauss orders across the valley (a, b) and along it (c): FEM_NGAUSS_AB, FEM_NGAUSS_C (default: FEM_NGAUSS in both)
NGAUSS_AB = int(os.environ.get("FEM_NGAUSS_AB", str(NGAUSS))); NGAUSS_C = int(os.environ.get("FEM_NGAUSS_C", str(NGAUSS)))
GA_AB, GW_AB = np.polynomial.legendre.leggauss(NGAUSS_AB); GA_C, GW_C = np.polynomial.legendre.leggauss(NGAUSS_C)
def quadrature():
    """the Gauss orders in use, as they are recorded in a saved state"""
    return dict(ngauss_ab=NGAUSS_AB, ngauss_c=NGAUSS_C)
@contextlib.contextmanager
def gauss(ngauss_ab, ngauss_c):
    """temporarily use other Gauss orders, to read a saved state with the ones it was computed with"""
    global NGAUSS_AB, NGAUSS_C, GA_AB, GW_AB, GA_C, GW_C
    old = (NGAUSS_AB, NGAUSS_C, GA_AB, GW_AB, GA_C, GW_C)
    NGAUSS_AB, NGAUSS_C = int(ngauss_ab), int(ngauss_c)
    GA_AB, GW_AB = np.polynomial.legendre.leggauss(NGAUSS_AB); GA_C, GW_C = np.polynomial.legendre.leggauss(NGAUSS_C)
    try: yield
    finally: NGAUSS_AB, NGAUSS_C, GA_AB, GW_AB, GA_C, GW_C = old
def check_quadrature(d, name="the state"):
    """A state that records its Gauss orders must be read with the same ones: they fix the norm of its coefficients."""
    q = d.get("fem")
    if q and (q["ngauss_ab"], q["ngauss_c"]) != (NGAUSS_AB, NGAUSS_C):
        raise SystemExit(f"{name} was computed with FEM_NGAUSS_AB={q['ngauss_ab']} FEM_NGAUSS_C={q['ngauss_c']}, "
                         f"and this run uses {NGAUSS_AB} and {NGAUSS_C}: set the two variables and run again")
AMAT = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

def jac(r1, r2, r3):
    return np.abs((r1**2 - r2**2) * (r1**2 - r3**2) * (r2**2 - r3**2)) * (r1 * r2 * r3) ** 6

def potential_terms(r1, r2, r3):
    """scalar coefficient functions: V_bos and f_t for the fermion terms (keys as in fspace_ops)"""
    r = (r1, r2, r3); out = {"Vbos": r1**2 * r2**2 + r1**2 * r3**2 + r2**2 * r3**2}
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        out[("KS2", (i, j))] = 1.0 / (2 * (r[i] - r[j]) ** 2); out[("KD2", (i, j))] = 1.0 / (2 * (r[i] + r[j]) ** 2)
    for j in range(3): out[("O", j)] = 1.0 / r[j] ** 2; out[("Y", j)] = r[j]
    return out

def env_none(r1, r2, r3):
    one = np.ones_like(r1); z = np.zeros_like(r1); return one, (z, z, z)
def env_valley(w=1.0):
    """exp(-w (r1 r2^2 + r1 r3^2 + r2 r3^2) / 2): the transverse oscillator ground state along every flat direction (r1 >= r2 >= r3)"""
    def f(r1, r2, r3):
        W = w * (r1 * r2**2 + r1 * r3**2 + r2 * r3**2) / 2; g = np.exp(-W)
        d1 = w * (r2**2 + r3**2) / 2; d2 = w * (2 * r1 * r2 + r3**2) / 2; d3 = w * (2 * r1 * r3 + 2 * r2 * r3) / 2
        return g, (-d1 * g, -d2 * g, -d3 * g)
    return f

def lagrange1d(order):
    """1D Lagrange shape functions on [0,1] with equispaced nodes: lists of (value(x), derivative(x)) callables"""
    xs = np.linspace(0, 1, order + 1); out = []
    for i in range(order + 1):
        others = [xj for j, xj in enumerate(xs) if j != i]
        den = np.prod([xs[i] - xj for xj in others])
        def val(x, others=others, den=den): return np.prod([x - xj for xj in others], axis=0) / den
        def der(x, others=others, den=den):
            tot = 0.0
            for m in range(len(others)): tot = tot + np.prod([x - xj for jj, xj in enumerate(others) if jj != m], axis=0)
            return tot / den
        out.append((val, der))
    return out

class Grid:
    def __init__(self, a, b, c, qs=None):
        if qs is None:                                                               # element orders (a, b, c)
            qab = int(os.environ.get("FEM_ORDER_AB", "3")); qc = int(os.environ.get("FEM_ORDER_C", "2")); qs = (qab, qab, qc)
        qab, qc = int(qs[0]), int(qs[2])
        self.qs = (qab, qab, qc); self.order = max(self.qs)
        self.ea, self.eb, self.ec = map(np.asarray, (a, b, c))                       # element vertices
        ref = lambda v, q: np.concatenate([np.linspace(v[i], v[i + 1], q + 1)[:-1] for i in range(len(v) - 1)] + [v[-1:]])
        self.a, self.b, self.c = ref(self.ea, qab), ref(self.eb, qab), ref(self.ec, qc); self.shape = (len(self.a), len(self.b), len(self.c))
        na, nb, nc = self.shape
        # interior + walls are unknowns; the outer faces (last node in each direction) are Dirichlet
        ia, ib, ic = np.meshgrid(np.arange(na), np.arange(nb), np.arange(nc), indexing="ij")
        self.free = ((ia < na - 1) & (ib < nb - 1) & (ic < nc - 1)).ravel()
        self.index = -np.ones(na * nb * nc, int); self.index[self.free] = np.arange(self.free.sum()); self.N = int(self.free.sum())
    def assemble(self, terms=None, env=env_none):
        """sparse stiffness K, mass M, potential matrices {term: P} for the basis env(r) N_alpha(r) (N_alpha: Lagrange elements).
        The element matrices have the same sparsity pattern at every Gauss point, so they are summed in dense (element, local,
        local) arrays and converted to CSR once."""
        a, b, c = self.ea, self.eb, self.ec; na, nb, nc = self.shape; qa, qb, qc = self.qs
        ha, hb, hc = np.diff(a), np.diff(b), np.diff(c)
        E = np.array(list(itertools.product(range(len(a) - 1), range(len(b) - 1), range(len(c) - 1))))   # element (i, j, k)
        La, Lb, Lc = lagrange1d(qa), lagrange1d(qb), lagrange1d(qc)
        corners = list(itertools.product(range(qa + 1), range(qb + 1), range(qc + 1)))    # local nodes
        gid = lambda i, j, k: (i * nb + j) * nc + k                                         # i, j, k: global node indices
        ids = np.stack([gid(qa * E[:, 0] + p_, qb * E[:, 1] + q_, qc * E[:, 2] + s_) for (p_, q_, s_) in corners], 1)   # (el, nl)
        Kacc = 0.0; Macc = 0.0; Pacc = {}
        for (ga, gb, gc), (wa_, wb_, wc_) in zip(itertools.product(GA_AB, GA_AB, GA_C), itertools.product(GW_AB, GW_AB, GW_C)):
            xa, xb, xc = (1 + ga) / 2, (1 + gb) / 2, (1 + gc) / 2                               # reference coordinates in [0,1]
            A_ = a[E[:, 0]] + xa * ha[E[:, 0]]; B_ = b[E[:, 1]] + xb * hb[E[:, 1]]; C_ = c[E[:, 2]] + xc * hc[E[:, 2]]
            r3 = A_; r2 = A_ + B_; r1 = r2 + C_
            vol = ha[E[:, 0]] * hb[E[:, 1]] * hc[E[:, 2]] * wa_ * wb_ * wc_ / 8.0                # Gauss weight x element volume
            Jw = jac(r1, r2, r3) * vol
            pt = potential_terms(r1, r2, r3)
            g, (g1, g2, g3) = env(r1, r2, r3)
            # gradient of g in (a, b, c): d_a = d_r1 + d_r2 + d_r3, d_b = d_r1 + d_r2, d_c = d_r1
            ga_, gb_, gc_ = g1 + g2 + g3, g1 + g2, g1
            N = []; dN = []
            for (p_, q_, s_) in corners:
                Na, dNa = La[p_][0](xa), La[p_][1](xa); Nb, dNb = Lb[q_][0](xb), Lb[q_][1](xb); Nc, dNc = Lc[s_][0](xc), Lc[s_][1](xc)
                da = dNa / ha[E[:, 0]]; db = dNb / hb[E[:, 1]]; dc = dNc / hc[E[:, 2]]
                Nv = Na * Nb * Nc; gradN = np.stack([da * Nb * Nc, Na * db * Nc, Na * Nb * dc], axis=1)
                N.append(g * Nv); dN.append(g[:, None] * gradN + Nv * np.stack([ga_, gb_, gc_], axis=1))
            N = np.stack(N, 1); dN = np.stack(dN, 1)                                            # (el, nl), (el, nl, 3)
            Kacc = Kacc + Jw[:, None, None] * np.einsum("eap,pq,ebq->eab", dN, AMAT, dN, optimize=True)
            mloc = Jw[:, None, None] * N[:, :, None] * N[:, None, :]; Macc = Macc + mloc
            for kk in pt:
                if terms is None or kk in terms: Pacc[kk] = Pacc.get(kk, 0.0) + mloc * pt[kk][:, None, None]
        nl = len(corners); rows = np.repeat(ids, nl, axis=1).ravel(); cols = np.tile(ids, (1, nl)).ravel()
        keep = self.free[rows] & self.free[cols]; ri, ci = self.index[rows[keep]], self.index[cols[keep]]
        mk = lambda v: sp.csr_matrix((v.ravel()[keep], (ri, ci)), shape=(self.N, self.N))
        K = mk(Kacc); M = mk(Macc)
        P = {kk: mk(v) for kk, v in Pacc.items()}
        return K, M, P

def prune_scale(K, M, pots, tol=1e-12, keep=None):
    """drop basis functions of negligible norm (the envelope makes them numerically null) and scale to unit mass diagonal;
    keep (optional) imposes the kept set"""
    d = M.diagonal().real
    if keep is None: keep = np.nonzero(d > tol * d.max())[0]
    s = 1 / np.sqrt(d[keep])
    Sd = sp.diags(s); sub = lambda A: (Sd @ A[keep][:, keep] @ Sd).tocsr()
    return keep, sub(K), sub(M), {k: sub(v) for k, v in pots.items()}

def graded(xmax, h0, q, hmax):
    """nodes from 0 to xmax: spacings h0 q^k up to hmax, then uniform (spacing <= hmax) to xmax"""
    x = [0.0]; h = h0
    while h < hmax and x[-1] + h < xmax: x.append(x[-1] + h); h *= q
    m = max(1, int(np.ceil((xmax - x[-1]) / hmax - 1e-9)))
    return np.r_[x, x[-1] + (xmax - x[-1]) * np.arange(1, m + 1) / m]

def geometric(n, xmax, h0):
    """n+1 nodes from 0 to xmax, first spacing ~h0, geometric growth"""
    if h0 * n >= xmax: return np.linspace(0, xmax, n + 1)
    from scipy.optimize import brentq
    q = brentq(lambda q: h0 * (q ** n - 1) / (q - 1) - xmax, 1 + 1e-9, 10)
    return np.r_[0, np.cumsum(h0 * q ** np.arange(n))]
