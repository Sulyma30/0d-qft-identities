"""Two-flavour Hubbard model in 0D with the composite fields of Sec. III A chosen to satisfy the cancellation
conditions after Eq. (86):
  particle-hole:  U^a_il (chi0^-1)_ab U^b_jk = - U^a_jl (chi0^-1)_ab U^b_ik = u_ijkl   (four components),
  pairing:        Ubar^pp_ij (chi0^pp)^-1 U^pp_kl = u_ijkl.
Definitions only (no computation at import)."""
import fermions as fm
from superalg import SA, sa_exp, sa_log1p, merge_sign
from itertools import product
import numpy as np

fm.NF = 2; fm.F = lambda i: i; fm.FD = lambda i: 2 + i; fm.SRC = lambda a: 4 + a
F, FD, SRC = fm.F, fm.FD, fm.SRC

h = np.array([[0.9, 0.3], [0.3, -1.1]])          # spin-mixing quadratic part
s2 = 1 / np.sqrt(2)
E = lambda i, j: np.array([[1.0 if (a, b) == (i, j) else 0.0 for b in range(2)] for a in range(2)])
vs = [(E(0, 0) + E(1, 1)) * s2, (E(0, 0) - E(1, 1)) * s2, (E(0, 1) + E(1, 0)) * s2, (E(0, 1) - E(1, 0)) * s2]
lams = [-1, +1, +1, -1]       # eigenvalues of N_(il),(jk) = u_ijkl in units of u (ph composites U^a = vs[a])
Upp = np.array([[0, 1.0], [-1.0, 0]])
Uppb = np.array([[0, 1.0], [-1.0, 0]])


def couplings(u0):
    """Interaction tensor and HS parameters for coupling u0 (u_0101 = u0)."""
    u = fm.antisym_u(2, {(0, 1, 0, 1): u0})
    chi0ph = [1 / (lam * u0) for lam in lams]
    chi0pp = 1 / u0
    return u, chi0ph, chi0pp


def condition_residuals(u0):
    """Max deviations of the three cancellation conditions."""
    u, chi0ph, chi0pp = couplings(u0)
    N = sum(np.einsum('il,jk->ijkl', v, v) / c for v, c in zip(vs, chi0ph))
    Nx = sum(np.einsum('jl,ik->ijkl', v, v) / c for v, c in zip(vs, chi0ph))
    Npp = np.einsum('ij,kl->ijkl', Uppb, Upp) / chi0pp
    return np.abs(N - u).max(), np.abs(-Nx - u).max(), np.abs(Npp - u).max()


def generating(u0, D):
    """W(J, K) with fermion sources (Grassmann generators 4..7) and composite sources K_0..K_5."""
    u, chi0ph, chi0pp = couplings(u0)
    nb = 6
    f = lambda g: SA.fvar(nb, D, g)
    Ex = SA(nb, D)
    for i, j in product(range(2), repeat=2):
        Ex = Ex - f(FD(i)) * f(F(j)) * h[i, j]
    for i, j, k, l in product(range(2), repeat=4):
        if u[i, j, k, l]:
            Ex = Ex - f(FD(i)) * f(FD(j)) * f(F(k)) * f(F(l)) * (u[i, j, k, l] / 4)
    for a in range(4):
        Ex = Ex - f(SRC(a)) * f(a)
    k = [SA.bvar(nb, D, i) for i in range(nb)]
    for al, (v, c0) in enumerate(zip(vs, chi0ph)):
        ph = SA(nb, D)
        for i, j in product(range(2), repeat=2):
            ph = ph + f(FD(i)) * f(F(j)) * v[i, j]
        Ex = Ex - k[al] * ph - k[al] * k[al] * (c0 / 2)
    pp = SA(nb, D); ppb = SA(nb, D)
    for i, j in product(range(2), repeat=2):
        pp = pp + f(F(i)) * f(F(j)) * (Upp[i, j] / 2)
        ppb = ppb + f(FD(i)) * f(FD(j)) * (Uppb[i, j] / 2)
    Ex = Ex - k[4] * pp - k[5] * ppb - k[4] * k[5] * chi0pp
    Zfull = sa_exp(Ex, 8 + D)
    Z = SA(nb, D)
    for (b, fs), v in Zfull.c.items():
        if set(range(4)) <= set(fs):
            rest = tuple(g for g in fs if g >= 4)
            s, _ = merge_sign((0, 1, 2, 3), rest)
            Z.c[(b, rest)] = Z.c.get((b, rest), 0.0) + s * v
    Z0 = Z.const_term()
    return -sa_log1p(Z * (1 / Z0) - 1, 4 + D), nb, u


def legendre_with_sources(W, nb, D):
    """Composite effective action Gamma(phibar, psi) (Grassmann 0..3 = phibar, bosonic = delta psi), together with
    the sources J(phibar, psi), K(phibar, psi)."""
    X = [W.dL(SRC(a)) for a in range(4)]
    Y = [W.db(i) for i in range(nb)]
    M = np.array([[X[a].c.get(((0,) * nb, (SRC(b),)), 0.0) for b in range(4)] for a in range(4)])
    Mi = np.linalg.inv(M)
    phi0 = [y.const_term() for y in Y]
    B = np.array([[Y[i].c.get((tuple(1 if q == j else 0 for q in range(nb)), ()), 0.0) for j in range(nb)]
                  for i in range(nb)])
    Bi = np.linalg.inv(B)
    pb = [SA.fvar(nb, D, a) for a in range(4)]
    dps = [SA.bvar(nb, D, i) for i in range(nb)]
    srcJ = [SA.fvar(nb, D, SRC(b)) for b in range(4)]
    kv = [SA.bvar(nb, D, i) for i in range(nb)]
    Nl = [X[a] - sum((srcJ[b] * M[a, b] for b in range(4)), SA(nb, D)) for a in range(4)]
    Rl = [Y[i] - phi0[i] - sum((kv[j] * B[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    J = [sum((pb[b] * Mi[a, b] for b in range(4)), SA(nb, D)) for a in range(4)]
    K = [sum((dps[j] * Bi[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    for _ in range(D + 4):
        fs = {SRC(b): J[b] for b in range(4)}
        NJ = [n.substitute(K, fs, nb) for n in Nl]
        RK = [r.substitute(K, fs, nb) for r in Rl]
        J = [sum(((pb[b] - NJ[b]) * Mi[a, b] for b in range(4)), SA(nb, D)) for a in range(4)]
        K = [sum(((dps[j] - RK[j]) * Bi[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    fs = {SRC(b): J[b] for b in range(4)}
    G = W.substitute(K, fs, nb)
    for a in range(4):
        G = G - J[a] * pb[a]
    for i in range(nb):
        G = G - K[i] * (dps[i] + phi0[i])
    return G, J, K
