"""Appendix C: bilinear plus trilinear composite fields (fermion-boson exchange), two-flavour Grassmann theory.

Fields: f0, f1 (generators 0, 1), f+0, f+1 (generators 2, 3). Fermion sources J_a: generators 4..7.
Bilinears: densities n_alpha = f+_alpha f_alpha (alpha = 0, 1) with bosonic sources K_alpha.
Trilinears phi^alpha f_i, phi^alpha f+_i: in 0D only n0 f1, n1 f0, n0 f+1, n1 f+0 are nonzero; sources 8..11.
(C1) is applied to the whole generating functional: K_alpha -> K_alpha(phi) = K_alpha + J_T f (or f+) both in the
source term and in the Hubbard-Stratonovich term 1/2 K chi0 K.
Trilinear variable as in (C2): psi^{alpha i} = phibar^{alpha i} - psi^alpha fbar_i (and the conjugate).
Coupling: chi0^-1 = [[0, -U], [-U, 0]], so that U^a_il (chi0^-1)_ab U^b_jk f+_j f_k f_l reproduces dS_int/df+_i term by
term, including the coefficients of the identically vanishing trilinears n_i f_i (zero diagonal).

Checks:
 (a) the cancellation identity  dS_int/df+_i = sum U^a_il (chi0^-1)_ab n_b f_l  as an operator identity;
 (b) Eq. (1): along the stationary point the composite action reproduces the exact 1PI vertices;
 (c) structure (C4)-(C5): Gamma[phibar, psi] = S_0[phibar] + (terms linear in psi with polynomial phibar
     coefficients) + Lambda[psi]; every other Taylor coefficient containing phibar must vanish, in particular the
     residual four-fermion vertex at fixed psi.
Usage: python3 appendixC.py [D]   (D = total degree of the truncation, default 4; D = 5 tests five-leg vertices)."""
import sys, time
import numpy as np
from itertools import product
import superalg2
from superalg2 import SA, sa_exp, sa_log1p, merge_sign
import fermions as fm

TRI = [(0, 1), (1, 0), (0, 3), (1, 2)]      # (alpha, field generator multiplying n_alpha)
h = np.array([[0.4, 0.25], [0.25, -0.6]])
U = 0.8
c = np.array([[0.0, -U], [-U, 0.0]])          # chi0^-1
res = []


def rep(label, val, ref=0.0, tol=1e-10):
    err = abs(val - ref)
    ok = err < tol * max(1.0, abs(ref))
    res.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {label:66s} {val:+.6e}  (expected {ref:+.6e})")


def build_W(D):
    superalg2.COUNTED[0] = lambda g: g >= 4          # field generators do not count toward the degree
    nb = 2
    f = lambda g: SA.fvar(nb, D, g)
    E = SA(nb, D)
    for i, j in product(range(2), repeat=2):
        E = E - f(2 + i) * f(j) * h[i, j]
    E = E - f(2) * f(3) * f(0) * f(1) * U             # 1/4 u f+ f+ f f with u_0101 = U
    for a in range(4):
        E = E - f(4 + a) * f(a)
    n = [f(2) * f(0), f(3) * f(1)]
    Kphi = [SA.bvar(nb, D, al) for al in range(2)]
    for t, (al, g) in enumerate(TRI):
        Kphi[al] = Kphi[al] + f(8 + t) * f(g)
    for al in range(2):
        E = E - Kphi[al] * n[al]
    chi0 = np.linalg.inv(c)
    for a, b in product(range(2), repeat=2):
        E = E - Kphi[a] * Kphi[b] * (chi0[a, b] / 2)
    Zf = sa_exp(E, 12 + D)
    Z = SA(nb, D)
    for (b, fs), v in Zf.c.items():
        if {0, 1, 2, 3} <= set(fs):
            rest = tuple(g for g in fs if g >= 4)
            s, _ = merge_sign((0, 1, 2, 3), rest)
            Z.c[(b, rest)] = Z.c.get((b, rest), 0.0) + s * v
    W = -sa_log1p(Z * (1 / Z.const_term()) - 1, 6 + D)
    superalg2.COUNTED[0] = lambda g: True
    return W


def legendre(W, D):
    """Odd sources 4..11 -> odd variables 0..7 (0..3 phibar, 4..7 trilinear phibar_T); bosonic K -> phibar_B."""
    nb = 2
    odd_src = list(range(4, 12))
    X = [W.dL(s) for s in odd_src]
    Y = [W.db(i) for i in range(nb)]
    M = np.array([[X[a].c.get(((0,) * nb, (odd_src[b],)), 0.0) for b in range(8)] for a in range(8)])
    Mi = np.linalg.inv(M)
    phi0 = [y.const_term() for y in Y]
    B = np.array([[Y[i].c.get((tuple(1 if k == j else 0 for k in range(nb)), ()), 0.0) for j in range(nb)]
                  for i in range(nb)])
    Bi = np.linalg.inv(B)
    xb = [SA.fvar(nb, D, a) for a in range(8)]
    db = [SA.bvar(nb, D, i) for i in range(nb)]
    srcs = [SA.fvar(nb, D, s) for s in odd_src]
    kv = [SA.bvar(nb, D, i) for i in range(nb)]
    Nl = [X[a] - sum((srcs[b] * M[a, b] for b in range(8)), SA(nb, D)) for a in range(8)]
    Rl = [Y[i] - phi0[i] - sum((kv[j] * B[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    J = [sum((xb[b] * Mi[a, b] for b in range(8)), SA(nb, D)) for a in range(8)]
    K = [sum((db[j] * Bi[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    for _ in range(D + 6):
        fs = {odd_src[b]: J[b] for b in range(8)}
        NJ = [x.substitute(K, fs, nb) for x in Nl]
        RK = [r.substitute(K, fs, nb) for r in Rl]
        J = [sum(((xb[b] - NJ[b]) * Mi[a, b] for b in range(8)), SA(nb, D)) for a in range(8)]
        K = [sum(((db[j] - RK[j]) * Bi[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    fs = {odd_src[b]: J[b] for b in range(8)}
    G = W.substitute(K, fs, nb)
    for a in range(8):
        G = G - J[a] * xb[a]
    for i in range(nb):
        G = G - K[i] * (db[i] + phi0[i])
    return G, phi0


def to_C2_variables(G, phi0, D):
    """Express Gamma through psi_T = phibar_T - psi_B fbar  (C2), i.e. substitute phibar_T = psi_T + psi_B fbar."""
    nb = 2
    fs = {a: SA.fvar(nb, D, a) for a in range(4)}
    for t, (al, g) in enumerate(TRI):
        psiB = SA.bvar(nb, D, al) + phi0[al]
        fs[4 + t] = SA.fvar(nb, D, 4 + t) + psiB * SA.fvar(nb, D, g)
    return G.substitute([SA.bvar(nb, D, 0), SA.bvar(nb, D, 1)], fs, nb)


def stationary(G, D, nb=2, nT=4):
    """Solve dGamma/dpsi = 0 for psi_T (odd) and psi_B (even) as series in phibar; return Gamma at the solution."""
    ET = [G.dL(4 + x) for x in range(nT)]
    EB = [G.db(a) for a in range(nb)]
    HT = np.array([[ET[x].c.get(((0,) * nb, (4 + y,)), 0.0) for y in range(nT)] for x in range(nT)])
    HB = np.array([[EB[a].c.get((tuple(1 if k == b else 0 for k in range(nb)), ()), 0.0) for b in range(nb)]
                   for a in range(nb)])
    HTi, HBi = np.linalg.inv(HT), np.linalg.inv(HB)
    RT = [ET[x] - sum((SA.fvar(nb, D, 4 + y) * HT[x, y] for y in range(nT)), SA(nb, D)) for x in range(nT)]
    RB = [EB[a] - sum((SA.bvar(nb, D, b) * HB[a, b] for b in range(nb)), SA(nb, D)) for a in range(nb)]
    psiT = [SA(nb, D) for _ in range(nT)]
    psiB = [SA(nb, D) for _ in range(nb)]
    for _ in range(D + 4):
        fs = {g: SA.fvar(nb, D, g) for g in range(4)}
        fs.update({4 + x: psiT[x] for x in range(nT)})
        rT = [r.substitute(psiB, fs, nb) for r in RT]
        rB = [r.substitute(psiB, fs, nb) for r in RB]
        psiT = [sum((rT[y] * (-HTi[x, y]) for y in range(nT)), SA(nb, D)) for x in range(nT)]
        psiB = [sum((rB[b] * (-HBi[a, b]) for b in range(nb)), SA(nb, D)) for a in range(nb)]
    fs = {g: SA.fvar(nb, D, g) for g in range(4)}
    fs.update({4 + x: psiT[x] for x in range(nT)})
    return G.substitute(psiB, fs, nb)


ALLOWED = {(2, 0, 0), (2, 0, 1), (1, 1, 0)}    # (deg phibar, deg psi_T, deg psi_B) of terms with phibar


if __name__ == '__main__':
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t0 = time.time()
    print("=== (a) cancellation identity: dS_int/df+_i = sum_ab U^a_il (chi0^-1)_ab n_b f_l ===")
    superalg2.COUNTED[0] = lambda g: True
    f = lambda g: SA.fvar(1, 1, g)
    Sint = f(2) * f(3) * f(0) * f(1) * U
    n = [f(2) * f(0), f(3) * f(1)]
    for i in range(2):
        rhs = SA(1, 1)
        for a, b in product(range(2), repeat=2):
            if a == i:                                     # density: U^a_il = delta_ai delta_il
                rhs = rhs + n[b] * f(i) * c[a, b]
        diff = Sint.dL(2 + i) - rhs
        rep(f"operator identity for f+_{i}: max|difference| of coefficients",
            max((abs(v) for v in diff.c.values()), default=0.0))

    W = build_W(D)
    G, phi0 = legendre(W, D)
    print(f"\n(composite effective action built at total degree {D}: {time.time() - t0:.0f}s)")

    print("\n=== (b) Eq. (1): Gamma_1PI[phibar] = Gamma[phibar, psi_phibar] ===")
    W1, nb1, _ = fm.Model(h=h, u=fm.antisym_u(2, {(0, 1, 0, 1): U}), Uph=np.eye(2), Upp=np.zeros((2, 2)),
                          Uppb=np.zeros((2, 2)), chi0ph=1, chi0pp=1).generating(False, 3)
    G1 = fm.legendre_ferm(W1, nb1, 3, False)
    Gst = stationary(G, D)
    for a, b in [(2, 0), (2, 1), (3, 0), (3, 1)]:
        rep(f"self-energy Gamma_ab, (a,b) = ({a},{b})", fm.dvals(Gst, [a, b]), fm.dvals(G1, [a, b]))
    rep("four-point vertex Gamma_{f+0 f+1 f0 f1}", fm.dvals(Gst, [2, 3, 0, 1]), fm.dvals(G1, [2, 3, 0, 1]))

    print("\n=== (c) structure (C4)-(C5) in the variables of (C2) ===")
    Gv = to_C2_variables(G, phi0, D)
    cats = {}
    for (b, fs), v in Gv.c.items():
        nphi = sum(1 for g in fs if g < 4)
        if nphi:
            key = (nphi, sum(1 for g in fs if g >= 4), sum(b))
            cats[key] = max(cats.get(key, 0.0), abs(v))
    rep("residual four-fermion vertex at fixed psi: max|coefficient|", cats.get((4, 0, 0), 0.0))
    forbidden = {k: v for k, v in cats.items() if k not in ALLOWED}
    rep(f"all forbidden coefficients up to total degree {D}: max|coefficient|",
        max(forbidden.values(), default=0.0))
    print("   allowed couplings present (deg phibar, deg psi_T, deg psi_B): max|coefficient| =",
          {k: round(v, 6) for k, v in sorted(cats.items()) if k in ALLOWED})
    print(f"\n{len(res)} checks, {sum(1 for ok in res if not ok)} failed ({time.time() - t0:.0f}s)")
