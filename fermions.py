"""Two-flavour Grassmann theory in 0D: exact checks of Sec. III A (SBE) with fermionic signs.
Fields phi^a: a=0 -> f_1, 1 -> f_2, 2 -> f^dag_1, 3 -> f^dag_2.
S = f^dag_i h_ij f_j + 1/4 u_ijkl f^dag_i f^dag_j f_k f_l ,   u_1212 = U (antisymmetrised).
Composites (Eq. 60): phi^ph = Uph_ij f^dag_i f_j, phi^pp = 1/2 Upp_ij f_i f_j, phi^ppbar = 1/2 Uppb_ij f^dag_i f^dag_j,
HS term (Eq. 61): +1/2 chi0ph J_ph^2 + chi0pp J_pp J_ppbar."""
from superalg import SA, sa_exp, sa_log1p, merge_sign
from itertools import product
import numpy as np
import sys

NF = 2
F = lambda i: i
FD = lambda i: NF + i
SRC = lambda a: 2 * NF + a
zeta = -1
res = []


def rep(label, lhs, rhs, tol=1e-10):
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    ok = err < tol
    res.append((label, ok))
    print(f"{'OK ' if ok else 'FAIL'} {label:60s} {lhs:+.12f} {rhs:+.12f} err={err:.1e}")
    sys.stdout.flush()


class Model:
    def __init__(self, h, u, Uph, Upp, Uppb, chi0ph, chi0pp):
        self.h, self.u = np.array(h), np.array(u)
        self.Uph, self.Upp, self.Uppb = np.array(Uph), np.array(Upp), np.array(Uppb)
        self.chi0ph, self.chi0pp = chi0ph, chi0pp

    def generating(self, composites, D):
        """W(J, k) as SA: bosonic vars = composite sources, Grassmann gens 4..7 = sources J_a."""
        nb = 3 if composites else 1
        f = lambda g: SA.fvar(nb, D, g)
        E = SA(nb, D)
        for i, j in product(range(NF), repeat=2):
            E = E - f(FD(i)) * f(F(j)) * self.h[i, j]
        for i, j, k, l in product(range(NF), repeat=4):
            if self.u[i, j, k, l]:
                E = E - f(FD(i)) * f(FD(j)) * f(F(k)) * f(F(l)) * (self.u[i, j, k, l] / 4)
        for a in range(2 * NF):
            E = E - f(SRC(a)) * f(a)
        if composites:
            k = [SA.bvar(nb, D, i) for i in range(3)]
            ph = SA(nb, D); pp = SA(nb, D); ppb = SA(nb, D)
            for i, j in product(range(NF), repeat=2):
                ph = ph + f(FD(i)) * f(F(j)) * self.Uph[i, j]
                pp = pp + f(F(i)) * f(F(j)) * (self.Upp[i, j] / 2)
                ppb = ppb + f(FD(i)) * f(FD(j)) * (self.Uppb[i, j] / 2)
            E = E - k[0] * ph - k[1] * pp - k[2] * ppb
            E = E - k[0] * k[0] * (self.chi0ph / 2) - k[1] * k[2] * self.chi0pp
        Zfull = sa_exp(E, 4 * NF + D)
        # Berezin integration: keep coefficient of f0 f1 f2 f3 (fixed ordering convention)
        Z = SA(nb, D)
        for (b, fs), v in Zfull.c.items():
            if set(range(2 * NF)) <= set(fs):
                rest = tuple(g for g in fs if g >= 2 * NF)
                s, _ = merge_sign(tuple(range(2 * NF)), rest)
                Z.c[(b, rest)] = Z.c.get((b, rest), 0.0) + s * v
        Z0 = Z.const_term()
        W = -sa_log1p(Z * (1 / Z0) - 1, 2 * NF + D)
        return W, nb, Z0

    def expect(self, ops_gens_coeffs):
        """<phi^a phi^b ...> at J=0 for a product of field generators (list of generator ids)."""
        pass


def legendre_ferm(W, nb, D, composites):
    """Gamma(phibar, psi) as SA in new algebra: Grassmann gens 0..3 = phibar^a, bosonic = delta psi."""
    X = [W.dL(SRC(a)) for a in range(2 * NF)]
    Y = [W.db(i) for i in range(nb)] if composites else []
    # linear coefficients
    M = np.zeros((2 * NF, 2 * NF))
    for a in range(2 * NF):
        for b in range(2 * NF):
            M[a, b] = X[a].c.get(((0,) * nb, (SRC(b),)), 0.0)
    Mi = np.linalg.inv(M)
    phi0 = [y.const_term() for y in Y]
    B = np.zeros((nb, nb))
    for i in range(len(Y)):
        for j in range(nb):
            e = [0] * nb; e[j] = 1
            B[i, j] = Y[i].c.get((tuple(e), ()), 0.0)
    Bi = np.linalg.inv(B) if composites else None
    pb = [SA.fvar(nb, D, a) for a in range(2 * NF)]            # phibar^a
    dps = [SA.bvar(nb, D, i) for i in range(nb)]          # delta psi
    J = [sum((pb[b] * Mi[a, b] for b in range(2 * NF)), SA(nb, D)) for a in range(2 * NF)]
    K = [sum((dps[j] * Bi[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)] if composites else [SA(nb, D)]
    srcJ = [SA.fvar(nb, D, SRC(b)) for b in range(2 * NF)]
    kv = [SA.bvar(nb, D, i) for i in range(nb)]
    Nlin = [X[a] - sum((srcJ[b] * M[a, b] for b in range(2 * NF)), SA(nb, D)) for a in range(2 * NF)]
    Rlin = [Y[i] - phi0[i] - sum((kv[j] * B[i, j] for j in range(nb)), SA(nb, D)) for i in range(len(Y))]
    for _ in range(D + 2 * NF):
        fs = {SRC(b): J[b] for b in range(2 * NF)}
        bs = K if composites else [SA(nb, D)]
        NJ = [n.substitute(bs, fs, nb) for n in Nlin]
        J = [sum(((pb[b] - NJ[b]) * Mi[a, b] for b in range(2 * NF)), SA(nb, D)) for a in range(2 * NF)]
        if composites:
            RK = [r.substitute(bs, fs, nb) for r in Rlin]
            K = [sum(((dps[j] - RK[j]) * Bi[i, j] for j in range(nb)), SA(nb, D)) for i in range(nb)]
    fs = {SRC(b): J[b] for b in range(2 * NF)}
    G = W.substitute(K if composites else [SA(nb, D)], fs, nb)
    for a in range(2 * NF):
        G = G - J[a] * pb[a]
    for i in range(len(Y)):
        G = G - K[i] * (dps[i] + phi0[i])
    return G


def dvals(G, ferm, bos=()):
    """delta/delta phibar^{ferm[0]} ... delta/delta phibar^{ferm[-1]} delta/delta psi... Gamma at 0."""
    x = G
    for i in bos:
        x = x.db(i)
    for a in reversed(ferm):
        x = x.dL(a)
    return x.const_term()


def run(model, D=3):
    W1, nb1, _ = model.generating(False, D)
    G1 = legendre_ferm(W1, nb1, D, False)
    Wc, nbc, _ = model.generating(True, D)
    Gc = legendre_ferm(Wc, nbc, D, True)
    # ---- fermionic 1PI basics
    Gprop = np.array([[W1.dL(SRC(b)).dL(SRC(a)).const_term() for b in range(2 * NF)] for a in range(2 * NF)])
    # note: delta/delta J_a delta/delta J_b W = dL_a(dL_b W)
    Gprop = np.array([[W1.dL(SRC(b)).dL(SRC(a)).const_term() for b in range(2 * NF)] for a in range(2 * NF)])
    G2 = np.array([[dvals(G1, [b, c]) for c in range(2 * NF)] for b in range(2 * NF)])
    prod_ = Gprop @ G2
    for a in range(2 * NF):
        for c in range(2 * NF):
            rep(f"(11) sum_b G^ab Gamma_bc = -zeta delta  [a={a},c={c}]", prod_[a, c], -zeta * (a == c))
    # ---- composite pieces (even theory: delta psi/delta phibar = 0 at phibar = 0, so I = partial derivatives)
    H = np.array([[dvals(Gc, [], (i, j)) for j in range(3)] for i in range(3)])
    chi = -np.linalg.inv(H)
    Iab = lambda a, b, beta: dvals(Gc, [a, b], (beta,))
    G4res = lambda a, b, c, d: dvals(Gc, [a, b, c, d])
    G41 = lambda a, b, c, d: dvals(G1, [a, b, c, d])
    PH, PP, PPB = 0, 1, 2
    Iph = lambda i, j: Iab(FD(i), F(j), PH)       # delta/delta fbar^dag_i delta^2 Gamma / delta fbar_j delta psi^ph
    Ipp = lambda i, j: Iab(FD(i), FD(j), PP)
    Ippb = lambda i, j: Iab(F(i), F(j), PPB)
    print("chi matrix:\n", chi)
    for n, k, l, m in product(range(NF), repeat=4):
        lhs = G41(FD(n), FD(k), F(l), F(m))
        if (n, k, l, m) == (0, 1, 0, 1):
            print("   channel terms:", G4res(FD(n), FD(k), F(l), F(m)), Ipp(n, k) * chi[PP, PPB] * Ippb(l, m),
                  Iph(k, l) * chi[PH, PH] * Iph(n, m), -Iph(n, l) * chi[PH, PH] * Iph(k, m))
        rhs63 = (G4res(FD(n), FD(k), F(l), F(m)) + Ipp(n, k) * chi[PP, PPB] * Ippb(l, m)
                 + Iph(k, l) * chi[PH, PH] * Iph(n, m) - Iph(n, l) * chi[PH, PH] * Iph(k, m))
        rep(f"(63) SBE decomposition Gamma1PI_nklm, nklm={n}{k}{l}{m}", lhs, rhs63)
        a, b, c, d = FD(n), FD(k), F(l), F(m)
        t = sum(Iab(a, b, be) * chi[be, ga] * Iab(c, d, ga) + zeta * Iab(a, c, be) * chi[be, ga] * Iab(b, d, ga)
                + Iab(a, d, be) * chi[be, ga] * Iab(b, c, ga) for be in range(3) for ga in range(3))
        rep(f"(77) general form with zeta-signed P_3, nklm={n}{k}{l}{m}", lhs, G4res(a, b, c, d) + t)
    # ---- (67) Schwinger-Dyson equation for chi^ph
    chi0 = model.chi0ph
    I0 = -model.Uph / chi0
    Gfm = lambda m_, i: Gprop[F(m_), FD(i)]          # G^{mi}: a = f_m, b = f^dag_i
    Pi = sum(I0[i, j] * Gfm(m_, i) * Gfm(j, n) * Iph(n, m_) for i, j, n, m_ in product(range(NF), repeat=4))
    rep("(67) chi^ph = chi0 + chi0 Pi chi^ph", chi[PH, PH], chi0 + chi0 * Pi * chi[PH, PH])
    # ---- (68) SD equation for the Yukawa vertex
    for k, l in product(range(NF), repeat=2):
        def T(n, m_):
            a, b, c, d = FD(n), FD(k), F(l), F(m_)
            return G41(a, b, c, d) - sum(Iab(b, c, be) * chi[be, ga] * Iab(a, d, ga) for be in range(3) for ga in range(3))
        rhs = I0[k, l] + sum(T(n, m_) * Gfm(m_, i) * Gfm(j, n) * I0[i, j] for n, m_, i, j in product(range(NF), repeat=4))
        rep(f"(68) Yukawa vertex I^ph_kl, kl={k}{l}", Iph(k, l), rhs)
    return Gprop, G1, Gc, chi


def antisym_u(NFl, vals):
    u = np.zeros((NFl,) * 4)
    for (i, j, k, l), v in vals.items():
        for (a, b, sab) in ((i, j, 1), (j, i, -1)):
            for (c, d, scd) in ((k, l, 1), (l, k, -1)):
                u[a, b, c, d] += sab * scd * v
    return u


if __name__ == '__main__':
    model = Model(h=[[0.3, 0.2], [0.2, -0.5]], u=antisym_u(2, {(0, 1, 0, 1): 1.1}), Uph=[[0.7, 0.3], [0.3, -0.4]],
                  Upp=[[0, 0.9], [-0.9, 0]], Uppb=[[0, 0.6], [-0.6, 0]], chi0ph=0.5, chi0pp=0.8)
    run(model)
    nf = sum(1 for _, ok in res if not ok)
    print(f"{len(res)} checks, {nf} failed")
