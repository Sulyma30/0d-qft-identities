from zerod import *
import sys

results = []


def report(label, lhs, rhs):
    lhs, rhs = mp.mpf(lhs), mp.mpf(rhs)
    err = abs(lhs - rhs) / max(abs(rhs), abs(lhs), mp.mpf(1))
    ok = err < mp.mpf('1e-20')
    results.append((label, ok))
    print(f"{'OK ' if ok else 'FAIL'} {label:58s} lhs={mp.nstr(lhs, 14):>22s} rhs={mp.nstr(rhs, 14):>22s} rel.err={mp.nstr(err, 3)}")


PHI = [mp.mpf(0), mp.mpf(1)]


def one_pi(th, J0, D, mom):
    W = W_series([PHI], mom, D)
    G1, X0 = legendre(W, [0], [J0])
    return [G1.d([0] * n) for n in range(D + 1)], X0[0]


def kappas(mom, D):
    K = joint_cumulants([PHI], mom, D)
    return [cumulant(K, (n,)) for n in range(D + 1)]


# =================================================================== A. Eq. (2a), (21)
print("\n=== A. Background-field structure, Eqs. (2a) and (21) ===")
th = Theory(1.0, 0.7, 1.2)
phib = mp.mpf('0.3')
J = th.J_for_mean(phib)
Z, mom = th.moments(J, 6)
Gam = -mp.log(Z) - J * phib
# shifted theory: couplings S2[phib], S3[phib], same lambda; evaluate Gamma at eta-bar = 0
ths = Theory(th.dS(2, phib), th.dS(3, phib), th.lam)
Js = ths.J_for_mean(0)
Zs, moms = ths.moments(Js, 6)
report("(2a): Gamma1PI(phib) - S(phib) = Omega(S2[phib],S3[phib])", Gam - th.S(phib), -mp.log(Zs))
kap = kappas(mom, 4)
G2, G3 = -kap[2], kap[3]
report("(21): Gamma_1 = S_1 - 1/2 S_3[phib] G + 1/6 S_4 G^3", -J,
       th.dS(1, phib) - th.dS(3, phib) * G2 / 2 + th.lam * G3 / 6)


# =================================================================== B. estimators
def estimator_checks(th, J0, tag, even=False):
    Z, mom = th.moments(J0, 30)
    D = 6
    g1pi, phib = one_pi(th, J0, D, mom)
    kap = kappas(mom, 2)
    G = -kap[2]
    Sig = -1 / G - th.m
    Q = [mp.mpf(0), -Sig, th.g / 2, th.lam / 6]
    R = [mp.mpf(0), mp.mpf(0), th.lam / 2]                 # R = 1/2 S_4 phi phi
    K = joint_cumulants([PHI, R, Q], mom, D)
    Gf = lambda m, k, l: (-1) ** (m + k + l - 1) * cumulant(K, (m, k, l))
    Gi = 1 / G
    L2 = Gf(0, 0, 2) + Gf(0, 1, 0)
    L3 = 3 * Gf(0, 1, 1) + Gf(0, 0, 3)
    L4 = Gf(0, 0, 4) + 6 * Gf(0, 1, 2) + 3 * Gf(0, 2, 0)
    L5 = Gf(0, 0, 5) + 10 * Gf(0, 1, 3) + 15 * Gf(0, 2, 1)
    L6 = Gf(0, 0, 6) + 15 * Gf(0, 1, 4) + 45 * Gf(0, 2, 2) + 15 * Gf(0, 3, 0)
    L12 = Gf(1, 0, 2) + Gf(1, 1, 0)
    L13 = Gf(1, 0, 3) + 3 * Gf(1, 1, 1)
    L14 = Gf(1, 0, 4) + 6 * Gf(1, 1, 2) + 3 * Gf(1, 2, 0)
    L21 = Gf(2, 0, 1)
    L22 = Gf(2, 0, 2) + Gf(2, 1, 0)
    L30 = Gf(3, 0, 0)
    pre = f"[{tag}] "
    report(pre + "(119) Gamma_2 = S_2 + L_2 - 1/2 S_4 phib^2", g1pi[2],
           th.dS(2, phib) + L2 - th.lam * phib**2 / 2)
    report(pre + "(114) Gamma_3 = S_3 + L_3", g1pi[3], th.dS(3, phib) + L3)
    report(pre + "(115) Gamma_4 = S_4 + L_4 - 3 L12 G^-1 L12", g1pi[4], th.lam + L4 - 3 * L12 * Gi * L12)
    report(pre + "rule, n=5: L5 - 10 L13 G^-1 L12 + 15 L12 G^-1 L21 G^-1 L12", g1pi[5],
           L5 - 10 * L13 * Gi * L12 + 15 * L12 * Gi * L21 * Gi * L12)
    g6 = (L6 - 15 * L14 * Gi * L12 - 10 * L13 * Gi * L13 + 60 * L13 * Gi * L21 * Gi * L12
          + 45 * L12 * Gi * L22 * Gi * L12 - 15 * L30 * (Gi * L12) ** 3
          - 90 * L12 * Gi * L21 * Gi * L21 * Gi * L12)
    report(pre + "rule, n=6 (full tree incl. odd terms)", g1pi[6], g6)
    if even:
        report(pre + "(118) Gamma_6 = L6 - 10 L13 G^-1 L13 (even)", g1pi[6], L6 - 10 * L13 * Gi * L13)


print("\n=== B. Symmetric estimators, Eqs. (114)-(119) and the rule of Sec. IV B ===")
th_odd = Theory(1.0, 0.7, 1.2)
estimator_checks(th_odd, mp.mpf('0.15'), "odd, phib!=0")
estimator_checks(th_odd, th_odd.J_for_mean(0), "odd, phib=0")
th_even = Theory(1.0, 0.0, 1.2)
estimator_checks(th_even, mp.mpf(0), "even", even=True)


# =================================================================== C. Omega and asymptotic classes
print("\n=== C. Omega[S2,S3] and asymptotic classes, Eqs. (98)-(113) ===")
th = th_odd
J0 = th.J_for_mean(0)
Z, mom = th.moments(J0, 30)
D = 6
ops = [PHI, [0, 0, mp.mpf(1) / 2], [0, 0, 0, mp.mpf(1) / 6]]   # sources: J, delta m, delta g
W = W_series(ops, mom, D)
Gp, X0 = legendre(W, [0], [J0, 0, 0])
nv = 3
# Omega(s2, s3): set phibar deviation to 0, k2 = -2 s2, k3 = 6 s3
z = Series(2, D)
s2, s3 = Series.var(2, D, 0), Series.var(2, D, 1)
Om = Gp.compose([z, s2 * (-2), s3 * 6], 2)
g1pi, _ = one_pi(th, J0, D, mom)
kap = kappas(mom, 3)
G = -kap[2]
report("dOmega/dS_2 = G^2", Om.d([0]), G)
report("dOmega/dS_3 = G^3", Om.d([1]), kap[3])
sp = [-th.g / 2, th.lam / 6]     # dS_bullet/dphibar at phibar = 0
spp = [-th.lam / 2, mp.mpf(0)]   # d^2 S_bullet/dphibar^2


def Kclass(n, m):
    """K^{m-1}_n = (S'')^{n-m} (S')^{2m-n} chi^{bullet m}  (99a)."""
    tot = mp.mpf(0)
    for idx in product(range(2), repeat=m):
        w = 1
        for a in idx[:n - m]:
            w *= spp[a]
        for a in idx[n - m:]:
            w *= sp[a]
        tot += w * Om.d(list(idx))
    return tot


def cnk(n, k):
    return factorial(n) // (2**k * factorial(k) * factorial(n - 2 * k))


for n in range(3, 7):
    tot = th.dS(n, 0) + sum(cnk(n, n - m) * Kclass(n, m) for m in range((n + 1) // 2, n + 1))
    report(f"(98) Gamma_{n} = S_{n} + sum_m P K^(m-1)_{n}", g1pi[n], tot)

Sig = -1 / G - th.m
Q = [mp.mpf(0), -Sig, th.g / 2, th.lam / 6]
R = [mp.mpf(0), mp.mpf(0), th.lam / 2]
K = joint_cumulants([PHI, R, Q], mom, D)
Gf = lambda m, k, l: (-1) ** (m + k + l - 1) * cumulant(K, (m, k, l))
Gi = 1 / G
report("(104) K^1_2 = G_2", Kclass(2, 2), Gf(0, 0, 2))
report("(106) K^2_3 = G_3", Kclass(3, 3), Gf(0, 0, 3))
report("(112) K^1_3 = G_(2)1", Kclass(3, 2), Gf(0, 1, 1))
report("(109) K^3_4 = G_4 - 3 G^1_2 G^-1 G^1_2", Kclass(4, 4), Gf(0, 0, 4) - 3 * Gf(1, 0, 2) ** 2 * Gi)
report("(113) K^1_4 = G_(2)(2) - G^1_(2) G^-1 G^1_(2)", Kclass(4, 2), Gf(0, 2, 0) - Gf(1, 1, 0) ** 2 * Gi)
report("(113) K^2_4 = G_(2)2 - G^1_(2) G^-1 G^1_2", Kclass(4, 3), Gf(0, 1, 2) - Gf(1, 1, 0) * Gf(1, 0, 2) * Gi)
report("core class K^4_5 = tree of G^m_n (n=5)", Kclass(5, 5),
       Gf(0, 0, 5) - 10 * Gf(1, 0, 3) * Gi * Gf(1, 0, 2) + 15 * Gf(1, 0, 2) * Gi * Gf(2, 0, 1) * Gi * Gf(1, 0, 2))


# =================================================================== D. composite-field tree expansions
def composite(th, J0, kind, D=5, U=mp.mpf('0.8'), chi0=mp.mpf('0.5')):
    Z, mom = th.moments(J0, 30)
    if kind == '2PI':
        ops, hs = [PHI, [0, 0, 1]], None
    elif kind == '3PI':
        ops, hs = [PHI, [0, 0, 1], [0, 0, 0, 1]], None
    elif kind == 'HS':
        ops, hs = [PHI, [0, 0, U]], {1: chi0}
    d = len(ops)
    W = W_series(ops, mom, D, hs)
    Gphi, X0 = legendre(W, range(d), [J0] + [0] * (d - 1))
    phi0 = X0[0]
    if kind in ('2PI', '3PI'):
        G20 = phi0**2 - X0[1]
        x = [Series.var(d, D, i) for i in range(d)]
        subs = [x[0], x[0] * (2 * phi0) + x[0] * x[0] - x[1]]
        if kind == '3PI':
            subs.append(x[2] + x[0] * (3 * phi0**2) + x[0] * x[0] * (3 * phi0) + x[0] * x[0] * x[0]
                        - x[1] * (3 * phi0) - x[0] * (3 * G20) - x[0] * x[1] * 3)
        Gpsi = Gphi.compose(subs, d)
    else:
        Gpsi = Gphi
    comps = list(range(1, d))
    H = mp.matrix(len(comps), len(comps))
    for a, i in enumerate(comps):
        for b, j in enumerate(comps):
            H[a, b] = Gpsi.d([i, j])
    chi = -(H**-1)
    v = [sum(Gpsi.d([0, comps[b]]) * chi[b, a] for b in range(len(comps))) for a in range(len(comps))]
    Dv = [mp.mpf(1)] + v

    def I(n, ext=()):
        tot = mp.mpf(0)
        for legs in product(range(d), repeat=n):
            w = 1
            for a in legs:
                w *= Dv[a]
            tot += w * Gpsi.d(list(legs) + list(ext))
        return tot

    path = implicit([Gpsi.deriv(i) for i in comps], comps)
    return dict(Gpsi=Gpsi, chi=chi, comps=comps, I=I, path=path, mom=mom, d=d, D=D)


def tree_checks(th, J0, kind, tag, **kw):
    C = composite(th, J0, kind, **kw)
    g1pi, _ = one_pi(th, J0, C['D'], C['mom'])
    I, chi, comps = C['I'], C['chi'], C['comps']
    nc = len(comps)
    pre = f"[{tag}] "
    # Eq. (1): Gamma1PI = Gamma at stationary point
    sub = [Series.var(1, C['D'], 0)] + [C['path'][i].compose([Series.var(1, C['D'], 0)] + [Series(1, C['D'])] * nc, 1)
                                        for i in comps]
    Gline = C['Gpsi'].compose(sub, 1)
    for n in (2, 3, 4, 5):
        report(pre + f"(1)  Gamma1PI_{n} = d^{n}/dphib^{n} Gamma[phib, psi_phib]", g1pi[n], Gline.d([0] * n))
    report(pre + "(72) Gamma1PI_2 = I_2", g1pi[2], I(2))
    report(pre + "(72) Gamma1PI_3 = I_3", g1pi[3], I(3))
    I2 = [I(2, (c,)) for c in comps]
    I3 = [I(3, (c,)) for c in comps]
    I1 = [[I(1, (c, e)) for e in comps] for c in comps]
    quad = lambda a, b: sum(a[i] * chi[i, j] * b[j] for i in range(nc) for j in range(nc))
    report(pre + "(77) Gamma_4 = I_4 + P_3 I_2. chi I_2.", g1pi[4], I(4) + 3 * quad(I2, I2))
    t15 = sum(I2[a] * chi[a, b] * I1[b][c] * chi[c, e] * I2[e]
              for a in range(nc) for b in range(nc) for c in range(nc) for e in range(nc))
    report(pre + "(81) Gamma_5 = I_5 + P10 I3.chiI2. + P15 I2.chiI1..chiI2.", g1pi[5],
           I(5) + 10 * quad(I3, I2) + 15 * t15)
    for a, c in enumerate(comps):
        p = C['path'][c]
        report(pre + f"(70) dpsi/dphib = Gamma_(phi .) chi  [comp {c}]", p.d([0]),
               sum(C['Gpsi'].d([0, comps[b]]) * chi[b, a] for b in range(nc)))
        report(pre + f"(76) d2psi/dphib2 = I_2. chi  [comp {c}]", p.d([0, 0]),
               sum(I2[b] * chi[b, a] for b in range(nc)))
        rhs = sum((I3[b] + 3 * sum(I2[e] * chi[e, f] * I1[f][b] for e in range(nc) for f in range(nc)))
                  * chi[b, a] for b in range(nc))
        report(pre + f"(82) d3psi/dphib3 = (I3. + P3 I2.chiI1..)chi  [comp {c}]", p.d([0, 0, 0]), rhs)
    return C, g1pi


print("\n=== D. Composite-field tree expansions, Sec. III B (odd theory, phib != 0) ===")
J0 = mp.mpf('0.15')
tree_checks(th_odd, J0, '2PI', "psi=G2")
tree_checks(th_odd, J0, '3PI', "psi=(G2,G3)")
tree_checks(th_odd, J0, 'HS', "HS, U=0.8, chi0=+0.5")
tree_checks(th_odd, J0, 'HS', "HS, U=1.3, chi0=-0.7", U=mp.mpf('1.3'), chi0=mp.mpf('-0.7'))

print("\n=== E. Even theory: Eqs. (32), (33), (37), (38) ===")
C2, g1pi = tree_checks(th_even, mp.mpf(0), '2PI', "even, psi=G2")
C3, _ = tree_checks(th_even, mp.mpf(0), '3PI', "even, psi=(G2,G3)")
kap = kappas(C2['mom'], 4)
G = -kap[2]
chi22 = C2['chi'][0, 0]
G4 = g1pi[4]
report("(32) chi^(2)(2) = G^4 Gamma_4 - 2 G^2", chi22, G**4 * G4 - 2 * G**2)
I2_2PI = C2['I'](2, (1,))
report("(33) Gamma_4 G G = I^2PI_2(2) chi^(2)(2)", G4 * G**2, I2_2PI * chi22)
I4_2PI = C2['I'](4)
report("(37) Gamma_4 = I^2PI_4 + P3 I^2PI_2(2) G G Gamma_4", G4, I4_2PI + 3 * I2_2PI * G**2 * G4)
report("fn.[76] I^2PI_4 = I^3PI_4 (even)", I4_2PI, C3['I'](4))
I33 = C3['I'](3, (2,))
report("(38) I^2PI_4 = S_4 + I^3PI_3(3) G G G Gamma_4", I4_2PI, th_even.lam + I33 * G**3 * G4)

print("\n=== F. Appendix B formulas ===")
# odd theory, 2PI and 3PI
J0 = mp.mpf('0.15')
C2o = composite(th_odd, J0, '2PI')
C3o = composite(th_odd, J0, '3PI')
g1o, _ = one_pi(th_odd, J0, 5, C2o['mom'])
kap = kappas(C2o['mom'], 3)
G = -kap[2]
Gpp = C2o['path'][1].d([0, 0])
report("(B4) chi^(2)(2) = G'' G G - 2 G G  (odd)", C2o['chi'][0, 0], Gpp * G**2 - 2 * G**2)
report("(B5) G'' = (Gamma4 + 2 Gamma3 G Gamma3) G G  (odd)", Gpp, (g1o[4] + 2 * g1o[3]**2 * G) * G**2)
p2, p3 = C3o['path'][1], C3o['path'][2]
report("(B6) chi^(3)(2) = G3'' G G - P2P3 G G' G  (odd)", C3o['chi'][1, 0],
       p3.d([0, 0]) * G**2 - 6 * G * p2.d([0]) * G)
# even theory
C2e = composite(th_even, mp.mpf(0), '2PI', D=6)
C3e = composite(th_even, mp.mpf(0), '3PI')
g1e, _ = one_pi(th_even, mp.mpf(0), 6, C2e['mom'])
kap = kappas(C2e['mom'], 6)
G = -kap[2]
p2, p3 = C3e['path'][1], C3e['path'][2]
Gpp = p2.d([0, 0])
G6 = -kap[6]
report("(B7) chi^(3)(3) G^-3 = G3''' - P3P3 G'' + 6  (even)", C3e['chi'][1, 1] / G**3,
       p3.d([0, 0, 0]) - 9 * Gpp + 6)
report("(B8) G3''' GGG = G6 - G G'' G G'' G  (even, 0D form)", p3.d([0, 0, 0]) * G**3, G6 - G * Gpp * G * Gpp * G)
H22 = C3e['Gpsi'].d([1, 1])
H33 = C3e['Gpsi'].d([2, 2])
I22 = C3e['I'](2, (1,))
report("below (B9): I^2PI_2(2) = I^3PI_2(2)|even", C2e['I'](2, (1,)), I22)
report("(B9) Gamma_(2)(2) G G = -1/2 I^3PI_2(2) G G + 1/2 delta  (even, 0D form)", H22 * G**2, -I22 * G**2 / 2 + mp.mpf(1) / 2)
I123 = C3e['I'](1, (1, 2))
I33 = C3e['I'](3, (2,))
report("(B11) 3 G Gamma_(3)(3) = -G I_1(2)(3) - 1/3 P3 Gamma_(2)(2)  (even)", 3 * G * H33, -G * I123 - H22)
report("(B12) I^3PI_3(3) + P3 (I_1(2)(3) + 3 Gamma_(3)(3)) G'' = 6 Gamma_(3)(3) + 1/6 P6 G^-3  (even)",
       I33 + 3 * (I123 + 3 * H33) * Gpp, 6 * H33 + 1 / G**3)
report("(B13) G'' I_1(2)(3) = -1/3 P3 Gamma4 G I^2PI_2(2) - 1/2 G'' I^3PI_3(3)  (even)", Gpp * I123,
       -g1e[4] * G * I22 - Gpp * I33 / 2)
m = C2e['mom']
report("(28) chi^(3)(3) = -<p3 p3> - <p3 p> G^-1 <p p3>  (even)", C3e['chi'][1, 1], -m[6] - m[4] * (1 / G) * m[4])

nfail = sum(1 for _, ok in results if not ok)
print(f"\n{len(results)} checks total, {nfail} failed")
for l, ok in results:
    if not ok:
        print("  FAILED:", l)
