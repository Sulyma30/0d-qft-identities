from twod import *
import time, sys

res = []


def rep(label, lhs, rhs, tol=mp.mpf('1e-18')):
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), mp.mpf(1))
    ok = err < tol
    res.append((label, ok))
    print(f"{'OK ' if ok else 'FAIL'} {label:62s} {mp.nstr(lhs, 12):>18s} {mp.nstr(rhs, 12):>18s} err={mp.nstr(err, 3)}")
    sys.stdout.flush()


def pairings(idx):
    """3 ways to split 4 positions into two pairs: returns list of ((i,j),(k,l)) position tuples."""
    a, b, c, d = idx
    return [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]


S2o = {(0, 0): 1.0, (0, 1): 0.3, (1, 1): 1.4}
S3o = {(0, 0, 0): 0.5, (0, 0, 1): -0.3, (0, 1, 1): 0.2, (1, 1, 1): 0.4}
S4 = {(0, 0, 0, 0): 1.2, (0, 0, 0, 1): 0.2, (0, 0, 1, 1): 0.5, (0, 1, 1, 1): -0.15, (1, 1, 1, 1): 1.5}


def onepi(th, J, D, mom):
    W = W_series2(FIELDS, mom, D)
    G1, X0 = legendre(W, [0, 1], list(J))
    return G1, X0


def estimator_tensor(th, J, tag):
    t = time.time()
    Z, mom = th.moments(J, 16)
    G1, phib = onepi(th, J, 4, mom)
    Gam = lambda *i: G1.d(list(i))
    K = joint_cumulants2(FIELDS, mom, 2)
    Gm = mp.matrix(2, 2)
    for a in range(2):
        for b in range(2):
            e = [0, 0]; e[a] += 1; e[b] += 1
            Gm[a, b] = -cumulant(K, e)
    Gi = Gm**-1
    Sig = -Gi - mp.matrix([[th.S2[(0, 0)], th.S2[(0, 1)]], [th.S2[(1, 0)], th.S2[(1, 1)]]])
    # composite operators
    dSint = []
    for a in range(2):
        p = {}
        for b, c in product(range(2), repeat=2):
            p = {**p}
            q = p2mul(FIELDS[b], FIELDS[c])
            for k, v in q.items():
                p[k] = p.get(k, 0) + th.S3[(a, b, c)] / 2 * v
        for b, c, d in product(range(2), repeat=3):
            q = p2mul(p2mul(FIELDS[b], FIELDS[c]), FIELDS[d])
            for k, v in q.items():
                p[k] = p.get(k, 0) + th.S4[(a, b, c, d)] / 6 * v
        dSint.append(p)
    Q = []
    for a in range(2):
        p = dict(dSint[a])
        for b in range(2):
            for k, v in FIELDS[b].items():
                p[k] = p.get(k, 0) - Sig[a, b] * v
        Q.append(p)
    R = []
    for (a, b) in PAIRS:
        p = {}
        for c, d in product(range(2), repeat=2):
            q = p2mul(FIELDS[c], FIELDS[d])
            for k, v in q.items():
                p[k] = p.get(k, 0) + th.S4[(a, b, c, d)] / 2 * v
        R.append(p)
    ops = FIELDS + R + Q              # indices: phi 0,1 ; R 2,3,4 ; Q 5,6
    Kc = joint_cumulants2(ops, mom, 4)
    PH = lambda a: a
    RR = lambda a, b: 2 + pair_index(a, b)
    QQ = lambda a: 5 + a

    def Gf(*opidx):
        e = [0] * len(ops)
        for o in opidx:
            e[o] += 1
        return (-1) ** (len(opidx) - 1) * cumulant(Kc, e)

    S2b, S3b, S4b = th.Sn(2, phib), th.Sn(3, phib), th.Sn(4, phib)
    pre = f"[{tag}] "
    for a, b in [(0, 0), (0, 1), (1, 1)]:
        rhs = S2b[(a, b)] + Gf(QQ(a), QQ(b)) + Gf(RR(a, b)) - sum(
            th.S4[(a, b, c, d)] * phib[c] * phib[d] for c in range(2) for d in range(2)) / 2
        rep(pre + f"(119) Gamma_{a}{b}", Gam(a, b), rhs)
    for idx in [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]:
        a, b, c = idx
        rhs = S3b[idx] + Gf(RR(a, b), QQ(c)) + Gf(RR(a, c), QQ(b)) + Gf(RR(b, c), QQ(a)) + Gf(QQ(a), QQ(b), QQ(c))
        rep(pre + f"(114) Gamma_{a}{b}{c}", Gam(a, b, c), rhs)

    def L12(e, x, y):
        return Gf(PH(e), QQ(x), QQ(y)) + Gf(PH(e), RR(x, y))

    for idx in [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]:
        L4 = Gf(*[QQ(i) for i in idx])
        for (p, q) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            rest = [idx[r] for r in range(4) if r not in (p, q)]
            L4 += Gf(RR(idx[p], idx[q]), QQ(rest[0]), QQ(rest[1]))
        for (x, y), (z, w) in pairings(idx):
            L4 += Gf(RR(x, y), RR(z, w))
        tree = sum(L12(e, x, y) * Gi[e, f] * L12(f, z, w)
                   for (x, y), (z, w) in pairings(idx) for e in range(2) for f in range(2))
        rep(pre + f"(115) Gamma_{''.join(map(str, idx))}", Gam(*idx), S4b[idx] + L4 - tree)
    print(f"   ({time.time() - t:.0f}s)")


def composite2(th, J, D):
    Z, mom = th.moments(J, 2 * D + 2)
    ops = FIELDS + [p2mul(FIELDS[a], FIELDS[b]) for a, b in PAIRS]
    W = W_series2(ops, mom, D)
    Gphi, X0 = legendre(W, range(5), list(J) + [0, 0, 0])
    phi0 = X0[:2]
    x = [Series.var(5, D, i) for i in range(5)]
    subs = x[:2]
    for k, (a, b) in enumerate(PAIRS):     # phibar^(ab) = phib_a phib_b - G^ab
        subs.append(x[a] * phi0[b] + x[b] * phi0[a] + x[a] * x[b] - x[2 + k])
    Gpsi = Gphi.compose(subs, 5)
    H = mp.matrix(3, 3)
    for i in range(3):
        for j in range(3):
            H[i, j] = Gpsi.d([2 + i, 2 + j])
    chi = -(H**-1)
    v = [[sum(Gpsi.d([a, 2 + g]) * chi[g, b] for g in range(3)) for b in range(3)] for a in range(2)]

    def I(fund, ext=()):
        """I_{fund..., ext...}: fund = fundamental external indices, ext = composite indices (0..2)."""
        tot = mp.mpf(0)
        choices = [[('f', a)] + [('c', b) for b in range(3)] for a in fund]
        for legs in product(*choices):
            w = mp.mpf(1)
            vars_ = []
            for (kind, j), a in zip(legs, fund):
                if kind == 'f':
                    vars_.append(a)
                else:
                    w *= v[a][j]
                    vars_.append(2 + j)
            tot += w * Gpsi.d(vars_ + [2 + e for e in ext])
        return tot

    return dict(Gpsi=Gpsi, chi=chi, I=I, mom=mom, phi0=phi0)


def tree_tensor(th, J, tag, D=4):
    t = time.time()
    C = composite2(th, J, D)
    G1, _ = onepi(th, J, D, C['mom'])
    I, chi = C['I'], C['chi']
    pre = f"[{tag}] "
    for idx in [(0, 1), (1, 1)]:
        rep(pre + f"(72) Gamma_{''.join(map(str, idx))} = I", G1.d(list(idx)), I(idx))
    for idx in [(0, 0, 1), (0, 1, 1)]:
        rep(pre + f"(72) Gamma_{''.join(map(str, idx))} = I", G1.d(list(idx)), I(idx))
    for idx in [(0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1)]:
        tree = sum(I((x, y), (b,)) * chi[b, g] * I((z, w), (g,))
                   for (x, y), (z, w) in pairings(idx) for b in range(3) for g in range(3))
        rep(pre + f"(77) Gamma_{''.join(map(str, idx))}", G1.d(list(idx)), I(idx) + tree)
    print(f"   ({time.time() - t:.0f}s)")
    return C, G1


if __name__ == '__main__':
    th_odd = Theory2(S2o, S3o, S4)
    th_even = Theory2(S2o, {}, S4)
    print("=== two-component theory: tensor estimators ===")
    estimator_tensor(th_odd, (mp.mpf('0.1'), mp.mpf('-0.2')), "odd")
    estimator_tensor(th_even, (mp.mpf(0), mp.mpf(0)), "even")
    print("=== two-component theory: composite tree expansion, psi = G^ab ===")
    tree_tensor(th_odd, (mp.mpf('0.1'), mp.mpf('-0.2')), "odd")
    nf = sum(1 for _, ok in res if not ok)
    print(f"{len(res)} checks, {nf} failed")
