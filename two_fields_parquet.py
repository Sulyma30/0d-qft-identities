from two_fields import *
w = lambda a, b: 1 if a == b else 2

def parquet_and_appendix(th, J, tag, D=4, even=True):
    t = time.time()
    C = composite2(th, J, D)
    G1, _ = onepi(th, J, D, C['mom'])
    I, chi, Gpsi = C['I'], C['chi'], C['Gpsi']
    K = joint_cumulants2(FIELDS, C['mom'], 2)
    Gm = mp.matrix(2, 2)
    for a in range(2):
        for b in range(2):
            e = [0, 0]; e[a] += 1; e[b] += 1
            Gm[a, b] = -cumulant(K, e)
    G4 = lambda a, b, c, d: G1.d([a, b, c, d])
    G3 = lambda a, b, c: G1.d([a, b, c])
    rng = range(2)
    pre = f"[{tag}] "
    # path G^ab(phibar)
    path = implicit([Gpsi.deriv(2 + i) for i in range(3)], [2, 3, 4])
    for (a, b) in PAIRS:
        for (c, d) in PAIRS:
            # (B4): chi^(ab)(cd) = d2G^ab/dphib^c' dphib^d' G^c'c G^d'd - G^ad G^bc - G^ac G^bd
            lhs = chi[pair_index(a, b), pair_index(c, d)]
            p = path[2 + pair_index(a, b)]
            rhs = sum(p.d([cp, dp]) * Gm[cp, c] * Gm[dp, d] for cp in rng for dp in rng) - Gm[a, d] * Gm[b, c] - Gm[a, c] * Gm[b, d]
            rep(pre + f"(B4) chi^({a}{b})({c}{d})", lhs, rhs)
            if even:
                rhs32 = sum(Gm[a, ap] * Gm[b, bp] * G4(ap, bp, cp, dp) * Gm[cp, c] * Gm[dp, d]
                            for ap in rng for bp in rng for cp in rng for dp in rng) - Gm[a, c] * Gm[b, d] - Gm[a, d] * Gm[b, c]
                rep(pre + f"(32) chi^({a}{b})({c}{d})", lhs, rhs32)
    for (c, d) in PAIRS:
        for (a, b) in [(0, 0), (0, 1), (1, 1)]:
            p = path[2 + pair_index(c, d)]
            rhs = sum((G4(a, b, h, k) + sum(G3(a, h, f) * Gm[f, e] * G3(e, b, k) + G3(a, k, f) * Gm[f, e] * G3(e, b, h)
                                            for f in rng for e in rng)) * Gm[h, c] * Gm[k, d] for h in rng for k in rng)
            rep(pre + f"(B5) d2G^{c}{d}/dphib_{a} dphib_{b}", p.d([a, b]), rhs)
    if not even:
        print(f"   ({time.time() - t:.0f}s)"); return
    Iord = lambda x, y, ap, bp: I((x, y), (pair_index(ap, bp),)) / w(ap, bp)
    for idx in [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1)]:
        tree = sum(Iord(x, y, ap, bp) * Gm[ap, cp] * Gm[bp, dp] * G4(cp, dp, z, ww)
                   for (x, y), (z, ww) in pairings(idx) for ap in rng for bp in rng for cp in rng for dp in rng)
        rep(pre + f"(37) parquet, ordered sums, Gamma_{''.join(map(str, idx))}", G4(*idx), I(idx) + tree)
    for (a, b) in PAIRS:
        for (c, d) in PAIRS:
            lhs = sum(G4(a, b, cp, dp) * Gm[cp, c] * Gm[dp, d] for cp in rng for dp in rng)
            rhs = sum(Iord(a, b, cp, dp) * chi[pair_index(cp, dp), pair_index(c, d)] for cp in rng for dp in rng)
            rep(pre + f"(33) BSE, ordered sums, ({a}{b})({c}{d})", lhs, rhs)
    # (B9): d^2 Gamma/dG^ab dG^c'd' G^c'c G^d'd = -1/2 I^3PI_ab(c'd') G^c'c G^d'd + 1/2 delta^cd_ab,
    # delta^cd_ab = 1/2 (delta^c_a delta^d_b + delta^d_a delta^c_b); symmetric-derivative convention for d/dG^ab.
    # For an even theory I^3PI_2(2) = I^2PI_2(2) (checked in single_field.py), so the 2PI action is used here.
    Gt = lambda a, ap, b, bp: Gpsi.d([2 + pair_index(a, ap), 2 + pair_index(b, bp)]) / (w(a, ap) * w(b, bp))
    for a, b, c, d in product(rng, repeat=4):
        lhs = sum(Gt(a, b, cp, dp) * Gm[cp, c] * Gm[dp, d] for cp in rng for dp in rng)
        rhs = (-sum(Iord(a, b, cp, dp) * Gm[cp, c] * Gm[dp, d] for cp in rng for dp in rng) / 2
               + ((a == c) * (b == d) + (a == d) * (b == c)) / mp.mpf(4))
        rep(pre + f"(B9) tensor form, a b c d = {a}{b}{c}{d}", lhs, rhs)
    print(f"   ({time.time() - t:.0f}s)")


th_odd = Theory2(S2o, S3o, S4)
th_even = Theory2(S2o, {}, S4)
parquet_and_appendix(th_even, (mp.mpf(0), mp.mpf(0)), "even")
parquet_and_appendix(th_odd, (mp.mpf('0.1'), mp.mpf('-0.2')), "odd", even=False)
nf = sum(1 for _, ok in res if not ok)
print(f"{len(res)} checks, {nf} failed")
