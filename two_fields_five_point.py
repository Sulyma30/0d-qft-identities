from two_fields import *
from itertools import combinations

def five_point(th, J, tag):
    t = time.time()
    C = composite2(th, J, 5)
    G1, phib = onepi(th, J, 5, C['mom'])
    I, chi = C['I'], C['chi']
    print(f"   composite action built ({time.time()-t:.0f}s)"); sys.stdout.flush()
    for idx in [(0, 0, 0, 0, 1), (0, 0, 1, 1, 1), (0, 1, 1, 1, 1)]:
        pos = range(5)
        t10 = sum(I(tuple(idx[i] for i in T), (b,)) * chi[b, g] * I(tuple(idx[i] for i in pos if i not in T), (g,))
                  for T in combinations(pos, 3) for b in range(3) for g in range(3))
        t15 = mp.mpf(0)
        for s in pos:
            rest = [i for i in pos if i != s]
            for (p1, p2) in [((rest[0], rest[1]), (rest[2], rest[3])), ((rest[0], rest[2]), (rest[1], rest[3])),
                             ((rest[0], rest[3]), (rest[1], rest[2]))]:
                for b, g, e, f in product(range(3), repeat=4):
                    t15 += (I((idx[p1[0]], idx[p1[1]]), (b,)) * chi[b, g] * I((idx[s],), (g, e)) * chi[e, f]
                            * I((idx[p2[0]], idx[p2[1]]), (f,)))
        rep(f"[{tag}] (81) Gamma_{''.join(map(str, idx))}", G1.d(list(idx)), I(idx) + t10 + t15)
    print(f"   ({time.time()-t:.0f}s)")

th_odd = Theory2(S2o, S3o, S4)
five_point(th_odd, (mp.mpf('0.1'), mp.mpf('-0.2')), "odd")
nf = sum(1 for _, ok in res if not ok)
print(f"{len(res)} checks, {nf} failed")
