"""Two real fields (0D), generic couplings, to test tensor/index structure.
S = 1/2 S_ab f^a f^b + 1/6 S_abc f^a f^b f^c + 1/24 S_abcd f^a f^b f^c f^d."""
import mpmath as mp
from itertools import product, permutations
from math import factorial
from mpmath.calculus.quadrature import GaussLegendre
from zerod import Series, joint_cumulants, cumulant, legendre, implicit, W_series
import zerod

mp.mp.dps = 30
N = 2


def sym_tensor(vals, rank):
    """vals: dict sorted-index-tuple -> value; returns full symmetric dict."""
    T = {}
    for idx in product(range(N), repeat=rank):
        T[idx] = mp.mpf(vals.get(tuple(sorted(idx)), 0))
    return T


class Theory2:
    def __init__(self, S2, S3, S4):
        self.S2 = sym_tensor(S2, 2)
        self.S3 = sym_tensor(S3, 3)
        self.S4 = sym_tensor(S4, 4)
        # polynomial dict {(i,j): coeff} of S
        P = {}
        for rank, T in ((2, self.S2), (3, self.S3), (4, self.S4)):
            for idx, v in T.items():
                e = (idx.count(0), idx.count(1))
                P[e] = P.get(e, 0) + v / factorial(rank)
        self.P = P

    def S(self, x, y):
        return sum(c * x**i * y**j for (i, j), c in self.P.items())

    def Sn(self, n, phib):
        """n-th derivative tensor of S at phib (dict idx->value)."""
        out = {}
        for idx in product(range(N), repeat=n):
            v = mp.mpf(0)
            for rank, T in ((2, self.S2), (3, self.S3), (4, self.S4)):
                k = rank - n
                if k < 0:
                    continue
                for rest in product(range(N), repeat=k):
                    w = T[tuple(idx) + rest]
                    for r in rest:
                        w *= phib[r]
                    v += w / factorial(k)
            out[idx] = v
        return out

    def moments(self, J, kmax, L=7, deg=7):
        gl = GaussLegendre(mp.mp)
        nodes = gl.calc_nodes(deg, mp.mp.prec)
        xs = [(L * x, L * w) for x, w in nodes]
        M = {}
        Z = mp.mpf(0)
        for x, wx in xs:
            xp = [mp.mpf(1)]
            for _ in range(kmax):
                xp.append(xp[-1] * x)
            for y, wy in xs:
                w = wx * wy * mp.exp(-self.S(x, y) - J[0] * x - J[1] * y)
                if w < mp.mpf(10) ** (-mp.mp.dps - 5):
                    continue
                Z += w
                yp = w
                for j in range(kmax + 1):
                    for i in range(kmax + 1 - j):
                        M[(i, j)] = M.get((i, j), 0) + xp[i] * yp
                    yp *= y
        return Z, {k: v / Z for k, v in M.items()}


# 2-variable polynomials as dicts
def p2mul(a, b):
    r = {}
    for (i, j), u in a.items():
        for (k, l), v in b.items():
            r[(i + k, j + l)] = r.get((i + k, j + l), 0) + u * v
    return r


def joint_cumulants2(ops, mom, D):
    d = len(ops)
    M = Series(d, D)
    cache = {}

    def opow(i, n):
        if (i, n) not in cache:
            cache[(i, n)] = {(0, 0): mp.mpf(1)} if n == 0 else p2mul(opow(i, n - 1), ops[i])
        return cache[(i, n)]

    for e in product(range(D + 1), repeat=d):
        if sum(e) > D:
            continue
        p = {(0, 0): mp.mpf(1)}
        f = 1
        for i, n in enumerate(e):
            if n:
                p = p2mul(p, opow(i, n))
            f *= factorial(n)
        M.c[e] = sum(c * mom[k] for k, c in p.items()) / f
    return zerod.series_log1p(M - 1)


def W_series2(ops, mom, D):
    K = joint_cumulants2(ops, mom, D)
    return Series(K.nv, D, {e: -v * (-1) ** sum(e) for e, v in K.c.items() if sum(e) > 0})


X = {(1, 0): mp.mpf(1)}
Y = {(0, 1): mp.mpf(1)}
FIELDS = [X, Y]
PAIRS = [(0, 0), (0, 1), (1, 1)]          # independent composite components


def pair_index(a, b):
    return PAIRS.index(tuple(sorted((a, b))))
