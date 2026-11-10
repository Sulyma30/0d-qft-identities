"""Truncated power series in bosonic variables with Grassmann generators.
Element: dict {(bexp, fgens): coeff}, bexp = tuple of ints (len nb), fgens = sorted tuple of generator ids.
Truncation: total bosonic degree <= D (Grassmann part is nilpotent)."""
from itertools import combinations


def merge_sign(A, B):
    if set(A) & set(B):
        return 0, None
    inv = 0
    for a in A:
        for b in B:
            if a > b:
                inv += 1
    return (-1 if inv % 2 else 1), tuple(sorted(A + B))


class SA:
    def __init__(self, nb, D, c=None):
        self.nb, self.D = nb, D
        self.c = {} if c is None else {k: v for k, v in c.items() if sum(k[0]) <= D and v != 0}

    @staticmethod
    def const(nb, D, a):
        return SA(nb, D, {((0,) * nb, ()): float(a)})

    @staticmethod
    def bvar(nb, D, i):
        e = [0] * nb
        e[i] = 1
        return SA(nb, D, {(tuple(e), ()): 1.0})

    @staticmethod
    def fvar(nb, D, g):
        return SA(nb, D, {((0,) * nb, (g,)): 1.0})

    def _l(self, o):
        return o if isinstance(o, SA) else SA.const(self.nb, self.D, o)

    def __add__(self, o):
        o = self._l(o)
        c = dict(self.c)
        for k, v in o.c.items():
            c[k] = c.get(k, 0.0) + v
        return SA(self.nb, self.D, c)

    __radd__ = __add__

    def __neg__(self):
        return SA(self.nb, self.D, {k: -v for k, v in self.c.items()})

    def __sub__(self, o):
        return self + (-self._l(o))

    def __rsub__(self, o):
        return self._l(o) - self

    def __mul__(self, o):
        if not isinstance(o, SA):
            return SA(self.nb, self.D, {k: v * o for k, v in self.c.items()})
        c = {}
        for (b1, f1), v1 in self.c.items():
            d1 = sum(b1)
            for (b2, f2), v2 in o.c.items():
                if d1 + sum(b2) > self.D:
                    continue
                s, f = merge_sign(f1, f2)
                if s == 0:
                    continue
                k = (tuple(x + y for x, y in zip(b1, b2)), f)
                c[k] = c.get(k, 0.0) + s * v1 * v2
        return SA(self.nb, self.D, c)

    __rmul__ = __mul__

    def const_term(self):
        return self.c.get(((0,) * self.nb, ()), 0.0)

    def dL(self, g):
        """Left derivative w.r.t. Grassmann generator g."""
        c = {}
        for (b, f), v in self.c.items():
            if g in f:
                p = f.index(g)
                k = (b, f[:p] + f[p + 1:])
                c[k] = c.get(k, 0.0) + (-v if p % 2 else v)
        return SA(self.nb, self.D, c)

    def db(self, i):
        c = {}
        for (b, f), v in self.c.items():
            if b[i] > 0:
                bb = list(b)
                bb[i] -= 1
                k = (tuple(bb), f)
                c[k] = c.get(k, 0.0) + v * b[i]
        return SA(self.nb, self.D, c)

    def substitute(self, bsubs, fsubs, nb_new):
        """bsubs[i]: even SA replacing bosonic var i; fsubs: dict gen -> odd SA replacing generator."""
        D = self.D
        res = SA(nb_new, D)
        bp = [[SA.const(nb_new, D, 1)] for _ in range(self.nb)]
        for (b, f), v in self.c.items():
            term = SA.const(nb_new, D, v)
            for i, n in enumerate(b):
                while len(bp[i]) <= n:
                    bp[i].append(bp[i][-1] * bsubs[i])
                if n:
                    term = term * bp[i][n]
            for g in f:
                term = term * fsubs[g]
            res = res + term
        return res

    def drop_gens(self, gens):
        """Keep only terms without the given generators (set them to zero)."""
        return SA(self.nb, self.D, {k: v for k, v in self.c.items() if not set(k[1]) & set(gens)})


def sa_exp(E, nmax):
    res = SA.const(E.nb, E.D, 1)
    p = SA.const(E.nb, E.D, 1)
    for n in range(1, nmax + 1):
        p = p * E * (1.0 / n)
        res = res + p
    return res


def sa_log1p(u, nmax):
    res = SA(u.nb, u.D)
    p = SA.const(u.nb, u.D, 1)
    for n in range(1, nmax + 1):
        p = p * u
        res = res + p * ((-1) ** (n + 1) / n)
    return res
