"""Exact numerics for a 0D (single real variable) field theory
S(phi) = m/2 phi^2 + g/6 phi^3 + lam/24 phi^4,  weight exp(-S - J phi).
Conventions follow the paper: W = -ln Z, G^n = d^n W/dJ^n = (-1)^(n-1) kappa_n,
Gamma = W - J.phibar.
"""
import mpmath as mp
from itertools import product
from math import factorial

mp.mp.dps = 40


# ---------------------------------------------------------------- power series
class Series:
    """Truncated multivariate power series: dict {exponent tuple: mpf}."""

    def __init__(self, nv, D, c=None):
        self.nv, self.D = nv, D
        self.c = {} if c is None else {k: v for k, v in c.items() if sum(k) <= D}

    @staticmethod
    def const(nv, D, a):
        return Series(nv, D, {(0,) * nv: mp.mpf(a)})

    @staticmethod
    def var(nv, D, i):
        e = [0] * nv
        e[i] = 1
        return Series(nv, D, {tuple(e): mp.mpf(1)})

    def _lift(self, o):
        return o if isinstance(o, Series) else Series.const(self.nv, self.D, o)

    def __add__(self, o):
        o = self._lift(o)
        c = dict(self.c)
        for k, v in o.c.items():
            c[k] = c.get(k, 0) + v
        return Series(self.nv, self.D, c)

    __radd__ = __add__

    def __neg__(self):
        return Series(self.nv, self.D, {k: -v for k, v in self.c.items()})

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        if not isinstance(o, Series):
            return Series(self.nv, self.D, {k: v * o for k, v in self.c.items()})
        c = {}
        for k1, v1 in self.c.items():
            d1 = sum(k1)
            for k2, v2 in o.c.items():
                if d1 + sum(k2) > self.D:
                    continue
                k = tuple(a + b for a, b in zip(k1, k2))
                c[k] = c.get(k, 0) + v1 * v2
        return Series(self.nv, self.D, c)

    __rmul__ = __mul__

    def coef(self, e):
        return self.c.get(tuple(e), mp.mpf(0))

    def deriv(self, i):
        c = {}
        for k, v in self.c.items():
            if k[i] > 0:
                kk = list(k)
                kk[i] -= 1
                c[tuple(kk)] = v * k[i]
        return Series(self.nv, self.D, c)

    def d(self, idx):
        """Partial derivative at the expansion point; idx = list of variable indices."""
        e = [0] * self.nv
        for i in idx:
            e[i] += 1
        f = 1
        for n in e:
            f *= factorial(n)
        return self.coef(e) * f

    def compose(self, subs, nv_new):
        """Substitute series subs[i] (in nv_new variables) for variable i."""
        D = self.D
        res = Series(nv_new, D)
        powers = [[Series.const(nv_new, D, 1)] for _ in range(self.nv)]
        for k, v in self.c.items():
            term = Series.const(nv_new, D, v)
            for i, n in enumerate(k):
                while len(powers[i]) <= n:
                    powers[i].append(powers[i][-1] * subs[i])
                if n:
                    term = term * powers[i][n]
            res = res + term
        return res


def series_log1p(u):
    """log(1+u) for u with zero constant term."""
    res = Series(u.nv, u.D)
    p = Series.const(u.nv, u.D, 1)
    for k in range(1, u.D + 1):
        p = p * u
        res = res + p * (mp.mpf((-1) ** (k + 1)) / k)
    return res


# ---------------------------------------------------------------- the theory
class Theory:
    def __init__(self, m, g, lam):
        self.m, self.g, self.lam = mp.mpf(m), mp.mpf(g), mp.mpf(lam)

    def S(self, x):
        return self.m / 2 * x**2 + self.g / 6 * x**3 + self.lam / 24 * x**4

    def dS(self, n, x):
        """n-th derivative of S at x."""
        m, g, l = self.m, self.g, self.lam
        return [self.S(x), m * x + g / 2 * x**2 + l / 6 * x**3,
                m + g * x + l / 2 * x**2, g + l * x, l][n] if n <= 4 else mp.mpf(0)

    def moments(self, J, kmax):
        w = lambda x: mp.exp(-self.S(x) - J * x)
        pts = [-mp.inf, -6, -2, 0, 2, 6, mp.inf]
        Z = mp.quad(w, pts)
        mom = [mp.quad(lambda x, k=k: x**k * w(x), pts) / Z for k in range(kmax + 1)]
        return Z, mom

    def J_for_mean(self, target):
        f = lambda J: self.moments(J, 1)[1][1] - target
        return mp.findroot(f, mp.mpf(0))


# polynomials in phi: list of coefficients [c0, c1, ...]
def pmul(a, b):
    r = [mp.mpf(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r


def pexp(p, mom):
    return sum(c * mom[k] for k, c in enumerate(p))


def joint_cumulants(ops, mom, D):
    """Cumulant generating function of operators ops (polynomials in phi),
    returned as Series K(t) = sum kappa_n t^n / n!  (degree D)."""
    d = len(ops)
    M = Series(d, D)
    powcache = {}

    def oppow(i, n):
        if (i, n) not in powcache:
            powcache[(i, n)] = [mp.mpf(1)] if n == 0 else pmul(oppow(i, n - 1), ops[i])
        return powcache[(i, n)]

    for e in product(range(D + 1), repeat=d):
        if sum(e) > D:
            continue
        p = [mp.mpf(1)]
        f = 1
        for i, n in enumerate(e):
            p = pmul(p, oppow(i, n))
            f *= factorial(n)
        M.c[e] = pexp(p, mom) / f
    u = M - 1
    return series_log1p(u)


def cumulant(K, e):
    f = 1
    for n in e:
        f *= factorial(n)
    return K.coef(e) * f


def W_series(ops, mom, D, hs=None):
    """W(J0 + j) - W(J0) as series in source deviations j coupling to ops.
    hs: optional dict {i: chi0} adding +1/2 chi0 j_i^2 (Hubbard-Stratonovich term)."""
    K = joint_cumulants(ops, mom, D)
    W = Series(K.nv, D, {e: -v * (-1) ** sum(e) for e, v in K.c.items() if sum(e) > 0})
    if hs:
        for i, chi0 in hs.items():
            W = W + Series.var(K.nv, D, i) * Series.var(K.nv, D, i) * (mp.mpf(chi0) / 2)
    return W


def legendre(W, T, J0):
    """Legendre transform of W (series in source deviations) w.r.t. variables T.
    J0: base sources (list, len nv). Returns (Gamma series in (dX_T, params), X0).
    Gamma = W - sum_{i in T} J_i X_i."""
    nv, D = W.nv, W.D
    T = list(T)
    X = [W.deriv(i) for i in T]
    X0 = [x.coef((0,) * nv) for x in X]
    A = mp.matrix(len(T), len(T))
    for a, i in enumerate(T):
        for b, j in enumerate(T):
            e = [0] * nv
            e[j] = 1
            A[a, b] = X[a].coef(e)
    Ainv = A**-1
    # rest B = X - X0 - A j_T
    B = []
    for a in range(len(T)):
        b = X[a] - X0[a]
        for bb, j in enumerate(T):
            b = b - Series.var(nv, D, j) * A[a, bb]
        B.append(b)
    dX = [Series.var(nv, D, i) for i in T]
    j = [sum((dX[b] * Ainv[a, b] for b in range(len(T))), Series(nv, D)) for a in range(len(T))]
    for _ in range(D + 1):
        subs = [Series.var(nv, D, i) for i in range(nv)]
        for a, i in enumerate(T):
            subs[i] = j[a]
        Bj = [b.compose(subs, nv) for b in B]
        j = [sum(((dX[b] - Bj[b]) * Ainv[a, b] for b in range(len(T))), Series(nv, D))
             for a in range(len(T))]
    subs = [Series.var(nv, D, i) for i in range(nv)]
    for a, i in enumerate(T):
        subs[i] = j[a]
    G = W.compose(subs, nv)
    for a, i in enumerate(T):
        G = G - (j[a] + J0[i]) * (dX[a] + X0[a])
    return G, X0


def implicit(F, solve_vars, D=None):
    """Solve F_k(x) = 0 (list of series, value 0 at origin) for x_solve as series in the
    remaining variables. Returns dict {var index: series}."""
    nv = F[0].nv
    D = D or F[0].D
    free = [i for i in range(nv) if i not in solve_vars]
    A = mp.matrix(len(F), len(solve_vars))
    for a, f in enumerate(F):
        for b, i in enumerate(solve_vars):
            e = [0] * nv
            e[i] = 1
            A[a, b] = f.coef(e)
    Ainv = A**-1
    R = []
    for a, f in enumerate(F):
        r = f
        for b, i in enumerate(solve_vars):
            r = r - Series.var(nv, D, i) * A[a, b]
        R.append(r)
    x = {i: Series(nv, D) for i in solve_vars}
    for _ in range(D + 1):
        subs = [Series.var(nv, D, i) for i in range(nv)]
        for i in solve_vars:
            subs[i] = x[i]
        Rx = [r.compose(subs, nv) for r in R]
        x = {i: sum((Rx[b] * (-Ainv[a, b]) for b in range(len(F))), Series(nv, D))
             for a, i in enumerate(solve_vars)}
    return x
