"""Sec. III A with three Grassmann flavours and a generic antisymmetrised interaction:
fermionic relation (11), SBE decomposition (63), its zeta-signed general form (77), SD equations (67), (68).
A bosonic truncation D = 2 suffices for these quantities; this is verified first on the two-flavour model."""
import fermions as fm
import numpy as np, time, io, contextlib

fm.res.clear()
m2 = fm.Model(h=[[0.3, 0.2], [0.2, -0.5]], u=fm.antisym_u(2, {(0, 1, 0, 1): 1.1}), Uph=[[0.7, 0.3], [0.3, -0.4]],
              Upp=[[0, 0.9], [-0.9, 0]], Uppb=[[0, 0.6], [-0.6, 0]], chi0ph=0.5, chi0pp=0.8)
with contextlib.redirect_stdout(io.StringIO()):
    fm.run(m2, D=2)
print("two flavours with D = 2 (consistency of the truncation):", len(fm.res), "checks,",
      sum(1 for _, ok in fm.res if not ok), "failed\n")
fm.res.clear()

fm.NF = 3
fm.F = lambda i: i
fm.FD = lambda i: 3 + i
fm.SRC = lambda a: 6 + a
rng = np.random.default_rng(1)
h = np.diag([1.0, -0.8, 1.3]) + 0.25 * np.array([[0, 1, -0.6], [1, 0, 0.8], [-0.6, 0.8, 0]])
pairs = [(0, 1), (0, 2), (1, 2)]
A = rng.normal(size=(3, 3)) * 0.6
A = (A + A.T) / 2                      # u_{(ij)(kl)} symmetric under pair exchange (Hermitian interaction)
vals = {(i, j, k, l): A[p, q] for p, (i, j) in enumerate(pairs) for q, (k, l) in enumerate(pairs)}
u = fm.antisym_u(3, vals)
Uph = rng.normal(size=(3, 3)); Uph = (Uph + Uph.T) / 2
Upp = rng.normal(size=(3, 3)); Upp = Upp - Upp.T
Uppb = rng.normal(size=(3, 3)); Uppb = Uppb - Uppb.T
m3 = fm.Model(h=h, u=u, Uph=Uph, Upp=Upp, Uppb=Uppb, chi0ph=0.6, chi0pp=0.9)
t = time.time()
print("three flavours:")
fm.run(m3, D=2)
print(f"\nthree flavours: {len(fm.res)} checks, {sum(1 for _, ok in fm.res if not ok)} failed ({time.time() - t:.0f}s)")
