"""Residual (interaction-irreducible) vertex of the SBE construction, Sec. III C, in the two-flavour Hubbard model:
 1. the cancellation conditions after Eq. (86) are satisfied by the chosen composites;
 2. leading orders: Gamma_ijkl = -2 u_ijkl + 0 * u^2 + ...  (bare vertex Gamma1PI = +u);
 3. Eq. (87) to all orders:
    Gamma_ijkl = -2 u_ijkl - 1/2 u_{i'j'k'l} d/dfbar+_i d/dfbar+_j d/dfbar_k G^{i'j'k'}_psi ,
    with G^{abc}_psi the connected three-point function of the theory with sources fixed by (phibar, psi)."""
import fermions as fm
import sbe_hubbard as hb
from itertools import product
import numpy as np

F, FD, SRC = hb.F, hb.FD, hb.SRC
D = 2
res = []


def rep(label, lhs, rhs, tol=1e-10):
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
    ok = err < tol
    res.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {label:58s} {lhs:+.12e} {rhs:+.12e} rel.err={err:.1e}")


print("=== 1. Cancellation conditions after Eq. (86) (u = 1) ===")
for name, r in zip(("ph:  U chi0^-1 U = u", "ph: -U chi0^-1 U (crossed) = u", "pp:  Ubar chi0^-1 U = u"),
                   hb.condition_residuals(1.0)):
    ok = r < 1e-12
    res.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {name:58s} max deviation = {r:.1e}")

print("\n=== 2. Leading orders of the residual vertex (component 0101) ===")
rows = []
for eps in [0.002, 0.004, 0.006, 0.008]:
    W, nb, u = hb.generating(eps, D)
    G, _, _ = hb.legendre_with_sources(W, nb, D)
    W1, nb1, _ = fm.Model(h=hb.h, u=u, Uph=np.eye(2), Upp=hb.Upp, Uppb=hb.Uppb, chi0ph=1,
                          chi0pp=1).generating(False, D)
    G1 = fm.legendre_ferm(W1, nb1, D, False)
    idx = [FD(0), FD(1), F(0), F(1)]
    rows.append((eps, fm.dvals(G, idx), fm.dvals(G1, idx)))
A = np.array([[e, e**2, e**3, e**4] for e, _, _ in rows])
cr = np.linalg.solve(A, np.array([r[1] for r in rows]))
c1 = np.linalg.solve(A, np.array([r[2] for r in rows]))
rep("residual vertex, coefficient of u", cr[0], -2.0, tol=1e-6)
ok = abs(cr[1]) < 1e-4
res.append(ok)
print(f"{'OK ' if ok else 'FAIL'} {'residual vertex, coefficient of u^2 (must vanish)':58s} {cr[1]:+.3e}")
rep("full 1PI vertex, coefficient of u", c1[0], 1.0, tol=1e-6)

print("\n=== 3. Eq. (87) to all orders ===")
for eps in (0.01, 0.3, 1.0, 3.0):
    W, nb, u = hb.generating(eps, D)
    G, J, K = hb.legendre_with_sources(W, nb, D)
    fs = {SRC(b): J[b] for b in range(4)}
    for i, j, k, l in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1)]:
        lhs = fm.dvals(G, [FD(i), FD(j), F(k), F(l)])
        corr = 0.0
        for ip, jp, kp in product(range(2), repeat=3):
            if u[ip, jp, kp, l] == 0:
                continue
            W3 = W.dL(SRC(F(kp))).dL(SRC(FD(jp))).dL(SRC(FD(ip)))    # G^{abc}, (a,b,c) = (f+_i', f+_j', f_k')
            corr += u[ip, jp, kp, l] * fm.dvals(W3.substitute(K, fs, nb), [FD(i), FD(j), F(k)])
        rep(f"(87) u = {eps:4.2f}, ijkl = {i}{j}{k}{l}", lhs, -2 * u[i, j, k, l] - corr / 2)
print(f"\n{len(res)} checks, {sum(1 for ok in res if not ok)} failed")
