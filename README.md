# Exact zero-dimensional checks of "Composite Fields and Tree Expansions: A Unified Framework for Renormalized Vertex Decompositions"  

O. Sulyma and B. Schneider. Equation numbers refer to the manuscript version [arXiv:2608.06449].

This package verifies the main equations of the manuscript against exact numerics in zero-dimensional field
theories, where every correlation function is a finite-dimensional integral (bosons) or a finite Berezin
integral (fermions). Composite effective actions are constructed independently, by multivariate Legendre
transforms of truncated power series in the sources; no formula of the manuscript enters the reference values.
Every check in the package passes; the outputs are in `logs/`.

## Requirements and usage

Python 3 with `mpmath`, `numpy`. Run `bash run_all.sh` (about 3 minutes in total) or any script individually,
e.g. `python3 single_field.py`. Each script prints one line per check (`OK`/`FAIL`, both sides and the
deviation) and a final summary.

## Models

| Module | Theory | Arithmetic |
|---|---|---|
| `zerod.py` | one real field, S = m/2 φ² + g/6 φ³ + λ/24 φ⁴, with (g ≠ 0) and without cubic term | 40 digits |
| `twod.py` | two real fields with generic tensors S_ab, S_abc, S_abcd | 30 digits |
| `fermions.py`, `superalg.py`, `superalg2.py` | two or three Grassmann flavours, S = f†_i h_ij f_j + ¼ u_ijkl f†_i f†_j f_k f_l, generic h and antisymmetrised u | double |
| `sbe_hubbard.py` | two-flavour Hubbard model with the composite fields of Sec. III A satisfying the cancellation conditions after Eq. (86) (four particle–hole components, one pairing pair) | double |

Conventions follow the manuscript: W = −ln Z, Gⁿ = δⁿW/δJⁿ = (−1)^(n−1)⟨φ…φ⟩_c, Γ = W − J·φ̄, ζ = ±1,
G^m_(2)kl = (−1)^(m+k+l−1)⟨φ^m R^k Q^l⟩_c with R = ½S₄φφ and Q = δS_int/δφ − Σφ. Two-field checks use ordered index
sums and the symmetric derivative δG^ab/δG^cd = ½(δ^a_c δ^b_d + δ^a_d δ^b_c).

## Scripts

| Script | Equations | Checks |
|---|---|---|
| `single_field.py` | (2a), (21); estimators (114), (115), (118), (119) and the rule of Sec. IV B for n = 5 and the full n = 6 tree (theory with cubic term); Ω derivatives, (98) for n = 3–6, classes (104), (106), (109), (112), (113), core class n = 5; composite tree expansions (1), (70), (72), (76), (77), (81), (82) for ψ• = G², ψ• = (G², G³) and two Hubbard–Stratonovich choices; even theory (32), (33), (37), (38), note [76]; Appendix B: (B4)–(B6), (B7) (even theory), (B8) (0D form), I^2PI_2(2) = I^3PI_2(2)\|even, (B9), (B11), (B12), (B13), (28) | 119 |
| `two_fields.py` | tensor forms of (114), (115), (119) with the P₃ and P₆ index sums; (72), (77) for ψ• = G^ab | 33 |
| `two_fields_parquet.py` | (32), (33), parquet (37) with ordered index sums; (B4), (B5); (B9) in tensor form | 75 |
| `two_fields_five_point.py` | tensor form of (81), including the P₁₀ and P₁₅ terms | 3 |
| `fermions.py` | fermionic relation (11) with ζ = −1; SBE decomposition (63) for all index combinations; its ζ-signed general form (77); Schwinger–Dyson equations (67) and (68) | 53 |
| `fermions_three_flavours.py` | the same with three flavours and a generic interaction (36 nonzero vertex components; all channel terms of (63) individually nonzero) | 208 (+53 truncation check) |
| `eq87.py` | cancellation conditions after Eq. (86); leading orders of the residual vertex (−2u, no u² term); Eq. (87) to all orders for u = 0.01 … 3 (G^{i′j′k′}_ψ from the theory with sources fixed by (φ̄, ψ)) | 18 |
| `appendixC.py` | bilinear plus trilinear composites: the cancellation identity, Eq. (1), and the structure (C4)–(C5), i.e. that apart from S₀[φ̄] the composite action depends on φ̄ only through terms linear in ψ•; in particular the residual four-fermion vertex at fixed ψ• vanishes. Run with argument 4 or 5 (total degree of the truncation; 5 includes all vertices with up to five legs) | 9 |

## Remarks specific to zero dimensions

- **(B8):** in 0D all index assignments coincide, so the test fixes the normalisation of the symmetrised tensor
  equation, not its index structure.
- **(B9):** the tensor form (two fields) tests the index structure, in particular the contraction of the second
  index pair and the factor of the identity term.
- **Appendix C, choice of composites:** with two flavours the trilinears built from a complete particle–hole basis are linearly dependent
  in 0D. The construction is therefore realised with the two density bilinears, whose four nonzero trilinears
  n₀f₁, n₁f₀, n₀f†₁, n₁f†₀ are independent. The coupling χ₀⁻¹ = [[0, −u], [−u, 0]] reproduces δS_int/δf† term by
  term, including the vanishing coefficients of the identically zero composites n_i f_i, which is what the
  cancellation condition of Appendix C guarantees in general.
- **Appendix C, (C1):** the substitution is applied to the whole generating functional, including the
  Hubbard–Stratonovich term.
- **Grassmann weight:** the 0D Grassmann weight is not positive; the checks test algebraic identities, which hold
  irrespective of positivity.

## Infrastructure

`zerod.py` and `twod.py` provide moments, joint cumulants, truncated power series and Legendre transforms for the
bosonic theories. `superalg.py` (bosonic-degree truncation) and `superalg2.py` (total-degree truncation) implement
power series in bosonic variables with Grassmann generators; `fermions.py` builds the fermionic generating
functional by exact Berezin integration and performs the super-Legendre transform with the left-derivative
conventions of the manuscript.
