# Kernel-Code Research — expm_forced (target: `expm_force`)

## What we're optimizing
The **source code** of the Triton kernels behind `expm_t18_block_triton` (augmented /
"forced" matrix exponential, returning exp(A) and phi_1(A)). Canonical:
`hyperconnections/ops/expm_triton.py`. Forward uses `_expm_t18_augmented_fwd`; backward
launches `_expm_t18_structured_fwd` (NP=next_pow2(N)) and `_expm_t18_fwd` (NP=next_pow2(3N)).
All fp32-accumulate, bf16 I/O. Optimize **forward or backward** — the eval script handles both.

## Sandbox — edit ONLY this file
`hyperconnections/kernel_research/sandbox/expm_triton_candidate.py`
Canonical `ops/expm_triton.py` is the baseline-to-beat — do NOT edit it.
Reseed a clean copy: `python -m hyperconnections.kernel_research.auto_research --target expm_force --reseed`.

## Evaluate (how you measure)
```
python -m hyperconnections.kernel_research.evaluate --target expm_force --pass {fwd|bwd} --post --note "<idea>"
```
Representative size: bf16, B=4096, N=16 (`--n 8 16 32` for the set).
- fwd correctness: E vs `torch.linalg.matrix_exp`, psi vs Gauss-Legendre quadrature; atol 5e-2.
- bwd correctness: grad vs autograd through `matrix_exp` of the 2N augmented matrix; atol 1e-1.
- Speed: candidate median vs canonical median (same process) → `speedup`.

## Rules
- One idea per forum entry. State WHAT you changed and WHY.
- Promotion-eligible only if **PASS** (err ≤ atol) **and** speedup > 1.0x.
- Build on the current best PASS candidate; if a change regresses, say so and reseed/revert.
- Keep **fp32 accumulation**: the broadcast/`tl.dot` contraction silently lowers to TF32 MMA
  at NP ≥ 16 and loses ~13 mantissa bits (see the `_matmul_nn` note) — that's why the fp32
  paths use scalar-FFMA outer products.
- Report, never crash. A broken edit shows up as an `ERR:` row.
- Promotion to `ops/` is a separate, human-gated step (then re-run the full bench).

## Result row (auto-emitted by evaluate.py; agents add the trailing idea)
```
CAND <id> | expm_force | B=<B> N=<N> bf16 <fwd|bwd> | cand=<F>ms | base=<F>ms | speedup=<F>x | err=<E> | PASS|FAIL | <idea>
```
