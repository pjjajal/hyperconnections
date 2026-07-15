# Kernel-Code Research — stream_mix big-NB (target: `stream_mix`)

## What we're optimizing
The **source code** of the **big-NB** stream-mixing Triton kernels
(`stream_mix_add_big_nb`). At the representative size (N=32, B=16384, D=1024) the dispatcher
routes to big-NB (`B*N*D = 537M ≥ 40M` L2 elements, N ≥ 16), so this is the variant under
study. Canonical: `hyperconnections/ops/stream_mix_big_nb.py`. Forward = `_stream_mix_fwd_big_nb`
(tl.dot); backward = `_stream_mix_bwd_dx_big_nb` + `_stream_mix_bwd_dPhi_big_nb`, plus
Python-side shared intermediates (alpha, beta). Optimize **forward or backward** — eval handles both.

## Sandbox — edit ONLY this file
`hyperconnections/kernel_research/sandbox/stream_mix_big_nb_candidate.py`
Canonical `ops/stream_mix_big_nb.py` is the baseline-to-beat — do NOT edit it.
Reseed a clean copy: `python -m hyperconnections.kernel_research.auto_research --target stream_mix --reseed`.

## Evaluate (how you measure)
```
python -m hyperconnections.kernel_research.evaluate --target stream_mix --pass {fwd|bwd} --post --note "<idea>"
```
Representative size: bf16, N=32, B=16384, D=1024, no-proj (`--proj` for the v-projected
variant; `--d 1536` etc. to vary D).
- fwd correctness: out vs einsum reference; magnitude-aware atol 2e-2.
- bwd correctness: (grad_Phi, grad_x, grad_Y) vs fp32 autograd through einsum; magnitude-aware
  per grad, atol 4e-2 (i.e. err ≤ atol·(1+‖ref‖∞)). grad_Phi sums over D, so its magnitude
  scales with √D — a flat absolute atol is meaningless there (the bf16 cast of grad_Phi alone
  gives err ≈ 0.5 at D=1024 for the canonical kernel too).
- Speed: candidate median vs canonical median (same process) → `speedup`.

## Rules
- One idea per forum entry. State WHAT you changed and WHY.
- Promotion-eligible only if **PASS** (err ≤ atol) **and** speedup > 1.0x.
- Build on the current best PASS candidate; if a change regresses, say so and reseed/revert.
- Keep **fp32 accumulation** (`allow_tf32=False` on the tl.dot calls) — bf16 grads already
  sit near the atol floor.
- Report, never crash. A broken edit shows up as an `ERR:` row.
- Promotion to `ops/` is a separate, human-gated step (then re-run the full bench).

## Result row (auto-emitted by evaluate.py; agents add the trailing idea)
```
CAND <id> | stream_mix | N=<N> B=<B> D=<D> bf16 <noproj|proj> <fwd|bwd> | cand=<F>ms | base=<F>ms | speedup=<F>x | err=<E> | PASS|FAIL | <idea>
```
