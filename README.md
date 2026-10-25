# Hyperconnections

A PyTorch implementations of **Hyperconnections**.

CGHC accepts `projection_variant="one_sided"` (default) or
`projection_variant="two_sided"`, alongside `projection="mean"` or `projection="v"`.
With projector `P` and transition matrix `T = exp(generator)`, these apply
`Px + T(I - P)x` and `Px + (I - P)T(I - P)x`, respectively. The module write-back
is added afterward. With `projection="none"`, both variants apply `Tx`.

Use `projection_variant="generator"` to project the generator before taking
the exponential: `T = exp((I - P) generator (I - P))`. This works with every
`generator_type` and applies `Tx` directly, preserving the projected component
before module write-back. With `projection="none"`, it uses the original generator.

CGHC and forced CGHC also accept:

- `mixer_chunks=C` (default `1`): split each stream's `D = embed_dim // m` features into `C`
  contiguous chunks, each mixed by its own `[n, n]` transition. The generator, timestep,
  projection direction and inflow are per chunk; read/write routing is shared. `D` must be
  divisible by `C`; `C = 1` is the unchunked model. Not supported by forced CGHC.
- `expm_backend` (`triton` | `flash` | `eager`; default: `flash` when `use_triton` and
  flash-hypercons is installed, else `triton` when `use_triton`, else eager): the matrix
  exponential; `flash` uses flash-hypercons' CUDA kernel.
- `num_readouts=K` (default `None`): use read weights of shape `[K, n]` (plus batch/token
  axes for dynamic weights), passing `[..., K, D]` directly to the module.
  The module selects its own branches and must return `[..., embed_dim]`.
  For `m=1`, each readout has the full embedding width. `num_readouts=None` preserves
  the existing flattened `[..., embed_dim]` module input; explicit `num_readouts=1`
  retains the singleton readout axis. Writeback is unchanged.

For example, a SwiGLU module receiving `num_readouts=2` can compute
`down(up(x[..., 0, :]) * silu(gate(x[..., 1, :])))`. An attention module
receiving `num_readouts=3` can select its Q, K, and V inputs the same way. These modules
own their projections and normalization; CGHC only supplies the readouts.
