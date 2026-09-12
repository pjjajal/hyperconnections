# Hyperconnections

A PyTorch implementations of **Hyperconnections**.

CGHC accepts `projection_variant="one_sided"` (default) or
`projection_variant="two_sided"`, alongside `projection="mean"` or `projection="v"`.
With projector `P` and transition matrix `T = exp(generator)`, these apply
`Px + T(I - P)x` and `Px + (I - P)T(I - P)x`, respectively. The module write-back
is added afterward. With `projection="none"`, both variants apply `Tx`.
