### Hardware and Kernel Constants
### A100 L2/L1 capacity in bytes
_SM80_A100_NUM_SM = 108
_SM80_A100_L2_BYTES = 40 * 1024 * 1024
_SM80_A100_L1_PER_SM_BYTES = 192 * 1024

### T18 matrix-exponential polynomial coefficients
### Bader, Blanes & Casas, "Computing the Matrix Exponential with an Optimized
### Taylor Polynomial Approximation", Mathematics 2019, 7, 1174.
### Equation 15 (k=5 product decomposition).  Source of truth for all
### implementations (expm.py, expm_block.py, expm_triton.py).
###
### Polynomial structure:
###   B1 = a01·I + a11·A + a21·A² + a31·A³
###   B2 = b01·I + b11·A + b21·A² + b31·A³ + b61·A⁶
###   B3 = b02·I + b12·A + b22·A² + b32·A³ + b62·A⁶
###   B4 = b03·I + b13·A + b23·A² + b33·A³ + b63·A⁶
###   B5 = b04·I + b14·A + b24·A² + b34·A³ + b64·A⁶
###   A9  = B1·B5 + B4
###   T18 = B2 + (B3 + A9)·A9
###
### Zero coefficients (a01, b01, b04, b14) are kept for documentation;
### implementations may drop them for efficiency.

# B1
_a01 =  0.0
_a11 = -0.10036558103014462
_a21 = -0.008029246482411570
_a31 = -0.000892138498045730

# B2
_b01 =  0.0
_b11 =  0.39784974949645076
_b21 =  1.36783778460411719
_b31 =  0.49828962252538268
_b61 = -0.00063789819459247233

# B3
_b02 = -10.96763960529620626
_b12 =  1.68015813878906197
_b22 =  0.05717798464788655
_b32 = -0.00698210122488052
_b62 =  3.349750170860705e-05

# B4
_b03 = -0.09043168323908106
_b13 = -0.06764045190713819
_b23 =  0.06759613017740597
_b33 =  0.02955525704293155
_b63 = -1.391802575160607e-05

# B5
_b04 =  0.0
_b14 =  0.0
_b24 = -0.09233646193671186
_b34 = -0.01693649390020817
_b64 = -1.400867981820361e-05

### Scaling threshold for float32 (Table 2, u ≤ 2⁻²⁴, m=18).
### ‖A‖₁ > _THETA_18_F32 triggers scaling-and-squaring.
_THETA_18_F32 = 3.01
