#!/bin/bash
# ncu_profile_expm.sh — NSight Compute profiling for expm_t18_triton (fwd + bwd)
#
# Output: ncu_reports/{triton,eager,compiled}_expm_{fwd,bwd}_B*_N*_bfloat16.ncu-rep
#
# Three implementations are profiled, mirroring expm_bench.py:
#   1. Triton    — hyperconnections.ops.expm_t18_triton   (fused T18 kernel + Fréchet bwd)
#   2. Eager     — torch.linalg.matrix_exp(A)             (Padé scaling-and-squaring, fp32)
#   3. Compiled  — torch.compile(expm_t18, "max-autotune") (pure-torch T18 + autograd)
#
# Defaults to N=4, B=1024 — the case where the new Fréchet backward currently
# regresses vs the old augmented-matrix backward.  Edit B/N below to profile
# other shapes.
#
# Download and view:
#   scp <user>@<host>:<projdir>/ncu_reports/*.ncu-rep .
#   Open in NSight Compute desktop app: https://developer.nvidia.com/nsight-compute
#
# NOTE: ncu requires hardware PMU access.  On this cluster try:
#   ncu --query-metrics 2>&1 | head  — if it errors, prepend "sudo" to the ncu calls below.

set -e

PROJDIR="$(cd "$(dirname "$0")" && pwd)"
REPORTS="${PROJDIR}/ncu_reports"
mkdir -p "${REPORTS}"
cd "${PROJDIR}"

### Python interpreter — point this at the venv you want ncu to use
NCU_PYTHON_VENV="/scratch/gilbreth/neliopou/venvs/hypercons/bin/python"

### Workload dimensions — edit here to change the profiled shapes
### N=4 is the regression case; N=8/16 also relevant for the Fréchet kernel
### bucket comparison.  No D axis: expm operates on [B, N, N] only.
B=1024
N=4
export B N

### Sections to collect (faster than --set full; cover the cache + roofline hypothesis)
NCU_COMMON=(
    ncu
    --nvtx
    --nvtx-include "profile_region/"
    --replay-mode application
    --launch-count 4
    --call-stack-type python
    --import-source on
    --set basic
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum,l1tex__t_bytes.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_op_shared_pred_on.sum
    --force-overwrite
)

### ── 1. Triton expm forward ─────────────────────────────────────────────────
cat > /tmp/_profile_triton_expm.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from hyperconnections.ops import expm_t18_triton

B, N = int(os.environ['B']), int(os.environ['N'])
A = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16) * 0.3

torch.cuda.synchronize()

### profiled region
torch.cuda.nvtx.range_push('profile_region')
expm_t18_triton(A)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Pre-warming Triton expm forward autotune cache ..."
TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_AUTOTUNING=1 "${NCU_PYTHON_VENV}" /tmp/_profile_triton_expm.py

echo "==> Profiling: Triton expm forward ..."
TRITON_SKIP_AUTOTUNING=1 "${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/triton_expm_fwd_B${B}_N${N}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_triton_expm.py
echo " "

### ── 2. PyTorch eager (torch.linalg.matrix_exp) forward ─────────────────────
cat > /tmp/_profile_eager_expm.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch

B, N = int(os.environ['B']), int(os.environ['N'])
A = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16) * 0.3

def f(A):
    ### matches expm_bench.py:ref_torch_matrix_exp — fp32 internally, cast back
    return torch.linalg.matrix_exp(A.float()).to(A.dtype)

### profiled region
torch.cuda.nvtx.range_push('profile_region')
f(A)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: PyTorch eager (matrix_exp) forward ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/eager_expm_fwd_B${B}_N${N}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_eager_expm.py
echo " "

### ── 3. torch.compile(expm_t18) forward ─────────────────────────────────────
cat > /tmp/_profile_compiled_expm.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from hyperconnections.ops import expm_t18 as _expm_t18

### Same compile config as expm_bench.py:148
expm_t18_compiled = torch.compile(_expm_t18, mode="max-autotune", fullgraph=False)

B, N = int(os.environ['B']), int(os.environ['N'])
A = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16) * 0.3

### warmup: trigger inductor compilation outside the profiled region
expm_t18_compiled(A)
torch.cuda.synchronize()

### profiled region
torch.cuda.nvtx.range_push('profile_region')
expm_t18_compiled(A)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: torch.compile(expm_t18) forward ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/compiled_expm_fwd_B${B}_N${N}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_compiled_expm.py
echo " "

### ── 4. Triton expm backward (Fréchet kernel) ───────────────────────────────
cat > /tmp/_profile_triton_expm_bwd.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from hyperconnections.ops import expm_t18_triton

B, N = int(os.environ['B']), int(os.environ['N'])
A = (torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16) * 0.3).requires_grad_(True)

### warmup: prime autotune caches for both fwd and Fréchet bwd kernels
out = expm_t18_triton(A)
out.sum().backward()
A.grad = None
torch.cuda.synchronize()

### run forward outside profiled region — saves A for the backward pass
out = expm_t18_triton(A)
torch.cuda.synchronize()

### profiled region: backward only — exercises _expm_t18_frechet_fwd
torch.cuda.nvtx.range_push('profile_region')
out.sum().backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Pre-warming Triton expm backward autotune cache ..."
TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_AUTOTUNING=1 "${NCU_PYTHON_VENV}" /tmp/_profile_triton_expm_bwd.py

echo "==> Profiling: Triton expm backward ..."
TRITON_SKIP_AUTOTUNING=1 "${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/triton_expm_bwd_B${B}_N${N}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_triton_expm_bwd.py
echo " "

### ── 5. PyTorch eager (torch.linalg.matrix_exp) backward ────────────────────
cat > /tmp/_profile_eager_expm_bwd.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch

B, N = int(os.environ['B']), int(os.environ['N'])
A = (torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16) * 0.3).requires_grad_(True)

def f(A):
    return torch.linalg.matrix_exp(A.float()).to(A.dtype)

### run forward outside profiled region
out = f(A)
torch.cuda.synchronize()

### profiled region: backward only
torch.cuda.nvtx.range_push('profile_region')
out.sum().backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: PyTorch eager (matrix_exp) backward ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/eager_expm_bwd_B${B}_N${N}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_eager_expm_bwd.py
echo " "

### ── 6. torch.compile(expm_t18) backward ────────────────────────────────────
cat > /tmp/_profile_compiled_expm_bwd.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from hyperconnections.ops import expm_t18 as _expm_t18

expm_t18_compiled = torch.compile(_expm_t18, mode="max-autotune", fullgraph=False)

B, N = int(os.environ['B']), int(os.environ['N'])
A = (torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16) * 0.3).requires_grad_(True)

### warmup: trigger compile and prime the backward graph
out = expm_t18_compiled(A)
out.sum().backward()
A.grad = None
torch.cuda.synchronize()

### run forward outside profiled region
out = expm_t18_compiled(A)
torch.cuda.synchronize()

### profiled region: backward only
torch.cuda.nvtx.range_push('profile_region')
out.sum().backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: torch.compile(expm_t18) backward ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/compiled_expm_bwd_B${B}_N${N}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_compiled_expm_bwd.py
echo " "

### ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Done. Reports written to: ${REPORTS}/"
# ls -lh "${REPORTS}"/*.ncu-rep 2>/dev/null || true
