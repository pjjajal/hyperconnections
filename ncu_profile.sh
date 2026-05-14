#!/bin/bash
# ncu_profile.sh — NSight Compute profiling for stream_mix_add (fwd + bwd)
#
# Output: ncu_reports/{triton,eager,compiled}_{fwd,bwd}_B*_N*_D*_bfloat16.ncu-rep
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
### Dispatch note: big_nb path activates when N>=16, N%16==0, and B*N*D*2 > 30 MB
###   e.g. B=512, N=32, D=1024 → 33 MB → big_nb; B=64, N=4, D=1024 → 0.5 MB → small_nb
B=1024
N=4
D=1536
export B N D

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

### ── 1. Triton ──────────────────────────────────────────────────────
cat > /tmp/_profile_triton.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from hyperconnections.ops import stream_mix_add

B, N, D = int(os.environ['B']), int(os.environ['N']), int(os.environ['D'])
Phi = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16)
x   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)
Y   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)

torch.cuda.synchronize()

### profiled region
torch.cuda.nvtx.range_push('profile_region')
stream_mix_add(Phi, x, Y)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Pre-warming Triton autotune cache ..."
TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_AUTOTUNING=1 "${NCU_PYTHON_VENV}" /tmp/_profile_triton.py

echo "==> Profiling: Triton small_nb ..."
TRITON_SKIP_AUTOTUNING=1 "${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/triton_fwd_B${B}_N${N}_D${D}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_triton.py
### ── 2. PyTorch eager ────────────────────────────────────────────────────────
cat > /tmp/_profile_eager.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from einops import einsum

B, N, D = int(os.environ['B']), int(os.environ['N']), int(os.environ['D'])
Phi = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16)
x   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)
Y   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)

def f():
    return einsum(Phi, x, 'b n1 n2, b n2 d -> b n1 d') + Y

### profiled region
torch.cuda.nvtx.range_push('profile_region')
f()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: PyTorch eager ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/eager_fwd_B${B}_N${N}_D${D}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_eager.py
echo " "

### ── 3. torch.compile ────────────────────────────────────────────────────────
cat > /tmp/_profile_compiled.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from einops import einsum

B, N, D = int(os.environ['B']), int(os.environ['N']), int(os.environ['D'])
Phi = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16)
x   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)
Y   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)

@torch.compile
def f(Phi, x, Y):
    return einsum(Phi, x, 'b n1 n2, b n2 d -> b n1 d') + Y

### profiled region
torch.cuda.nvtx.range_push('profile_region')
f(Phi, x, Y)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: torch.compile ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/compiled_fwd_B${B}_N${N}_D${D}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_compiled.py
echo " "

### ── 4. Triton backward ─────────────────────────────────────────────────────
cat > /tmp/_profile_triton_bwd.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from hyperconnections.ops import stream_mix_add

B, N, D = int(os.environ['B']), int(os.environ['N']), int(os.environ['D'])
Phi = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
x   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
Y   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)

### warmup: prime autotune caches for both fwd and bwd kernels
out = stream_mix_add(Phi, x, Y)
out.sum().backward()
Phi.grad = x.grad = Y.grad = None
torch.cuda.synchronize()

### run forward outside profiled region — saves tensors needed by backward
out = stream_mix_add(Phi, x, Y)
torch.cuda.synchronize()

### profiled region: backward only
torch.cuda.nvtx.range_push('profile_region')
out.sum().backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Pre-warming Triton backward autotune cache ..."
TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_AUTOTUNING=1 "${NCU_PYTHON_VENV}" /tmp/_profile_triton_bwd.py

echo "==> Profiling: Triton backward ..."
TRITON_SKIP_AUTOTUNING=1 "${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/triton_bwd_B${B}_N${N}_D${D}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_triton_bwd.py
echo " "

### ── 5. PyTorch eager backward ──────────────────────────────────────────────
cat > /tmp/_profile_eager_bwd.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from einops import einsum

B, N, D = int(os.environ['B']), int(os.environ['N']), int(os.environ['D'])
Phi = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
x   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
Y   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)

def f(Phi, x, Y):
    return einsum(Phi, x, 'b n1 n2, b n2 d -> b n1 d') + Y

### run forward outside profiled region
out = f(Phi, x, Y)
torch.cuda.synchronize()

### profiled region: backward only
torch.cuda.nvtx.range_push('profile_region')
out.sum().backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: PyTorch eager backward ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/eager_bwd_B${B}_N${N}_D${D}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_eager_bwd.py
echo " "

### ── 6. torch.compile backward ──────────────────────────────────────────────
cat > /tmp/_profile_compiled_bwd.py << 'PYEOF'
import sys, os; sys.path.insert(0, '.')
import torch
from einops import einsum

B, N, D = int(os.environ['B']), int(os.environ['N']), int(os.environ['D'])
Phi = torch.randn(B, N, N, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
x   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
Y   = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16).requires_grad_(True)

@torch.compile
def f(Phi, x, Y):
    return einsum(Phi, x, 'b n1 n2, b n2 d -> b n1 d') + Y

### warmup: trigger compile and prime backward graph
out = f(Phi, x, Y)
out.sum().backward()
Phi.grad = x.grad = Y.grad = None
torch.cuda.synchronize()

### run forward outside profiled region
out = f(Phi, x, Y)
torch.cuda.synchronize()

### profiled region: backward only
torch.cuda.nvtx.range_push('profile_region')
out.sum().backward()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
PYEOF

echo "==> Profiling: torch.compile backward ..."
"${NCU_COMMON[@]}" \
    --kernel-name-base function \
    --export "${REPORTS}/compiled_bwd_B${B}_N${N}_D${D}_bfloat16" \
    "${NCU_PYTHON_VENV}" /tmp/_profile_compiled_bwd.py
echo " "

### ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Done. Reports written to: ${REPORTS}/"
# ls -lh "${REPORTS}"/*.ncu-rep 2>/dev/null || true
