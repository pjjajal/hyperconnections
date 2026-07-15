"""
Kernel-research framework: a registry of optimization *targets* plus a generic
forward/backward evaluation engine.

A **target** is one kernel under optimization (e.g. the expm_forced backward, the
stream_mix big-NB backward).  To add a target you subclass `KernelTarget`, fill in
a few declarative attributes, and implement a small set of hooks:

    make_inputs / call_args / diff_inputs / loss / check_forward / check_backward

The engine (`KernelTarget.evaluate`) then handles, identically for every target:
  - importing the sandbox candidate (reloaded each call) and the canonical baseline,
  - timing forward or backward (backward-only, via retain_graph),
  - computing speedup = baseline_median / candidate_median,
  - correctness vs the target's trusted reference.

Nothing in the engine or the CLI is target-specific, so a new kernel never touches
them — register it in `TARGETS` and write its hooks.

Reseeding the sandbox from canonical is also data-driven here (`reseed`), using each
target's `baseline_file` / `candidate_file` / `import_subs`.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch

### ── path setup so benchmarks/ helpers import ─────────────────────────────────
_THIS = os.path.abspath(__file__)
_KR   = os.path.dirname(_THIS)
_PKG  = os.path.dirname(_KR)
_ROOT = os.path.dirname(_PKG)
_OPS_DIR     = os.path.join(_PKG, "ops")
_SANDBOX_DIR = os.path.join(_KR, "sandbox")
for _p in (_ROOT, os.path.join(_ROOT, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_utils import bench_stats, DEVICE          # noqa: E402
from expm_common import _check as _mag_check          # noqa: E402  (magnitude-aware)

### ── bench knobs (mutable; set_bench overrides from the CLI) ──────────────────
_CFG = {"warmup": 16, "rep": 128}


def set_bench(warmup: int, rep: int) -> None:
    _CFG["warmup"], _CFG["rep"] = warmup, rep


def _median(fn) -> float:
    return bench_stats(fn, _CFG["warmup"], _CFG["rep"]).median


###
### Base class — the generic evaluation engine
###
class KernelTarget:
    ### declarative attributes (subclass sets these)
    name: str                 # registry key AND forum name
    candidate_module: str     # importable path to the sandbox candidate
    baseline_module: str      # importable path to the canonical kernel
    public_fn: str            # attribute fetched from each module
    atol_fwd: float
    atol_bwd: float
    default_n: list[int]
    default_b: int
    ### reseed metadata
    baseline_file: str        # basename in ops/
    candidate_file: str       # basename in sandbox/
    import_subs: list[tuple[str, str]] = []   # (old, new) rewrites for the sandbox copy

    ### ── hooks (subclass implements) ──
    def make_inputs(self, size: dict, requires_grad: bool) -> dict:
        """Build bf16 inputs for `size`; set requires_grad on differentiable ones.
        Must be deterministic (fixed seed) so candidate and baseline see identical data."""
        raise NotImplementedError

    def call_args(self, inputs: dict) -> tuple:
        """Positional args passed to the kernel's public fn."""
        raise NotImplementedError

    def diff_inputs(self, inputs: dict) -> list[torch.Tensor]:
        """The grad-bearing inputs, in a stable order (matches check_backward)."""
        raise NotImplementedError

    def loss(self, out) -> torch.Tensor:
        """Scalar to backprop for backward timing."""
        raise NotImplementedError

    def check_forward(self, inputs: dict, out) -> tuple[bool, float]:
        raise NotImplementedError

    def check_backward(self, inputs: dict, grads: list[torch.Tensor]) -> tuple[bool, float]:
        raise NotImplementedError

    def sizes_from_args(self, args) -> list[dict]:
        """Build the list of size-dicts to evaluate from parsed CLI args."""
        raise NotImplementedError

    def size_label(self, size: dict, direction: str) -> str:
        raise NotImplementedError

    ### ── generic engine (target-agnostic) ──
    def _load(self, module_path: str, reload: bool = False):
        m = importlib.import_module(module_path)
        if reload:
            m = importlib.reload(m)        # pick up the agent's latest sandbox edit
        return getattr(m, self.public_fn)

    def _forward(self, fn, inputs):
        return fn(*self.call_args(inputs))

    def _time_backward(self, fn, size):
        inputs = self.make_inputs(size, requires_grad=True)
        out = self._forward(fn, inputs)
        loss = self.loss(out)
        diff = self.diff_inputs(inputs)

        def _bwd():
            for t in diff:
                t.grad = None
            loss.backward(retain_graph=True)

        med = _median(_bwd)
        return med, [t.grad for t in diff], inputs

    def evaluate(self, direction: str, size: dict) -> dict:
        base_fn = self._load(self.baseline_module)
        cand_fn = self._load(self.candidate_module, reload=True)

        if direction == "fwd":
            in_c = self.make_inputs(size, requires_grad=False)
            cand_med = _median(lambda: self._forward(cand_fn, in_c))
            in_b = self.make_inputs(size, requires_grad=False)   # same seed → identical
            base_med = _median(lambda: self._forward(base_fn, in_b))
            out = self._forward(cand_fn, in_c)
            passed, err = self.check_forward(in_c, out)
        else:
            cand_med, grads, in_c = self._time_backward(cand_fn, size)
            base_med, _, _        = self._time_backward(base_fn, size)
            passed, err = self.check_backward(in_c, grads)

        return dict(size=self.size_label(size, direction), cand=cand_med, base=base_med,
                    speedup=base_med / cand_med, err=err, passed=passed)


###
### Trusted references (copied, not imported, to stay free of the benches'
### top-level torch.compile setup)
###
def _build_aug_2N(A):
    B_, N_, _ = A.shape
    eye  = torch.eye(N_, dtype=A.dtype, device=A.device).expand(B_, N_, N_).contiguous()
    zero = torch.zeros_like(eye)
    return torch.cat([torch.cat([A, eye], dim=-1), torch.cat([zero, zero], dim=-1)], dim=-2)


def _ref_grad_matrix_exp_aug(A):
    A_r = A.detach().float().requires_grad_(True)
    expM = torch.linalg.matrix_exp(_build_aug_2N(A_r))
    expM[:, :A_r.shape[-1], :].sum().backward()
    return A_r.grad


def _ref_phi1_quadrature(A, n_nodes=64):
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    A64 = A.double()
    psi = torch.zeros_like(A64)
    for xq, w in zip(nodes, weights):
        theta = 0.5 * (xq + 1.0)
        psi += (0.5 * w) * torch.linalg.matrix_exp(theta * A64)
    return psi.to(A.dtype)


def _ref_no_proj(Phi, x, Y):
    return torch.einsum("bij,bjd->bid", Phi, x) + Y


def _ref_proj(Phi, x, Y, v):
    n = x.shape[1]
    proj = torch.einsum("bi,bj->bij", v, v)
    orth = torch.eye(n, device=x.device, dtype=x.dtype) - proj
    x_proj = torch.einsum("bij,bjd->bid", proj, x)
    x_orth = torch.einsum("bij,bjd->bid", orth, x)
    return x_proj + torch.einsum("bij,bjd->bid", Phi, x_orth) + Y


def _ref_smix_backward(Phi, x, Y, v):
    Phi_r = Phi.detach().float().requires_grad_(True)
    x_r   = x.detach().float().requires_grad_(True)
    Y_r   = Y.detach().float().requires_grad_(True)
    if v is None:
        _ref_no_proj(Phi_r, x_r, Y_r).sum().backward()
    else:
        _ref_proj(Phi_r, x_r, Y_r, v.detach().float()).sum().backward()
    return Phi_r.grad, x_r.grad, Y_r.grad


###
### Concrete target: expm_forced (expm_t18_block_triton)
###
class ExpmForceTarget(KernelTarget):
    name             = "expm_force"
    candidate_module = "hyperconnections.kernel_research.sandbox.expm_triton_candidate"
    baseline_module  = "hyperconnections.ops.expm_triton"
    public_fn        = "expm_t18_block_triton"
    atol_fwd, atol_bwd = 5e-2, 1e-1
    default_n, default_b = [16], 4096
    baseline_file  = "expm_triton.py"
    candidate_file = "expm_triton_candidate.py"
    import_subs    = [("from .numbers import", "from hyperconnections.ops.numbers import")]

    def make_inputs(self, size, requires_grad):
        torch.manual_seed(0)
        A = torch.randn(size["B"], size["N"], size["N"], device=DEVICE, dtype=torch.bfloat16) * 0.3
        if requires_grad:
            A = A.detach().clone().requires_grad_(True)
        return {"A": A}

    def call_args(self, inputs):
        return (inputs["A"],)

    def diff_inputs(self, inputs):
        return [inputs["A"]]

    def loss(self, out):
        E, psi = out
        return E.sum() + psi.sum()

    def check_forward(self, inputs, out):
        A = inputs["A"]
        E, psi = out
        p1, e1 = _mag_check(E,   torch.linalg.matrix_exp(A.float()).to(torch.bfloat16), self.atol_fwd)
        p2, e2 = _mag_check(psi, _ref_phi1_quadrature(A),                                self.atol_fwd)
        return (p1 and p2), max(e1, e2)

    def check_backward(self, inputs, grads):
        (gA,) = grads
        return _mag_check(gA, _ref_grad_matrix_exp_aug(inputs["A"]).to(torch.bfloat16), self.atol_bwd)

    def sizes_from_args(self, args):
        ns = args.n if args.n is not None else self.default_n
        B  = args.b if args.b is not None else self.default_b
        return [{"B": B, "N": N} for N in ns]

    def size_label(self, size, direction):
        return f"B={size['B']} N={size['N']} bf16 {direction}"


###
### Concrete target: stream_mix big-NB (stream_mix_add_big_nb)
###
class StreamMixTarget(KernelTarget):
    name             = "stream_mix"
    candidate_module = "hyperconnections.kernel_research.sandbox.stream_mix_big_nb_candidate"
    baseline_module  = "hyperconnections.ops.stream_mix_big_nb"
    public_fn        = "stream_mix_add_big_nb"
    atol_fwd, atol_bwd = 2e-2, 4e-2
    default_n, default_b = [32], 16384
    baseline_file  = "stream_mix_big_nb.py"
    candidate_file = "stream_mix_big_nb_candidate.py"
    import_subs    = []   # no relative imports

    def make_inputs(self, size, requires_grad):
        B, N, D = size["B"], size["N"], size["D"]
        torch.manual_seed(0)
        Phi = torch.randn(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        x   = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
        Y   = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
        v = None
        if size.get("proj"):
            torch.manual_seed(7)
            v = torch.nn.functional.normalize(
                torch.randn(B, N, device=DEVICE, dtype=torch.bfloat16), dim=-1)
        if requires_grad:
            Phi = Phi.detach().clone().requires_grad_(True)
            x   = x.detach().clone().requires_grad_(True)
            Y   = Y.detach().clone().requires_grad_(True)
        return {"Phi": Phi, "x": x, "Y": Y, "v": v}

    def call_args(self, inputs):
        return (inputs["Phi"], inputs["x"], inputs["Y"], inputs["v"])

    def diff_inputs(self, inputs):
        return [inputs["Phi"], inputs["x"], inputs["Y"]]

    def loss(self, out):
        return out.sum()

    def check_forward(self, inputs, out):
        Phi, x, Y, v = inputs["Phi"], inputs["x"], inputs["Y"], inputs["v"]
        ref = (_ref_proj(Phi.float(), x.float(), Y.float(), v.float()) if v is not None
               else _ref_no_proj(Phi.float(), x.float(), Y.float())).to(torch.bfloat16)
        return _mag_check(out, ref, self.atol_fwd)

    def check_backward(self, inputs, grads):
        ### Magnitude-aware per grad (like the expm target): grad_Phi sums over D, so
        ### its magnitude scales with sqrt(D) and a flat absolute atol is meaningless —
        ### the bf16 cast of grad_Phi alone gives err ~0.5 at D=1024 for BOTH the
        ### canonical and any candidate.  _mag_check uses atol*(1 + ||ref||_inf).
        Phi, x, Y, v = inputs["Phi"], inputs["x"], inputs["Y"], inputs["v"]
        refs = _ref_smix_backward(Phi, x, Y, v)
        passed, err = True, 0.0
        for g, r in zip(grads, refs):
            p, e = _mag_check(g, r, self.atol_bwd)
            passed &= bool(p)
            err = max(err, e)
        return passed, err

    def sizes_from_args(self, args):
        ns = args.n if args.n is not None else self.default_n
        B  = args.b if args.b is not None else self.default_b
        D  = args.d if args.d is not None else 1024
        return [{"B": B, "N": N, "D": D, "proj": args.proj} for N in ns]

    def size_label(self, size, direction):
        pp = "proj" if size.get("proj") else "noproj"
        return f"N={size['N']} B={size['B']} D={size['D']} bf16 {pp} {direction}"


###
### Registry — add a target here and it is immediately CLI- and reseed-visible.
###
TARGETS: dict[str, KernelTarget] = {t.name: t for t in (ExpmForceTarget(), StreamMixTarget())}


_RESEED_HEADER = (
    "### SANDBOX CANDIDATE — reseeded from canonical by auto_research.py --reseed.\n"
    "### Edit freely; the canonical module is the baseline-to-beat.\n"
)


def reseed(name: str | None = None) -> None:
    """Copy canonical kernel(s) into the sandbox, applying each target's import_subs.
    `name=None` reseeds every registered target."""
    items = TARGETS.values() if name is None else [TARGETS[name]]
    for t in items:
        with open(os.path.join(_OPS_DIR, t.baseline_file)) as f:
            text = f.read()
        for old, new in t.import_subs:
            if old not in text:
                raise RuntimeError(f"reseed {t.name}: import {old!r} not found in {t.baseline_file}")
            text = text.replace(old, new)
        with open(os.path.join(_SANDBOX_DIR, t.candidate_file), "w") as f:
            f.write(_RESEED_HEADER + text)
        print(f"[reseed] {t.baseline_file} -> sandbox/{t.candidate_file}")
