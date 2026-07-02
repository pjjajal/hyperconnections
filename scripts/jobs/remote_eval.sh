#!/bin/bash
# Run kernel_research/evaluate.py on the user's interactive A100 allocation, from
# the login node (gilbreth-fe02), over ssh — so the research loop never has to
# spell out the ssh/venv/cd plumbing.
#
# Node discovery: uses $KR_NODE if set, else the node of the user's first RUNNING
# a100-80gb job (squeue).  The ssh session is adopted into that job's cgroup
# (pam_slurm_adopt), so the A100 is visible without CUDA_VISIBLE_DEVICES.
#
# All arguments are forwarded verbatim to evaluate.py, e.g.:
#   scripts/jobs/remote_eval.sh --target expm_force --pass bwd --post --note "tf32 mma"
#
# Report, never abort (no `set -e`) so a correctness FAIL/ERR still returns its row.
set -uo pipefail

PROJ=/scratch/gilbreth/neliopou/pj-hyperconnections
VENV_PY=/scratch/gilbreth/neliopou/venvs/uvhypercon/bin/python

NODE="${KR_NODE:-}"
if [ -z "$NODE" ]; then
    NODE=$(squeue -u "$USER" -h -t RUNNING -p a100-80gb -o '%N' 2>/dev/null | head -1)
fi
if [ -z "$NODE" ]; then
    echo "remote_eval: no RUNNING a100-80gb job found for $USER (salloc one, or set KR_NODE)." >&2
    exit 2
fi

echo "remote_eval: node=$NODE  args=$*" >&2
### Shell-quote each arg so notes with spaces/parens survive the remote shell re-parse.
REMOTE_ARGS=$(printf '%q ' "$@")
ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" \
    "cd $PROJ && TRITON_ALWAYS_COMPILE=0 $VENV_PY -m hyperconnections.kernel_research.evaluate $REMOTE_ARGS"
