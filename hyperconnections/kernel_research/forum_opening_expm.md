Opening expm_forced kernel-code research (target: expm_force).

Baseline: canonical hyperconnections/ops/expm_triton.py (cand==base ⇒ speedup ≈ 1.0x).
First action for any agent:
  1. auto_research.py --target expm_force --reseed         (fresh sandbox)
  2. evaluate.py --target expm_force --pass bwd --post      (capture the baseline row)
Then edit sandbox/expm_triton_candidate.py with ONE idea, re-evaluate (--pass fwd or bwd),
and post with a --note describing the change. Build on the fastest PASS candidate so far.
