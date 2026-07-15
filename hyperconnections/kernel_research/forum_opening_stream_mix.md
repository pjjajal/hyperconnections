Opening stream_mix (big-NB) kernel-code research (target: stream_mix).

Baseline: canonical hyperconnections/ops/stream_mix_big_nb.py (cand==base ⇒ speedup ≈ 1.0x).
First action for any agent:
  1. auto_research.py --target stream_mix --reseed          (fresh sandbox)
  2. evaluate.py --target stream_mix --pass bwd --post       (capture the baseline row)
Then edit sandbox/stream_mix_big_nb_candidate.py with ONE idea, re-evaluate (--pass fwd or
bwd), and post with a --note describing the change. Build on the fastest PASS candidate.
