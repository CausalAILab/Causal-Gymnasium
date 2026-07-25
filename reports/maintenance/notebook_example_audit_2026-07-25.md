# Notebook example audit — 2026-07-25

## Scope and method

This audit covers all 16 existing `examples/*.ipynb` files on
`ey/notebook-example-maintenance`, based directly on `origin/main`.

Two independent checks were applied:

1. **Execution:** clear outputs in memory, start a fresh
   `causal-gymnasium` kernel, execute every cell from the top, and inspect
   exceptions, timeouts, swallowed errors, dependencies, assets, paths,
   device assumptions, checkpoints, and visual artifacts.
2. **Output:** compare declared formulas and expected values with the actual
   result. For visual environments, check non-empty rendered output and
   episode progression. For training examples, check the flow without claiming
   full algorithm performance.

The run used headless SDL, the Matplotlib inline backend, and a 180-second
per-cell timeout. Executed copies and raw JSONL results were kept outside the
repository under `/private/tmp/causal_gym_notebook_audit_final/`; the three
final theory/report fixes were rechecked under
`/private/tmp/causal_gym_notebook_audit_theory_fix/`.

## Summary

- Execution: **15 Pass, 1 Blocked, 0 remaining Fail**.
- Output: **13 Pass/reasonable, 1 flow-only, 1 Incorrect, 1 Not checked**.
- Masked Atari is Blocked because no legally supplied Pong ROM is installed.
- MAB executes, but its output does not match the Chapter 7 equations. The
  cause is the public SCM reward mechanism, so it is recorded for a semantic
  branch rather than changed here.

## Per-notebook evidence

| Notebook | Execution evidence | Output evidence | Verdict |
| --- | --- | --- | --- |
| AntMaze | 9 code cells; 8.93 s; checkpoint ZIP generated and loaded | Reset, short SAC/HER flow, prediction, and step succeed; no policy-quality claim | Pass / Flow pass |
| CartPole | 8 cells; PNG file plus inline figure | `see` average reward 22.52 versus `do(X=0)` 9.21; qualitative comparison is consistent | Pass / Pass |
| CartPole visual | 5 cells; 4 inline PNG outputs | State changes over real steps; heavy wind and angle-offset runs terminate after progression | Pass / Reasonable |
| DTR | 6 cells; 1 inline PNG | 1,000 observational and interventional episodes complete; binary distributions are valid and the process advances through two stages | Pass / Qualitative pass |
| FrozenLake | 8 cells; 4 PNG and 1 HTML animation | Forced action advances the environment and produces reward/state output | Pass / Reasonable |
| Highway | 8 cells; 1,000 × 800 GIF generated (about 3 MB) | Non-empty visual rollout and saved animation | Pass / Reasonable |
| Highway single-step | 9 cells; 5 PNG outputs | Observational and forced actions produce state transitions | Pass / Reasonable |
| Lava | 24 cells; 21 PNG outputs | Multiple grid configurations render and advance | Pass / Reasonable |
| LunarLander | 7 cells; 2 PNG and 6 HTML animations | Six runs advance for 65–108 steps in the full audit; exceptions now propagate instead of being printed and swallowed | Pass / Reasonable |
| MAB (Ch. 7) | 16 cells; 7 PNG outputs | Observational recovery is about 0.05 instead of 0.30; `do(0)` about 0.08 instead of 0.40; `do(1)` about 0.04 instead of 0.30; stochastic policy about 0.06 instead of 0.35 | Pass / **Incorrect** |
| Masked Atari | Fails in cell 1 while ALE loads Pong | No output judgment is possible without the ROM | **Blocked** / Not checked |
| MDP (Ch. 7) | 22 cells; 17.13 s; 5 PNG outputs | Daily return 0.1005 vs 0.10; discounted return 1.0285 vs 1.0; policy return 8.1440 vs 8.2; transition rows sum to 1 and estimates are within ±0.05 | Pass / Pass |
| MNIST | 5 cells; 1 PNG | `see()` and `do()` both return valid binary observations/info and rendering produces a digit figure | Pass / Mechanics pass |
| MuJoCo random-friction Ant | 2 cells; 1 PNG and 1 HTML animation | RGB frames and a multi-step reward trace are generated | Pass / Mechanics pass |
| Race | 5 cells; 1 HTML animation | 40-step rollout renders without the former display-surface or animation-size error | Pass / Reasonable |
| Windy MiniGrid | 20 cells; 14 PNG and 1 HTML animation | Grid rollouts render and progress; only the pinned-compatible Matplotlib deprecation warning remains | Pass / Reasonable |

## Explicit bugs fixed in this branch

- Restricted package discovery in `setup.py` to `causal_gym` and its
  subpackages after reproducing the `Multiple top-level packages discovered`
  editable-install failure.
- Removed display-dependent `convert_alpha()` calls from the off-screen
  `rgb_array` fog paths in Race and Highway.
- Made AntMaze portable: fixed its syntax typo, replaced author-specific paths,
  removed forced CUDA/TensorBoard, used a single notebook-safe vector
  environment, and reduced the run to a flow check.
- Fixed LunarLander to pass a callable policy to `do()` and changed helper
  exception handling so execution failures cannot appear as success.
- Made MDP notebook evaluation bounded and notebook-safe, cast NumPy booleans
  before indexing, fixed its fresh-kernel f-string error, and corrected one
  reversed theoretical transition row.
- Corrected the MAB summary's action-frequency calculation; this does not alter
  the deferred SCM semantics.
- Reduced the Race example to 40 steps so its HTML animation remains below the
  default Jupyter embed limit.

## Deferred issues

1. **MAB semantic bug:** with confounding enabled, `MABSCM.step()` computes a
   probability using `U` and then draws another random number. This does not
   implement the notebook's stated structural equation
   `Y = I[U < 0.4 - D X]`. Fixing it changes public SCM semantics and belongs in
   a separately approved semantic branch.
2. **Atari ROM:** `pong.bin` is absent. ROM installation must remain a legal,
   user-supplied prerequisite; the output cannot be judged here.
3. **MDP return type:** `MDPSCM.state_transition()` currently produces a NumPy
   boolean. The notebook now casts to integer for array indexing; a public API
   type cleanup is deferred.
4. **Upstream deprecations:** Race reports an upstream `racetrack-v0` version
   warning, and Windy MiniGrid uses Matplotlib `tostring_rgb`. Both execute
   under the currently pinned dependencies.
5. **Editable install environment:** the package-discovery error is fixed, but
   a clean `pip install -e .` can still fail while building `box2d-py` if the
   local SWIG Python wrapper is unavailable. The existing environment passes
   `pip check` and imports `causal_gym`; dependency/toolchain cleanup is
   separate from notebook correctness.

No new pytest files, packaging tests, per-notebook tests, CI expansion, or
unapproved public causal API changes were added.
