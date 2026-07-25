# Notebook example maintenance changes — 2026-07-25

## Purpose and scope

This document records the implementation changes made on
`ey/notebook-example-maintenance`. The branch is based directly on
`origin/main` and is parallel to `ey/api-maintenance-smoke-tests`.

The two layers below are intentionally separate:

- **Code layer:** changes required for a notebook to execute from a fresh
  kernel, remain portable, propagate real failures, and generate its intended
  visual artifact.
- **Theory/result layer:** changes to an expected value, formula, experiment
  claim, or the standard used to judge the output.

No new pytest files, packaging tests, per-notebook tests, CI jobs, or public
causal API redesigns were added. Stored notebook outputs were not treated as
evidence; verification used cleared in-memory copies and fresh kernels.

## Repository-wide supporting changes

### Editable package discovery

File: `setup.py`

**Code layer**

The original editable install let setuptools perform automatic flat-layout
discovery. Because the repository also contains a top-level `reports/`
directory, the command:

```bash
python -m pip install -e .
```

failed with:

```text
Multiple top-level packages discovered in a flat-layout:
['reports', 'causal_gym']
```

Package discovery is now explicitly restricted:

```python
from setuptools import find_packages, setup

packages=find_packages(
    include=["causal_gym", "causal_gym.*"]
),
```

This makes `reports/` documentation rather than an installable Python package.
No packaging test file was added.

**Theory/result layer**

There is no causal or statistical change. This only determines which Python
packages are installed.

The discovery error is fixed. A separate local toolchain issue can still occur
while building `box2d-py` when the SWIG Python wrapper is unavailable. In the
existing environment, `python -m pip check` and
`python -c "import causal_gym"` both pass.

### Off-screen Race and Highway rendering

Files:

- `causal_gym/envs/race.py`
- `causal_gym/envs/highway.py`

**Code layer**

Both `rgb_array` fog overlays previously used:

```python
pygame.image.frombuffer(buffer, size, "RGBA").convert_alpha()
```

`convert_alpha()` depends on an initialized display surface. An off-screen
`rgb_array` render has no display mode, so it raised:

```text
pygame.error: No video mode has been set
```

The off-screen paths now use:

```python
pygame.image.frombuffer(buffer, size, "RGBA")
```

The source buffer already contains per-pixel alpha, so conversion is not
required. Display-dependent conversion in human-rendering paths was not
changed.

**Theory/result layer**

The SCM dynamics, observations, actions, rewards, and causal claims are
unchanged. This is only a rendering correction. The output standard is that
`rgb_array` rendering produces non-empty visual frames without opening a GUI.

## Per-example changes

### AntMaze

Notebook: `examples/test_antmaze.ipynb`

#### Code layer

The notebook initially had four independent execution/portability problems:

1. A syntax error: `return _init``` had two stray backticks.
2. Training and checkpoint paths were hard-coded under
   `/home/et2842/causal/antmaze_expert`.
3. Training and loading forced `device="cuda"`.
4. The default example launched 32 subprocess environments and requested
   10,000,000 timesteps.

The notebook now:

- uses `Path("antmaze_artifacts")` for a repository-relative artifact
  directory;
- fixes the factory return to `return _init`;
- replaces `SubprocVecEnv` with notebook-safe `DummyVecEnv`;
- reduces `NUM_ENVS` from 32 to 1;
- defines `TOTAL_TIMESTEPS = 128`;
- changes both train and load devices to `device="auto"`;
- removes optional TensorBoard logging, which previously required an
  undeclared TensorBoard installation;
- reduces `batch_size` from 512 to 64;
- disables the progress-bar dependency;
- saves and reloads
  `antmaze_artifacts/sac_her_antmaze_scm_example.zip`.

`learning_starts` is intentionally set above the 128-step budget. The example
therefore validates environment collection, SAC/HER integration, callback
execution, serialization, loading, prediction, and one evaluation step. It
does not claim to validate gradient optimization or policy quality.

#### Theory/result layer

No AntMaze equation or causal mechanism was changed. The experiment claim was
made narrower and explicit:

- **Before:** the notebook looked like a full expert-policy training job tied
  to one author's machine.
- **Now:** it is a portable end-to-end pipeline smoke example.

The correct result judgment is therefore **Flow pass**, not “the agent learned
the maze.” The fresh-kernel run completed 9 code cells and generated a
loadable checkpoint.

### LunarLander

Notebook: `examples/test_lunar_lander.ipynb`

#### Code layer

`LunarLanderPCH.do()` expects a policy callable. The notebook sampled an
integer action and passed it directly:

```python
env.do(action)
```

This produced:

```text
'numpy.int64' object is not callable
```

The notebook now wraps the sampled action in the required policy shape:

```python
env.do(lambda _obs, action=action: action)
```

The helper also previously caught reset, render, and step exceptions, printed
them, and returned or broke out of the episode. This allowed a notebook with a
step-0 failure and zero reward to appear successful to an execution runner.
Those handlers now raise contextual `RuntimeError` exceptions with the
original exception chained, so future failures make the notebook fail.

Finally, reset-time wind output now says that wind is initialized on the first
environment step instead of incorrectly reporting that wind retrieval failed.

#### Theory/result layer

No LunarLander physics, reward equation, wind distribution, or public API was
changed.

The validity of the demonstrated result did change:

- **Before:** each visual experiment stopped at step 0 while the helper hid
  the exception.
- **Now:** no-wind, positive-wind, negative-wind, and moderate-wind episodes
  advance through real environment steps and generate animations.

The output remains a random-policy visual demonstration. Landing performance
is not a strict theoretical target, so the result is judged by episode
progression and valid rendering rather than a required reward value.

### MAB (Chapter 7)

Notebook: `examples/test_mab (Ch 7).ipynb`

#### Code layer

The final comparison cell mislabeled conditional recovery rates as behavioral
action frequencies:

```python
p_x0_behavioral = recovery_given_arm0
p_x1_behavioral = recovery_given_arm1
```

It now computes actual action frequencies:

```python
behavioral_action_count = arm0_count + arm1_count
p_x0_behavioral = arm0_count / behavioral_action_count
p_x1_behavioral = arm1_count / behavioral_action_count
```

The displayed expected order was also corrected from `(0.8, 0.2)` to
`P(X=0)=0.2, P(X=1)=0.8`, consistent with the notebook's physician rule
`X = I[U <= 0.8]`. The rerun reported approximately `0.20` and `0.80`.

No code in `causal_gym/envs/mab.py` was modified.

#### Theory/result layer

The notebook's Chapter 7 expected values were **not** changed:

- observational `P(Y=1) = 0.30`;
- `P(Y=1 | do(X=0)) = 0.40`;
- `P(Y=1 | do(X=1)) = 0.30`;
- uniform stochastic-policy value `0.35`.

Fresh execution still produces values around:

- observational recovery: `0.05`;
- `do(X=0)`: `0.08`;
- `do(X=1)`: `0.04`;
- uniform stochastic policy: `0.06`.

Therefore the final status remains:

- **Execution:** Pass.
- **Output/theory:** Incorrect.

The deferred cause is the public `MABSCM.step()` mechanism. With confounding
enabled, it uses `U` to construct a success probability and then draws a
second random number. That is not the notebook's stated structural equation:

```text
Y = I[U < 0.4 - D X]
```

Changing that mechanism would alter public SCM semantics, so it is documented
for a separately approved semantic branch rather than fixed here.

### MDP (Chapter 7)

Notebook: `examples/test_mdp (Ch 7).ipynb`

#### Code layer

The original notebook launched hundreds or thousands of processes from
multiple Jupyter cells. Some cells combined 1,000–2,000 processes with
10,000-step episodes, causing the fresh-kernel audit to time out.

The notebook now:

- removes the `multiprocess` and `Process` imports;
- evaluates seeded episodes sequentially with `tqdm`;
- changes the common sample budget from 1,000 to 200;
- changes the common episode cap from 1,000 to 200;
- fixes `run_episode()` so it uses the common 200-step cap instead of a hidden
  10,000-step default;
- reduces interventional samples from 1,000 to 200;
- reduces observational samples from 5,000 to 500;
- reduces the final policy comparison from 1,000 to 200 episodes;
- removes a stray quote from the `NUM_EPISODES` line.

`MDPSCM.state_transition()` currently returns `numpy.bool_`. NumPy interprets
a boolean array index as a mask rather than the binary state index `0` or `1`,
which caused both transition entries to be incremented together. The notebook
now casts reset states, next states, and natural actions to `int` before using
them as array indices. The public return type was not changed.

Sequential execution also exposed an f-string that depended on a leaked loop
variable:

```python
f"P(S_{i+1}=..."
```

The mathematical label is now escaped as the literal `S_{i+1}` so a fresh
kernel does not raise `NameError`.

#### Theory/result layer

One expected transition row in the notebook was reversed. The implemented and
documented transition mechanism is:

```text
S' = (U1 XOR U2) XOR (S OR X)
```

Given `P(U1=1)=0.9` and `P(U2=1)=0.1`:

```text
P(U1 XOR U2 = 1) = 0.9*0.9 + 0.1*0.1 = 0.82
```

For `(S=1, X=1)`, `(S OR X)=1`, so the result is inverted:

```text
P(S'=0 | S=1, do(X=1)) = 0.82
P(S'=1 | S=1, do(X=1)) = 0.18
```

The notebook's expected-value dictionary previously listed these as
`0.18/0.82`; it now lists `0.82/0.18`.

After the integer-index and expected-row corrections:

- every estimated transition row sums to 1;
- daily profit is `0.1005` versus theory `0.10`;
- behavioral discounted return is about `1.03` versus theory `1.0`;
- the `X=S` policy daily profit is about `0.8185` versus theory `0.82`;
- its discounted return is about `8.14–8.16` versus theory `8.2`;
- interventional transition/reward estimates are within the audit tolerance
  of ±0.05.

The result is therefore **Pass (numeric, tolerance-based)**. The remaining
NumPy-boolean return type is recorded for later public API cleanup.

### Race

Files:

- `examples/test_race.ipynb`
- `causal_gym/envs/race.py`

#### Code layer

The environment-level off-screen rendering correction is described above:
the `rgb_array` fog path no longer calls display-dependent
`convert_alpha()`.

The notebook also generated a 100-step inline animation that could exceed
Jupyter's default 20 MB embed limit and silently drop later frames. The example
now uses 40 steps and explains that the limit is intentional.

#### Theory/result layer

No Race dynamics, perception mechanism, intervention semantics, or reward
formula was changed. Race is a visual/behavioral example, so it has no strict
textbook value to compare.

The result criterion is:

- the episode advances;
- `rgb_array` frames are non-empty;
- an HTML animation is actually produced;
- the animation stays below the default embed limit.

All four criteria pass. The upstream `racetrack-v0` version warning remains a
non-blocking dependency warning.

### Highway

Notebook: `examples/test_highway.ipynb` (notebook source unchanged)

Supporting file: `causal_gym/envs/highway.py`

#### Code layer

No Highway notebook cell was edited. The existing notebook failed because the
shared Highway `rgb_array` fog renderer called `convert_alpha()` without a
display mode. The supporting environment code now uses the RGBA buffer
directly, matching the Race fix.

#### Theory/result layer

No Highway SCM or behavioral claim changed. The notebook is judged visually:
the rollout must advance and create a real animation artifact. Fresh execution
completed and generated a 1,000 × 800 GIF of about 3 MB.

## Audit documentation changes

### `examples/README.md`

Added a concise 16-row audit table with independent columns for:

- execution status;
- output/theory status;
- requirements;
- known issues.

It reports 15 notebooks as executable and Masked Atari as Blocked by the
missing legal Pong ROM. It does not treat MAB's successful execution as proof
that its numerical output is correct.

### `reports/maintenance/notebook_example_audit_2026-07-25.md`

Added the detailed audit evidence: method, artifacts, numerical comparisons,
fixed bugs, and deferred issues.

This change document complements that report:

- the audit report answers **“What currently passes or fails?”**
- this document answers **“Exactly what changed, at the code and theory
  layers?”**

## Audited examples that required no source edit

The following notebooks passed the fresh-kernel execution and result checks
without a notebook-source change:

- `test_cartpole.ipynb`
- `test_cartpole_visual.ipynb`
- `test_dtr.ipynb`
- `test_frozenlake.ipynb`
- `test_highway_single_step.ipynb`
- `test_lava.ipynb`
- `test_mnist.ipynb`
- `test_mujoco_random_friction_ant.ipynb`
- `test_windyminigrid.ipynb`

`test_masked_atari.ipynb` was also not edited. It is correctly recorded as
Blocked because ALE cannot load `pong.bin`; no output/theory judgment is made
without a legally supplied ROM.

## Verification summary

- All 16 notebook JSON documents and all code cells compile.
- Every notebook was started in an independent fresh kernel.
- 15 notebooks execute completely.
- Masked Atari is Blocked only by the absent Pong ROM.
- MDP passes its tolerance-based numerical checks.
- MAB remains explicitly marked Incorrect at the theory/result layer.
- Race and Highway render headlessly.
- LunarLander exceptions can no longer be swallowed as apparent success.
- `python -m pip check` passes.
- `python -c "import causal_gym"` passes.
- `git diff --check` passes.
- `.venv/` remains excluded through `.git/info/exclude` and is not in the Git
  index.
