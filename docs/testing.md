# Testing

This document explains the current testing plan for Causal-Gymnasium.

The goal is not to prove every environment is scientifically correct in one test. The goal is to catch basic API failures early, then add more specific tests for causal semantics.

## Why smoke tests?

Smoke tests are quick checks.

They answer:

```text
Does the basic interface work?
```

They do not answer:

```text
Is the full causal model correct?
Does the algorithm result match the paper?
Is the environment ready for final experiments?
```

Smoke tests are still useful because many maintenance bugs are basic interface bugs.

Examples:

- package import fails
- registered environment ID points to a class that does not exist
- `reset()` returns the wrong shape
- `action_space` is missing
- `see()` or `do()` crashes
- `terminated` and `truncated` are returned in the wrong order

## Current smoke test

The current smoke test file is:

```text
tests/test_env_api_smoke.py
```

It covers lightweight environments:

- `CartPoleWindPCH`
- `DTRPCH`
- `FrozenLakePCH`
- `MABPCH`
- `MDPPCH`
- `RobotWalkPCH`

It checks:

- `reset()`
- `action_space`
- `observation_space`
- `see()`
- `do()`
- `ctf_do()`
- selected Gymnasium registry entry points

## How to run

From the repository root:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/private/tmp/causal_gym_mpl XDG_CACHE_HOME=/private/tmp/causal_gym_cache python -m pytest tests/test_env_api_smoke.py -q
```

Expected result:

```text
13 passed
```

The `MPLCONFIGDIR` and `XDG_CACHE_HOME` variables avoid writing Matplotlib and font cache files into the user home directory.

## Why did the first manual smoke test pass?

The first manual smoke test only checked a direct import path.

Example:

```python
from causal_gym.envs import MDPPCH
env = MDPPCH()
```

This can pass even if the Gymnasium registry is wrong.

The public registry path is different:

```python
import gymnasium as gym
import causal_gym

env = gym.make("causal_gym/MDPExample-v0")
```

This path uses the registered entry point in `causal_gym/envs/__init__.py`.

So the distinction is:

```text
direct import path: developer uses the class directly
registry path: user or baseline uses the Gymnasium environment ID
```

Both paths matter.

## Test levels

The maintenance plan should use several test levels.

### Level 1: import smoke test

Checks:

- package imports
- selected modules import
- no unrelated optional dependency breaks a lightweight import

### Level 2: API contract smoke test

Checks:

- `reset()`
- `see()`
- `do()`
- `ctf_do()`
- `action_space`
- `observation_space`
- Gymnasium registry

This is the current test level.

### Level 3: causal semantic tests

Checks causal meaning.

Examples:

- `ctf_do()` reuses the same exogenous variables
- graph edges match the SCM mechanism
- no bidirected edge is added when there is no hidden confounding
- latent variables used in the SCM appear in the graph

These tests catch bugs that do not crash at runtime.

### Level 4: experiment-level tests

Checks empirical behavior.

Examples:

- Monte Carlo estimates match known theoretical values
- observational estimates differ from interventional estimates when the SCM is confounded
- causal baseline and non-causal baseline can both run on the example

These tests are slower and should be added after the API is stable.

## Heavy environments

Some environments need extra dependencies or assets:

- MuJoCo
- Atari
- MNIST
- OGBench
- Gymnasium Robotics

These should not block the lightweight smoke test.

They should have separate optional tests, for example:

```bash
python -m pytest tests/test_heavy_envs.py -q
```

or:

```bash
python -m pytest -m heavy
```

This keeps basic maintenance fast while still allowing full reproducibility checks later.

## Current rule

For normal maintenance work:

```text
Run lightweight smoke tests first.
Fix API bugs before adding new environments.
Add causal semantic tests when fixing counterfactual or graph behavior.
Keep heavy dependency tests separate.
```
