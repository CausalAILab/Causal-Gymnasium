# API Maintenance and Smoke Tests Report

Date: 2026-07-24

Branch: `ey/api-maintenance-smoke-tests`

## 1. What problem was changed?

This round focused on the first maintenance pass for Causal-Gymnasium.

The changes include:

- Renamed the branch from `ey/env-api-smoke-tests` to `ey/api-maintenance-smoke-tests`.
- Removed the unused `.venv39` virtual environment.
- Kept `.venv` as the local development environment.
- Added `.venv/` to `.gitignore`.
- Fixed wrong Gymnasium registry entry points.
- Fixed the return order in `RobotWalkPCH`.
- Added basic Gymnasium API fields to lightweight SCM environments.
- Fixed one counterfactual bug in `MDPPCH.ctf_do()`.
- Added the first automated API smoke tests.

## 2. Where was the problem?

The problems were in these files:

- `.gitignore`
- `causal_gym/envs/__init__.py`
- `causal_gym/envs/robowalk.py`
- `causal_gym/envs/mab.py`
- `causal_gym/envs/mdp.py`
- `causal_gym/envs/dtr.py`
- `tests/test_env_api_smoke.py`

More specifically:

- In `causal_gym/envs/__init__.py`, two registered Gymnasium entry points pointed to classes that do not exist:
  - `WindyGridWorldEnv`
  - `MDPExamplePCH`

- In `causal_gym/envs/robowalk.py`, `RobotWalkPCH.see()` and `RobotWalkPCH.do()` returned:

```text
state, reward, truncated, terminated, info
```

but Gymnasium expects:

```text
state, reward, terminated, truncated, info
```

- In `causal_gym/envs/mab.py`, `MABSCM` did not define:
  - `action_space`
  - `observation_space`
  - `observation()`

- In `causal_gym/envs/mdp.py`, `MDPSCM` did not define:
  - `action_space`
  - `observation_space`
  - `observation()`

- In `causal_gym/envs/dtr.py`, `DTRSCM` did not define:
  - `action_space`
  - `observation_space`
  - `observation()`

- Also in `causal_gym/envs/dtr.py`, custom structural functions were stored but not called.

- In `causal_gym/envs/mdp.py`, `MDPPCH.ctf_do()` sampled one exogenous variable for the counterfactual world but used a different one to generate the natural action.

## 3. Why was this a problem?

These problems matter because they break maintenance and reproducibility.

If a Gymnasium registry entry point is wrong, then a user can call `gymnasium.make(...)` and the environment will fail to load.

If `terminated` and `truncated` are returned in the wrong order, algorithms may misunderstand why an episode ended. This can silently affect training, evaluation, and reported results.

If an environment does not define `action_space` and `observation_space`, then standard Gymnasium tools and many RL baselines cannot interact with it properly. For example, code like this may fail:

```python
env.action_space.sample()
env.observation_space.contains(observation)
```

If an SCM does not implement `observation()`, then counterfactual methods such as `ctf_do()` can fail when they need the current observation.

If `MDPPCH.ctf_do()` does not reuse the same exogenous variable, then the result is not a true counterfactual. A counterfactual comparison should keep the hidden randomness fixed between the natural world and the counterfactual world.

If there are no smoke tests, these basic API problems can come back later without being noticed.

## 4. How was it changed?

The branch was renamed to:

```bash
ey/api-maintenance-smoke-tests
```

The unused Python 3.9 virtual environment was removed:

```text
.venv39
```

The active local environment is:

```text
.venv
```

`.gitignore` now ignores:

```text
.venv/
```

In `causal_gym/envs/__init__.py`:

- `WindyGridWorldEnv` was changed to `WindyMiniGridPCH`.
- `MDPExamplePCH` was changed to `MDPPCH`.

In `causal_gym/envs/robowalk.py`:

- `RobotWalkPCH.see()` and `RobotWalkPCH.do()` now return:

```text
state, reward, terminated, truncated, info
```

In `causal_gym/envs/mab.py`:

- Added:

```python
self.action_space = spaces.Discrete(2)
self.observation_space = spaces.Discrete(1)
```

- Added `observation()`, returning the constant observation `0`.
- Updated `reset()` and `step()` to return this observation.
- Clipped `success_prob` into the valid probability range `[0, 1]`.
- Only adds the bidirected edge `X <-> Y` when `confounding_strength != 0`.
- Updated `MABPCH.do()` so it supports the standard policy form:

```python
policy(observation)
```

while still supporting the old no-argument policy form:

```python
policy()
```

In `causal_gym/envs/mdp.py`:

- Added `action_space`.
- Added `observation_space`.
- Added `observation()`.
- Updated `MDPPCH.ctf_do()` so the natural action and counterfactual outcome use the same `u1`.

In `causal_gym/envs/dtr.py`:

- Added `action_space`.
- Added `observation_space`.
- Added `observation()`.
- Fixed custom structural functions so they are called instead of only being stored.
- Added latent node `U` to the graph.

Added a new smoke test file:

```text
tests/test_env_api_smoke.py
```

The smoke tests check lightweight environments:

- `CartPoleWindPCH`
- `DTRPCH`
- `FrozenLakePCH`
- `MABPCH`
- `MDPPCH`
- `RobotWalkPCH`

The tests verify:

- `reset()`
- `action_space`
- `observation_space`
- `see()`
- `do()`
- `ctf_do()`
- selected Gymnasium registry entry points

## Verification

The smoke tests were run with:

```bash
MPLCONFIGDIR=/private/tmp/causal_gym_mpl XDG_CACHE_HOME=/private/tmp/causal_gym_cache .venv/bin/python -m pytest tests/test_env_api_smoke.py -q
```

Result:

```text
13 passed
```
