# Environment API Contract

This document records the minimum API contract for Causal-Gymnasium environments.

The goal is maintenance. Each environment can have different causal mechanisms, but the public interface should stay predictable. This makes it easier to run baselines, write examples, add tests, and compare causal and non-causal methods.

## Basic rule

Each environment should satisfy two layers:

1. Standard Gymnasium API
2. Causal PCH API

The standard Gymnasium API is for compatibility with RL tools.

The PCH API is for causal interaction:

- `see()`: observe the natural behavior policy
- `do(policy)`: intervene with a policy
- `ctf_do(policy)`: run a counterfactual intervention

## SCM requirements

Every SCM should define:

```python
reset(seed=None, options=None)
step(action, ...)
action()
observation()
get_graph
action_space
observation_space
```

The base class contains the expected shape of the interface, but subclasses still need to implement the actual behavior.

### `reset()`

`reset()` should return:

```text
observation, info
```

where:

- `observation` is the initial observation available to the agent
- `info` is a dictionary

The returned observation should satisfy:

```python
env.observation_space.contains(observation)
```

### `step()`

`step()` should return the Gymnasium 5-tuple:

```text
observation, reward, terminated, truncated, info
```

The order matters.

Wrong order:

```text
observation, reward, truncated, terminated, info
```

This can silently break training and evaluation because both `terminated` and `truncated` are booleans. Python will not crash, but the algorithm will understand the episode ending incorrectly.

### `action_space`

Each environment should define `action_space`.

For example, a two-arm bandit should use:

```python
self.action_space = spaces.Discrete(2)
```

This allows standard code to sample a valid action:

```python
env.action_space.sample()
```

### `observation_space`

Each environment should define `observation_space`.

Even if an environment has no meaningful state, it should still return a valid observation.

For example, a stateless bandit can use:

```python
self.observation_space = spaces.Discrete(1)
```

and:

```python
def observation(self):
    return 0
```

This keeps the environment compatible with Gymnasium without changing the causal meaning of the model.

## PCH requirements

Each PCH environment should expose:

```python
see()
do(do_policy)
ctf_do(ctf_policy)
```

Each method should return:

```text
observation, reward, terminated, truncated, info
```

## Policy input convention

The preferred convention is:

```python
do_policy(observation)
ctf_policy(observation, natural_action)
```

This convention lets the same policy shape work across environments.

Some older environments used different policy signatures. For example:

- `MABPCH.do()` called `do_policy()` with no observation.
- `DTRPCH.do()` used different inputs depending on the treatment stage.
- Some driving environments passed many separate variables instead of one observation object.

This is hard to reuse across baselines.

When possible, new code should use one observation object as the policy input.

## Counterfactual rule

`ctf_do()` must reuse the same exogenous context between the natural world and the counterfactual world.

This is important.

A counterfactual asks:

```text
Given the same hidden randomness, what would have happened under a different action?
```

So this is wrong:

```python
u1, u2, u3 = env.sample_u()
natural_action = env.action(env.s, env._u1())
action = ctf_policy(env.s, natural_action)
result = env.step(action, u1, u2, u3)
```

The natural action used a new `u1`, not the original one.

This is correct:

```python
u1, u2, u3 = env.sample_u()
natural_action = env.action(env.s, u1)
action = ctf_policy(env.s, natural_action)
result = env.step(action, u1, u2, u3)
```

The natural and counterfactual worlds now share the same causal context.

## Graph rule

The graph should match the SCM mechanism.

Examples:

- If there is no unobserved confounding, do not add a bidirected edge.
- If the mechanism uses a latent variable `U`, include `U` in the graph node list.
- If `get_graph` is not implemented, downstream causal analysis cannot rely on the environment.

The graph is not only a picture. It is part of the environment's causal documentation.

## What counts as a bug?

These should be treated as API bugs:

- `gymnasium.make(...)` cannot create a registered environment.
- `reset()` does not return `(observation, info)`.
- `step()`, `see()`, `do()`, or `ctf_do()` does not return a 5-tuple.
- `terminated` and `truncated` are returned in the wrong order.
- `action_space` or `observation_space` is missing.
- `observation_space.contains(observation)` fails for a normal reset observation.
- `ctf_do()` samples a different hidden context for the natural and counterfactual worlds.
- The graph shows causal relations that are not in the SCM.

## Current maintenance scope

For the current maintenance branch, the first target is the lightweight environments:

- `MABPCH`
- `MDPPCH`
- `DTRPCH`
- `CartPoleWindPCH`
- `FrozenLakePCH`
- `RobotWalkPCH`

Heavy environments such as MuJoCo, Atari, MNIST, and OGBench should be tested separately because they need extra dependencies, assets, or system setup.
