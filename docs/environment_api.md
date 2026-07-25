# Environment API Contract

This file defines the public contract shared by Causal-Gymnasium environments.
Environment-specific equations and experimental expectations belong in the
environment source, examples, or maintenance reports.

## SCM contract

Every concrete SCM must provide:

```python
action_space
observation_space
reset(seed=None, options=None)
step(action, ...)
action()
observation()
get_graph
```

`reset()` returns:

```text
observation, info
```

`step()` returns the Gymnasium tuple in this exact order:

```text
observation, reward, terminated, truncated, info
```

For normal execution:

```python
env.observation_space.contains(observation)
env.action_space.contains(action)
```

must be true. Stateless environments must still expose a valid constant
observation and matching observation space.

## PCH contract

Every concrete PCH wrapper exposes:

```python
see()
do(do_policy)
ctf_do(ctf_policy)
```

All three methods return the same five fields and ordering as `step()`.

The preferred reusable policy signatures are:

```python
do_policy(observation)
ctf_policy(observation, natural_action)
```

If an environment needs a richer observation, that information should be
represented by one observation object rather than a different policy calling
convention.

## Counterfactual invariant

`ctf_do()` must hold the exogenous context fixed between the factual action and
the counterfactual outcome. It may change the intervention, but it must not
resample the hidden randomness being conditioned on.

For example, if `u1` determines both the natural action and a transition, the
same sampled `u1` must be used for both computations.

## Graph consistency

`get_graph` must describe the implemented SCM:

- latent variables used by the mechanism must be represented;
- a bidirected edge must only represent actual unobserved confounding;
- graph node and edge names must match the environment variables.

## Minimum API validation

An environment-level smoke test should verify:

- construction directly and, when registered, through `gymnasium.make()`;
- seeded `reset()` and valid spaces;
- five-field results from `see()`, `do()`, and `ctf_do()`;
- the order and types of `terminated`, `truncated`, and `info`.

These checks validate interface compatibility. Numerical agreement with causal
equations or textbook results requires separate semantic tests.
