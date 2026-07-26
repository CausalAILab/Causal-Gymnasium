# Causal Semantics Fixes Report

Date: 2026-07-25

Branch: `ey/causal-semantics-fixes`

Base: API maintenance commit `988877e`

## Scope

This branch fixes confirmed causal or episode-semantics defects in the MAB,
MDP, and DTR environments. It does not modify notebook source files, redesign
the public causal API, or add broad CI infrastructure.

The branch is stacked on `ey/api-maintenance-smoke-tests` because the semantic
tests rely on the repaired SCM/PCH interface from that branch. After both code
branches are integrated, `ey/notebook-example-maintenance` can be rebased and
its MAB audit result updated from incorrect to passing.

## MAB

### Code-level problem

The Chapter 7 SCM defines one structural threshold equation:

```text
U ~ Uniform(0, 1)
X = I(U < 0.8)
Y = I(U < 0.4 - 0.1 X)
```

The previous implementation first subtracted `U` from the threshold and then
used the result as a probability for a second random draw. That changed the
SCM itself. It produced approximately `0.08` for `do(X=0)` and `0.045` for
`do(X=1)`, rather than the stated `0.4` and `0.3`.

The default action policy also did not implement `X = I(U < 0.8)` despite its
comment describing a standard behavioral policy.

### Code-level fix

- The reward is now the deterministic structural threshold
  `I(U_Y < arms_probs[X])`; there is no second reward draw.
- The default behavioral policy is now `I(U < 0.8)`.
- `confounding_strength=1` shares the action and reward exogenous variable,
  reproducing the Chapter 7 SCM.
- `confounding_strength=0` uses independent action and reward exogenous draws.
- Intermediate strengths are defined as the probability of sharing the same
  exogenous draw. This preserves the interventional arm marginals while giving
  the existing parameter a coherent continuous meaning.
- Invalid strengths and invalid arm probabilities are rejected early.

### Theory-level verification

During development, a deterministic grid check verified:

```text
P(X=0) = 0.2
P(X=1) = 0.8
P(Y=1) = 0.3
P(Y=1 | X=0) = 0
P(Y=1 | X=1) = 0.375
P(Y=1 | do(X=0)) = 0.4
P(Y=1 | do(X=1)) = 0.3
P(Y=1 | uniform intervention policy) = 0.35
```

The Chapter 7 notebook was then executed from a fresh kernel. All 16 code
cells completed with no exception and produced seven inline figures. Its
Monte Carlo results agreed with the targets, including `0.3966` for
`do(X=0)`, `0.3013` for `do(X=1)`, and `0.3471` for the uniform policy.

## MDP

### Code-level problem

An episode configured with `max_step=N` truncated only after step `N+1`
because the termination check used `num_step > max_step`.

The API branch had already fixed a separate counterfactual defect: `ctf_do()`
must reuse the same sampled `u1` for the natural action and counterfactual
transition. This branch does not reimplement that API fix; it adds a semantic
regression check for the invariant.

### Code-level fix

The horizon check now uses `num_step >= max_step`, so the episode truncates at
the configured step.

### Theory-level verification

Development checks verified both the exact horizon and same-`u1`
counterfactual invariant. The Chapter 7 MDP notebook was executed from a fresh
kernel: all 22 code cells completed with no exception and produced five inline
figures. Representative results were within Monte Carlo error of the notebook
targets:

```text
behavioral daily profit: 0.1005, target 0.10
behavioral discounted return: 1.0285, target 1.0
do(X=S) daily profit: 0.8185, target 0.82
do(X=S) discounted return: 8.1440, target 8.2
```

## DTR

### Code-level problem

The implemented graph did not match the structural equations. It contained
nonexistent direct dependencies, marked ordinary latent-variable arrows with
an unsupported `latent` edge type, and connected the shared confounder `U` to
variables that use independent disturbances.

The printed outcome equation in the
[official technical report](https://www.causalai.net/r65.pdf) and
[latest arXiv version](https://arxiv.org/abs/2606.24160) omits an independent
logistic outcome disturbance. Taken literally, it cannot reproduce the
reference Q table or the reported optimal-policy reward. The same reference
lists an additional exogenous `U5`; integrating a Logistic `U5` reproduces
every stated Q value.

### Code-level fix

- Added the independent logistic outcome disturbance to the default outcome
  structural equation.
- Rebuilt the graph from the implemented parent relationships.
- Kept `U -> Y` and added `U -> X1` or `U -> X2` only when the corresponding
  confounding coefficient is nonzero.
- Represented all latent-node causal arrows as supported directed edges.

### Theory-level verification

A deterministic numerical integration test reproduces the five reference
Q values within `0.001`:

```text
0.9846294445
0.7851467237
0.2148532763
0.0153705555
0.0007840943
```

Development checks also covered the unconfounded graph, conditionally active
confounder edges, and independent outcome disturbance. The DTR notebook was
executed from a fresh kernel: all six code cells completed with no exception
and produced one inline figure. It is a workflow/comparison notebook rather
than a strict reproduction of the Q table, so its random summary is not a
strict numerical reproduction of the reference Q table.

## Verification

Inherited API smoke tests:

```bash
MPLCONFIGDIR=/private/tmp/causal_gym_mpl \
XDG_CACHE_HOME=/private/tmp/causal_gym_cache \
.venv/bin/python -m pytest \
  tests/test_env_api_smoke.py -q
```

Result:

```text
13 passed
```

Focused semantic checks were used during development, but the standalone
`tests/test_causal_semantics.py` file is not retained on this branch.

Fresh-kernel notebook verification used copies from
`ey/notebook-example-maintenance`; executed copies were written only under
`/private/tmp` and were not added to Git.

No pull request was created.
