# Documentation Update Report

Date: 2026-07-24

Branch: `ey/api-maintenance-smoke-tests`

## 1. What problem was changed?

This round added clearer project documentation for maintenance work.

The goal was to make the branch easier to review and easier to maintain later.

The changes include:

- Added a formal environment API contract document.
- Added a testing document explaining the smoke tests.
- Added README links to the new docs and maintenance reports.
- Added a README testing command for the current lightweight smoke test.

## 2. Where was the problem?

Before this change, the maintenance explanation mainly existed in:

- the PDF draft
- the screenshot TODO list
- the conversation
- the maintenance report

That was useful for planning, but not enough for people who open the repository later.

The repository did not yet have a clear place explaining:

- what API each environment should satisfy
- why `action_space` and `observation_space` matter
- why `ctf_do()` needs to reuse the same hidden context
- what the smoke test checks
- what the smoke test does not check
- why heavy environments should be tested separately

## 3. Why was this a problem?

This is a problem for maintenance and reproducibility.

If the expected API is only known from conversation, then future contributors may not know what they are supposed to preserve.

If the smoke test is added without explanation, reviewers may not know why it exists or why it does not cover every environment.

If README does not link to the documentation, then the docs are easy to miss.

For a research codebase, documentation should make the following clear:

- what the code is supposed to do
- what counts as a bug
- how to run the basic checks
- what is intentionally out of scope for the current test

## 4. How was it changed?

Added:

```text
docs/environment_api.md
```

This document explains:

- the minimum SCM API
- the minimum PCH API
- `reset()` and `step()` return formats
- `action_space` and `observation_space`
- preferred policy signatures
- the counterfactual rule for `ctf_do()`
- graph requirements
- what counts as an API bug
- current lightweight maintenance scope

Added:

```text
docs/testing.md
```

This document explains:

- why smoke tests are useful
- what the current smoke test checks
- why the earlier manual smoke test did not catch registry bugs
- the difference between direct import path and Gymnasium registry path
- test levels from import smoke tests to experiment-level tests
- why heavy environments should be tested separately

Updated:

```text
README.md
```

The README now links to:

- `docs/environment_api.md`
- `docs/testing.md`
- `reports/maintenance/`

It also includes the command for running the lightweight API smoke test.

## Verification

The smoke tests were run after the documentation changes:

```bash
MPLCONFIGDIR=/private/tmp/causal_gym_mpl XDG_CACHE_HOME=/private/tmp/causal_gym_cache .venv/bin/python -m pytest tests/test_env_api_smoke.py -q
```

Result:

```text
13 passed
```
