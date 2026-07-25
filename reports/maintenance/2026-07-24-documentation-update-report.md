# Documentation Update Report

Date: 2026-07-24

Branch: `ey/api-maintenance-smoke-tests`

## 1. What problem was changed?

This round added clearer project documentation for maintenance work.

The goal was to make the branch easier to review and easier to maintain later.

The final changes include:

- Added a concise environment API contract.
- Kept the smoke-test command and scope directly in the README.
- Added README links to the API contract and maintenance reports.
- Removed a separate testing document that duplicated the README, tests, and
  maintenance report.

## 2. Where was the problem?

Before this change, the maintenance explanation mainly existed in:

- the PDF draft
- the screenshot TODO list
- the conversation
- the maintenance report

That was useful for planning, but not enough for people who open the repository later.

The repository did not yet have a concise normative place explaining:

- what API each environment should satisfy
- why `action_space` and `observation_space` matter
- why `ctf_do()` needs to reuse the same hidden context
- what the smoke test validates at the API layer

## 3. Why was this a problem?

This is a problem for maintenance and reproducibility.

If the expected API is only known from conversation, then future contributors may not know what they are supposed to preserve.

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

This concise contract records only durable requirements:

- the minimum SCM API
- the minimum PCH API
- `reset()` and `step()` return formats
- `action_space` and `observation_space`
- preferred policy signatures
- the counterfactual rule for `ctf_do()`
- graph requirements
- the distinction between API checks and semantic tests

Updated:

```text
README.md
```

The README now links to:

- `docs/environment_api.md`
- `reports/maintenance/`

It also contains the command and scope note for the lightweight API smoke
test. Keeping that short operational guidance in the README avoids a second
document that would become stale as test coverage changes.

## Verification

The smoke tests were run after the documentation changes:

```bash
MPLCONFIGDIR=/private/tmp/causal_gym_mpl XDG_CACHE_HOME=/private/tmp/causal_gym_cache .venv/bin/python -m pytest tests/test_env_api_smoke.py -q
```

Result:

```text
13 passed
```
