# CausalGym Testing Area

This directory (`causalgym/test/`) serves as the primary location for all test scripts, notebooks, and related resources for the CausalGym framework and its custom environments.

## Directory Structure and Purpose

*   **Root of `test/`**: Contains general test files, often in the form of Jupyter notebooks (`.ipynb`) or Python scripts (`.py`), for various environments. These might include:
    *   Basic environment interaction tests.
    *   Visualization and rendering checks.
    *   Simple agent behavior demonstrations.
    *   Tests for specific causal mechanisms or interventions in the environments (e.g., `test_frozenlake.py`, `test_cartpole.ipynb`).

*   **`learning/` subdirectory**: This subdirectory is dedicated to more formal reinforcement learning experiments.
    *   It contains scripts for training and evaluating learning agents (like `UCBVI`) on the custom CausalGym environments.
    *   For detailed information on the scripts, setup, and experiments within this subdirectory, please refer to its specific README: [`./learning/README.md`](./learning/README.md).

*   **`logs/` subdirectory**: May contain log files generated during test runs or experiments.

## Environments Under Test

Tests in this area cover a range of environments, including but not limited to:

*   **Grid World Variants:**
    *   `FrozenLakeSCM` (Custom Windy FrozenLake with reward shaping)
    *   `WindyGridWorld`
    *   `WindyMiniGrid`
    *   `LavaCrossing` (various modes)
*   **Classic Control:**
    *   `CartPoleWind` (CartPole with wind effects)
*   **Lunar Lander:**
    *   `LunarLander` (with potential causal interventions)
*   **Simple MDPs:**
    *   `MDPExample` for foundational testing.

Many environments exist in `SCM` (Structural Causal Model) and `PCH` (Pearl's Causal Hierarchy) variants to support causal reasoning, interventions, and counterfactual queries.

## Running Tests

Refer to individual script/notebook documentation or the README within specific subdirectories (like `learning/`) for instructions on how to execute tests and interpret results.

---
*Previous content related to specific environment details and counterfactual additions has been integrated into environment-specific documentation or the more focused README in the `learning` subdirectory.*



