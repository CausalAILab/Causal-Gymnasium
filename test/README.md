# CausalGym Testing Area

This directory (`causalgym/test/`) serves as the primary location for all test scripts, notebooks, and related resources for the CausalGym framework and its custom environments.

## Directory Structure and Purpose

*   **Root of `test/`**: Contains general test files, often in the form of Jupyter notebooks (`.ipynb`) or Python scripts (`.py`), for various environments. These might include:
    *   Basic environment interaction tests.
    *   Visualization and rendering checks.
    *   Simple agent behavior demonstrations.
    *   Tests for specific causal mechanisms or interventions in the environments (e.g., `test_frozenlake.py`, `test_cartpole.ipynb`).
    *   `cartpole_visual_test.ipynb`: A notebook for visually inspecting the behavior of the `CartPoleWindPCH` environment under different wind conditions (no wind, default wind, heavy wind) and initial pole angle offsets. It helps verify the environment's dynamics and rendering.
    *   `test_lunar_lander.ipynb`: This notebook provides visual tests for the `LunarLanderSCM` environment, specifically focusing on the effects of different wind conditions (no wind, positive wind, negative wind, moderate wind) on the lander's trajectory and landing success. It includes helper functions for running episodes and displaying rendered frames as animations.

*   **`learning/` subdirectory**: This subdirectory is dedicated to more formal reinforcement learning experiments.
    *   It contains scripts for training and evaluating learning agents (like `UCBVI` and `UCBQ`) on various custom CausalGym environments. Key scripts include `train_lander_ucbvi.py`, `train_cartpole_ucbvi.py`, and `test_frozenlake_ucbvi.py`.
    *   For detailed information on these scripts, their setup, command-line arguments, and specific experiments, please refer to the dedicated README within this subdirectory: [`./learning/README.md`](./learning/README.md).

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



