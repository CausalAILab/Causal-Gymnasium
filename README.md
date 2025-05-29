# CausalGym: A Library for Causal Reinforcement Learning Experiments

## Overview

**CausalGym** is a Python library designed for developing and testing custom reinforcement learning environments with a particular focus on incorporating causal structures and confounders. It provides a framework for creating Structural Causal Model (SCM) based environments and their corresponding Probabilistic Causal Hacked (PCH) versions, facilitating research in causal reinforcement learning.

This repository contains the core CausalGym library.

## Key Environments within CausalGym

CausalGym hosts several custom environments, primarily within its `causal_gym/envs/` directory. These environments are used to explore agent interactions with various causal factors.

### 1. Lunar Lander with Latent Confounder
*   **Environment Module**: `causal_gym.envs.lunar_lander` (or similar, intended to contain `LunarLanderPCH` and related SCMs)
    *   Provides a modification of the classic LunarLander where a latent wind confounder can affect the lander's dynamics.
    *   Often used with discretization for tabular RL agents.

### 2. Custom FrozenLake Environment with Per-Cell Wind
*   **Environment Module**: `causal_gym.envs.frozen_lake`
    *   **`FrozenLakeSCM`**: Implements a FrozenLake grid world with cell-specific wind conditions defined by a `wind_map` sampled per episode.
    *   **Programmatic Rendering**: Features `pygame`-based `rgb_array` rendering for clear visualization of tiles, the agent, and wind indicators.
    *   **`FrozenLakePCH`**: The corresponding PCH for this custom SCM.

### 3. Support for CartPole Experiments
*   CausalGym is designed to support CartPole variations where causal factors are introduced. Users can create their own PCH scripts (e.g., `cartpole_pch.py`, `cartpole_wind_pch.py`) that leverage the CausalGym framework to build and test such environments.

## Learning Algorithms

CausalGym environments are designed to be used with various reinforcement learning algorithms.
*   **Core Implementations**: The `causal_gym/algorithms/` directory is intended for implementations of algorithms like UCBVI (Upper Confidence Bound Value Iteration) and UCBQ (Upper Confidence Bound Q-learning) that are part of the CausalGym library.
*   **User-Defined Algorithms**: Users can develop their own algorithm scripts (e.g., a `linear_ucbvi.py` for specific experiments) that interface with CausalGym's environments.

## Navigating the CausalGym Repository

*   **Core Environments**: Located in `causal_gym/envs/`.
*   **Algorithm Implementations**: Intended for `causal_gym/algorithms/`.
*   **Core Framework Logic**: The `causal_gym/core/` directory contains fundamental classes and utilities for the CausalGym framework.
*   **Tests**: The `test/` directory (e.g., `causalgym/test/`) contains unit tests or basic environment interaction tests for CausalGym components.
*   **Examples & Demonstrations**: Users typically create Jupyter notebooks (e.g., for testing environments like `frozen_lake_test.ipynb` or demonstrating features) in their own project directories, importing and using the CausalGym library.

## Installation and Usage
1.  **Installation**: Clone this repository. To install the CausalGym library and its dependencies, navigate to the root directory of the cloned `causalgym` repository and run:
    ```bash
    pip install -e .
    ```
2.  **Dependencies**: Requires `pygame`, `numpy`, `gymnasium`, and other common scientific Python libraries (see `setup.py` for full list of dependencies).
3.  **Usage**: Import environments from `causalgym.causal_gym.envs` and algorithms from `causalgym.causal_gym.algorithms` in your Python scripts or notebooks.
4.  **Path Configuration**: When using CausalGym from scripts or notebooks in your own project directories, ensure Python's import system can discover the `causalgym` modules (e.g., by installing it via `pip install .` as above, or by adjusting `PYTHONPATH`).

## Future Development of CausalGym

*   Expansion with more diverse and complex causal environments.
*   Enhanced tools for defining and intervening on causal variables.
*   Standardization of interfaces for broader algorithm compatibility.
*   More examples and documentation for users and developers, potentially within an `examples/` directory in this repository.
