# CausalGym: A Library for Causal Reinforcement Learning Experiments

## Overview

**CausalGym** is a Python library designed for developing and testing custom reinforcement learning environments with a particular focus on incorporating causal knowledge. It provides a framework for creating Structural Causal Model (SCM) based environments and their corresponding Pearl Causal Hierarchy (PCH) interface for learning and acting, facilitating research in causal reinforcement learning.

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
    *   **`FrozenLakePCH`**: The corresponding PCH for this custom SCM.

## Navigating the CausalGym Repository
*   **Core Environments**: Located in `causal_gym/envs/`.
*   **Core Framework Logic**: The `causal_gym/core/` directory contains fundamental classes and utilities for the CausalGym framework.
*   **Examples & Demonstrations**: We provide Jupyter notebooks (e.g., for testing environments like `frozen_lake_test.ipynb` or demonstrating features) under the `test/` directory, importing and using the CausalGym library.

## Installation and Usage
1.  **Installation**: Clone this repository. To install the CausalGym library and its dependencies, navigate to the root directory of the cloned `causalgym` repository and run:
    ```bash
    pip install -e .
    ```
2.  **Dependencies**: Requires `pygame`, `numpy`, `gymnasium`, and other common scientific Python libraries (see `setup.py` for full list of dependencies).
3.  **Usage**: Import environments from `causalgym.causal_gym.envs` in your Python scripts or notebooks.

## Future Development of CausalGym
*   Expansion with more diverse and complex causal environments.
*   Enhanced interfaces for defining and intervening on environment variables.
*   More examples and documentation for users and developers.
