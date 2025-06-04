# CausalGym: A Library for Causal Reinforcement Learning Experiments

## Overview

**CausalGym** is a Python library designed for developing and testing custom reinforcement learning environments with a particular focus on incorporating causal structures and confounders. It provides a framework for creating Structural Causal Model (SCM) based environments and their corresponding Pearl Causal Hierarchy (PCH) versions, facilitating research in causal reinforcement learning.

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
*   **Core Implementations**: The `causal_gym/algorithms/` directory is intended for implementations of algorithms like UCBVI (Upper Confidence Bound Value Iteration) and UCBQ (optimistic Q-builder) that are part of the CausalGym library.
*   **User-Defined Algorithms**: Users can develop their own algorithm scripts (e.g., a `linear_ucbvi.py` for specific experiments) that interface with CausalGym's environments.

## Navigating the CausalGym Repository

*   **Core Environments**: Located in `causal_gym/envs/`.
*   **Algorithm Implementations**: Intended for `causal_gym/algorithms/`.
*   **Core Framework Logic**: The `causal_gym/core/` directory contains fundamental classes and utilities for the CausalGym framework.
*   **Tests**: The `test/` directory (e.g., `causalgym/test/`) contains unit tests or basic environment interaction tests for CausalGym components.
*   **Examples & Demonstrations**: Users typically create Jupyter notebooks (e.g., for testing environments like `frozen_lake_test.ipynb` or demonstrating features) in their own project directories, importing and using the CausalGym library.

## Installation and Usage

1. **Installation**: Clone this repository and install CausalGym in a virtual environment:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   
   # On Windows:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\venv\Scripts\activate
   
   # On Unix/MacOS:
   source venv/bin/activate
   
   # Install CausalGym
   pip install -e .
   ```
   To install with development dependencies (e.g., `ipykernel` for notebooks), use:
   ```bash
   pip install -e .[dev]
   ```

2. **Dependencies**: CausalGym requires several Python packages:
   - Core dependencies: `pygame`, `numpy`, `gymnasium`
   - Additional scientific libraries (see `setup.py` for complete list)

3. **Usage**: Import CausalGym components in your Python code:
   ```python
   from causalgym.causal_gym.envs import FrozenLakePCH, CartPoleWindPCH
   from causalgym.causal_gym.algorithms import UCBVI, UCBQ
   ```

4. **Path Configuration**: Ensure Python can find CausalGym modules by either:
   - Installing via `pip install -e .` (recommended)
   - Adding the repository to your `PYTHONPATH`

## Future Development of CausalGym

*   Expansion with more diverse and complex causal environments.
*   Enhanced tools for defining and intervening on causal variables.
*   Standardization of interfaces for broader algorithm compatibility.
*   More examples and documentation for users and developers, potentially within an `examples/` directory in this repository.

# CausalGym PCH Environments: FrozenLake and CartPoleWind

This document provides an overview of two PCH (Pearl Causal Hierarchy) environments available in this CausalGym setup: `FrozenLakePCH` and `CartPoleWindPCH`. It details their mechanics, latent variables, and how to interact with them using the `see()` and `do()` methods.

## General PCH Interaction

PCH environments wrap an SCM (Structural Causal Model) and provide two primary methods for interaction:

*   `see()`: Allows observation of the environment. Its exact behavior (whether it advances the environment state or not) can vary by specific PCH implementation. It typically reveals what action the SCM's internal policy would take.
*   `do(action)`: Allows intervention by executing a chosen `action` in the environment, advancing its state and returning the outcome.

---

## 1. FrozenLakePCH Environment (`causal_gym/FrozenLakePCH-v0`)

**Description:**
The FrozenLake environment is a classic grid world problem. The agent controls a character that starts at a designated 'Start' (S) tile and must navigate to a 'Goal' (G) tile, avoiding 'Hole' (H) tiles. The grid is composed of 'Frozen' (F) tiles, which can be slippery.

**LatENT Variable (Wind):**
*   **`wind_map`**: A 2D grid, the same dimensions as the playing map, where each cell can have a wind direction (North, East, South, West) or no wind.
*   **Sampling**: The `wind_map` is sampled once at the beginning of each episode (i.e., when `env.reset()` is called).
*   **Configuration**: The probability of wind in each direction (and no wind) for 'F' and 'S' cells can be configured via the `wind_probabilities` parameter during environment initialization. This parameter is a tuple of 5 floats (probabilities for No Wind, North, East, South, West, respectively) that must sum to 1.0.

**Causal Interaction & Wind Effect:**
The effect of the wind is critically dependent on the `is_slippery` boolean parameter (default `True`), set during environment initialization:
*   `is_slippery=True`:
    *   Wind is visual only; the wind arrows displayed on the grid do *not* influence the agent's movement mechanics.
    *   Actions are stochastic: if the agent chooses an action, it might "slip" and move to an adjacent tile different from the intended one. This is the standard behavior of `FrozenLake-v1`.
*   `is_slippery=False`:
    *   Wind and the agent's chosen action combine sequentially to determine movement.
        1.  **Wind Action First**: If there is wind in the agent's current cell, an action corresponding to the wind's direction is determined and applied to the agent's current state. This results in an *intermediate state*. If the wind pushes the agent into a wall, the intermediate state is the same as the current state. If the wind blows the agent into a hole, this step is terminal.
        2.  **Agent's Chosen Action Second**: If the wind action was not terminal (e.g., did not blow into a hole), the agent's originally chosen `action` (passed to `env.do()`) is then applied, starting from the *intermediate state* reached after the wind action. The outcome of this second action determines the final state and reward.
    *   If there is no wind in the cell, the agent's chosen action is executed deterministically from the current state.

**Core PCH Methods for `FrozenLakePCH`:**
*   `scm_intended_action, current_obs, reward, terminated, truncated, info = env.see()`:
    *   `scm_intended_action` (int): The action (0:Left, 1:Down, 2:Right, 3:Up) the SCM's internal default policy would take based on `current_obs` and the wind at the agent's current position.
    *   `current_obs` (int): The agent's current state (tile index).
    *   **Important Behavior**: This `see()` method for `FrozenLakePCH` **does not** advance the environment state or actually execute `scm_intended_action`. The returned `reward` is typically 0.0, and `terminated`/`truncated` are `False`.
    *   `info` (dict): Contains details like `{'wind_action_component': WA, 'intermediate_state_after_wind': S_int, 'agent_action_component': AA}` if a two-part action occurred, or `{'prob': p}` indicating transition probability for slippery dynamics.
*   `next_obs, reward, terminated, truncated, info = env.do(action_to_take)`:
    *   Executes the specified `action_to_take` in the environment.
    *   This updates the agent's state according to the environment dynamics (including wind effects if `is_slippery=False`, and slippery physics if `is_slippery=True`).
    *   `reward` (float): The reward calculation is as follows:
        *   `+1.0` if the agent reaches the 'Goal' (G) tile.
        *   `-1.0` if the agent falls into a 'Hole' (H) tile.
        *   For 'Frozen' (F) or 'Start' (S) tiles: A distance-based shaped reward is given, calculated as `(max_manhattan_distance_to_goal - current_manhattan_distance_to_goal) / max_manhattan_distance_to_goal`. This value is between 0 and 1 (exclusive of 1 unless at the goal, which is handled separately), encouraging movement towards the goal.
    *   `terminated` (bool): `True` if the agent reaches the Goal or falls into a Hole, `False` otherwise.
    *   `truncated` (bool): `True` if the episode ends due to other reasons (e.g., step limit, not typically used in standard FrozenLake), `False` otherwise.
    *   `info` (dict): Contains details like `{'wind_action_component': WA, 'intermediate_state_after_wind': S_int, 'agent_action_component': AA}` if a two-part action occurred, or `{'prob': p}` indicating transition probability for slippery dynamics.

**Usage Example (inspired by `test/test_frozenlake.ipynb`):**
```python
import gymnasium as gym
import causal_gym # Registers CausalGym environments

# Initialize the PCH environment
env = gym.make(
    "causal_gym/FrozenLakePCH-v0",
    is_slippery=False,  # Set to False for wind to have a direct causal effect
    wind_probabilities=(0.1, 0.2, 0.2, 0.2, 0.3), # (P_None, P_N, P_E, P_S, P_W)
    render_mode="rgb_array"  # or "human" for an external window
)

# Reset for a new episode
# Use seed=None for random initialization, or an integer for reproducible episodes
initial_obs, info = env.reset(seed=None) 
print(f"Initial observation: {initial_obs}")
print(f"Info from reset: {info}") # Includes wind generation status

# Render the environment (returns an RGB numpy array if render_mode="rgb_array")
frame = env.render()
# (Code to display frame, e.g., with matplotlib, would go here)

# Example interaction loop
for step_num in range(10):
    # 1. Use see() to get the SCM's intended action and current state details
    #    Remember: see() for FrozenLakePCH does NOT take a step.
    scm_action, current_state, _, _, _, see_info = env.see()
    wind_at_agent = see_info.get('wind_in_cell')
    print(f"Step {step_num}: State={current_state}, SCM intends={scm_action}, Wind here={wind_at_agent}")

    # Decide on an action (e.g., use SCM's action, or your own policy)
    action_to_execute = scm_action 
    # action_to_execute = env.action_space.sample() # Alternative: random action

    # 2. Use do() to execute the chosen action and advance the environment
    next_state, reward, terminated, truncated, do_info = env.do(action_to_execute)
    print(f"  Executed: {action_to_execute} -> New State: {next_state}, Reward: {reward}, Done: {terminated or truncated}")
    if do_info.get('wind_action_component') is not None:
        print(f"  Wind component: {do_info['wind_action_component']}, Intermediate state: {do_info['intermediate_state_after_wind']}")
        print(f"  Agent component (from intermediate): {do_info['agent_action_component']}")
    
    current_frame = env.render()
    # (Display current_frame)

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()
```

---

## 2. CartPoleWindPCH Environment (`causal_gym/CartPoleWindPCH-v0`)

**Description:**
This environment is based on the classic CartPole problem. The agent's goal is to balance a pole upright on a cart by applying horizontal forces (left or right) to the cart. The episode ends if the pole tilts too far or the cart moves out of bounds.

**Latent Variable (Wind):**
*   **`current_wind`**: A scalar floating-point value representing a horizontal wind force.
*   **Sampling**: This wind force is sampled *at every time step* from a normal distribution, defined by `wind_mean` and `wind_std` parameters given during environment initialization.
*   **Timing**: The wind value sampled at the end of step `t` (after an action is taken) becomes the `current_wind` that will be applied to influence the cart's velocity at the *beginning* of step `t+1`.

**Causal Interaction & Wind Effect:**
*   The `current_wind` value is directly added to the cart's horizontal velocity (`state[1]`) *before* the standard physics of that CartPole step are processed. This provides a continuous, per-step exogenous influence on the cart's dynamics.

**Core PCH Methods for `CartPoleWindPCH`:**
*   `scm_action, next_obs, reward, terminated, truncated, info = env.see()`:
    *   `scm_action` (int): The action (0 for left, 1 for right) chosen by the SCM's internal default policy.
    *   **Important Behavior**: This `see()` method for `CartPoleWindPCH` **executes** `scm_action` in the environment and advances its state.
    *   `next_obs`, `reward`, `terminated`, `truncated`, `info` are the direct results of taking `scm_action` in the SCM.
*   `next_obs, reward, terminated, truncated, info = env.do(action_to_take)`:
    *   Executes the specified `action_to_take` in the environment.
    *   Updates the cart-pole state according to the wind and action.

**Usage Example:**
```python
import gymnasium as gym
import causal_gym # Registers CausalGym environments
import numpy as np

# Initialize the PCH environment
env = gym.make(
    "causal_gym/CartPoleWindPCH-v0",
    wind_mean=0.0,       # Mean of the wind force distribution
    wind_std=0.05,       # Standard deviation of the wind force
    render_mode="rgb_array" # or "human"
)

# Reset for a new episode
initial_obs, info = env.reset(seed=42)
# The SCM is accessible via env.env for PCH wrappers
print(f"Initial observation: {np.round(initial_obs, 2)}")
print(f"Initial wind (for first step): {env.env.current_wind:.3f}")

current_frame = env.render()
# (Display current_frame)

# Option 1: Run with SCM's internal policy using env.see()
print("\nEpisode 1: Running with SCM's internal policy (env.see())")
for step_num in range(5): # Max 5 steps for brevity
    # env.see() gets SCM action AND executes it
    wind_this_step = env.env.current_wind # Wind that will be applied in this .see() call
    scm_action, next_obs, reward, terminated, truncated, step_info = env.see()
    # env.env.current_wind now holds wind for *next* step
    
    print(f"Step {step_num}: Wind applied={wind_this_step:.3f}, SCM Action={scm_action} -> Next State={np.round(next_obs,2)}, Reward={reward:.1f}")
    
    current_frame = env.render()
    # (Display current_frame)

    if terminated or truncated:
        print("Episode finished.")
        break

# Reset for a new episode to demonstrate env.do()
initial_obs, info = env.reset(seed=123) 
print(f"\nEpisode 2: Initial observation: {np.round(initial_obs, 2)}")
print(f"Initial wind (for first step): {env.env.current_wind:.3f}")

# Option 2: Intervene with custom actions using env.do()
print("Running with custom actions (env.do())")
for step_num in range(5):
    # Decide on an action
    my_action = env.action_space.sample() # Example: random action
    
    wind_this_step = env.env.current_wind # Wind that will be applied in this .do() call
    # env.do() executes your chosen action
    next_obs, reward, terminated, truncated, step_info = env.do(my_action)
    # env.env.current_wind now holds wind for *next* step

    print(f"Step {step_num}: Wind applied={wind_this_step:.3f}, Custom Action={my_action} -> Next State={np.round(next_obs,2)}, Reward={reward:.1f}")

    current_frame = env.render()
    # (Display current_frame)

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()
```
