# Custom Causal Environments for CausalGym

This directory contains custom reinforcement learning environments designed for use with the CausalGym framework. These environments often include specific causal mechanisms or observational challenges. A key example is the `FrozenLakeSCM` detailed below.

## `FrozenLakeSCM` (Custom Windy FrozenLake)

This document describes the `FrozenLakeSCM` environment, a custom FrozenLake for the CausalGym framework. It extends the standard `gymnasium.FrozenLake-v1` with per-cell wind dynamics, configurable reward shaping, and enhanced procedural rendering.

## Overview

The `WindyFrozenLake` (implemented as `FrozenLakeSCM` and wrapped by `FrozenLakePCH`) introduces a stochastic wind factor (a latent confounder) that can influence agent movement. Each traversable tile (Start or Frozen) can have its own wind direction, sampled at the beginning of each episode.

## Key Features

*   **Per-Cell Wind**: Wind direction (a latent confounder) is sampled for each 'F' (Frozen) and 'S' (Start) tile independently at the start of every episode. The probabilities for these wind directions are configurable.
*   **Wind Effect on Agent Movement**:
    *   **Non-Slippery Mode (`is_slippery=False`)**: If wind is present in the agent's current cell, it deterministically dictates the agent's movement direction, overriding the agent's chosen action.
    *   **Slippery Mode (`is_slippery=True`)**: In the current implementation, the wind in the agent's cell **does not affect the movement dynamics**. The agent's movement is determined solely by the standard slippery probabilities of the underlying FrozenLake environment based on the agent's chosen `action` (i.e., 1/3 chance for the intended direction, 1/3 for one perpendicular, 1/3 for the other perpendicular). The `wind_in_cell` is reported in the `info` dictionary and acts as an observational variable but does not influence the transition probabilities.
*   **Procedural `rgb_array` Rendering**: Provides a clear visual representation suitable for notebooks and generating image arrays.
    *   **Agent**: Red circle.
    *   **Goal ('G')**: Green diamond.
    *   **Start ('S')**: Yellow square.
    *   **Frozen ('F')**: Light gray square.
    *   **Hole ('H')**: Black square.
    *   **Wind Indicators**: Displayed on 'S' and 'F' tiles.
        *   Blue arrows indicate North, East, South, or West wind.
        *   Small blue circles indicate no wind in that cell.
*   **Configurable Wind**: The probability distribution for wind directions (None, N, E, S, W) can be specified during environment initialization.
*   **Configurable Reward Shaping**:
    *   `+1.0` for reaching the Goal ('G').
    *   `-1.0` for falling into a Hole ('H').
    *   For all other states, a normalized Manhattan distance-based reward is given by default: `(max_manhattan_dist - current_manhattan_dist_to_goal) / max_manhattan_dist`. This provides a positive incentive for moving closer to the goal. The `max_manhattan_dist` is calculated based on the map size. This behavior can be modified within the environment's code.
*   **Standard CausalGym Integration**: Implemented as an `SCM` (`FrozenLakeSCM`) and typically used via its `PCH` wrapper (`FrozenLakePCH`).

## Initialization Parameters

The environment is initialized via `FrozenLakePCH` (or directly via `FrozenLakeSCM`) with the following key parameters:

*   `map_name` (str): Specifies the grid layout (e.g., `"4x4"`, `"8x8"`). Defaults to `"4x4"`.
*   `is_slippery` (bool): If `True`, the agent may not always move in the intended direction (standard FrozenLake slipperiness). If `False`, movement is deterministic based on action (or wind, if present). Defaults to `True`.
*   `wind_probabilities` (tuple[float, float, float, float, float]): A tuple of 5 floats that sum to 1.0, representing the probabilities for `(WIND_NONE, WIND_NORTH, WIND_EAST, WIND_SOUTH, WIND_WEST)` respectively. Defaults to `(0.7, 0.075, 0.075, 0.075, 0.075)`.
*   `render_mode` (str | None):
    *   `"rgb_array"`: Renders the environment to a NumPy array (suitable for notebooks).
    *   `"human"`: Renders in a pop-up Pygame window.
    *   `None`: No rendering.
*   `seed` (int): Seed for the random number generator for reproducibility. Defaults to `0`.

Example:
```python
from causal_gym.causal_gym.envs import FrozenLakePCH # Adjust import path as needed

env = FrozenLakePCH(
    map_name="4x4",
    is_slippery=False,
    wind_probabilities=(0.1, 0.2, 0.2, 0.2, 0.3), # Example: Higher chance of wind
    render_mode="rgb_array",
    seed=42
)
obs, info = env.reset()
# ... interact with env ...
```

## State and Actions

*   **Observation (State)**: An integer representing the agent's current tile index (0 to N-1, where N is the total number of tiles).
*   **Action Space**: Discrete(4)
    *   `0`: Move Left
    *   `1`: Move Down
    *   `2`: Move Right
    *   `3`: Move Up

## Information Dictionary (`info`)

The `info` dictionary returned by `reset()` and `step()` contains useful diagnostic information:

*   **From `reset()`**:
    *   `wind_map_generated` (bool): Indicates that a new `wind_map` was sampled.
    *   `initial_agent_cell_wind` (int): The wind direction in the agent's starting cell.
    *   `wind_map` (numpy.ndarray): The actual per-cell wind map for the episode (accessible via `env.env.wind_map` if `env` is the PCH wrapper).
*   **From `step()`**:
    *   `prob` (float): The probability of the sampled transition occurring.
    *   `action_was` (int): The original action supplied by the policy/agent.
    *   `wind_overrode_action_to` (int, optional): If in non-slippery mode and wind was present, this shows the action dictated by the wind.
    *   `wind_in_cell` (int): The wind direction in the cell the agent was in *before* taking the step.
    *   `agent_pos_rc` (tuple[int, int]): The agent's new position as (row, column) after the step.

## Dependencies
*   `gymnasium`
*   `numpy`
*   `pygame` (for rendering)