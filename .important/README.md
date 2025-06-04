# README: Algorithm Tuning Insights for UCBVI/UCBQ

This document summarizes key insights gained from tuning the UCBVI (Upper Confidence Bound Value Iteration) and UCBQ-like algorithms, particularly focusing on parameters critical for effective learning. These insights were primarily derived from experiments with the FrozenLake (Non-Slippery) environment.

## 1. The Critical Role of the `horizon` Parameter

Initially, a single `horizon` parameter in the UCBVI/UCBQ implementations was used for multiple purposes:
*   As the number of planning sweeps in value iteration (an integer).
*   As the value for clipping optimistic Q-values (e.g., `Q = min(horizon, ...)`).
*   As a scaling factor for the exploration bonus term (e.g., `bonus = c_bonus * horizon * sqrt(...)`).

This dual (or triple) use led to significant issues:
*   If `horizon` was set high (e.g., 100, suitable for planning sweeps or a theoretical max episode length), it caused the Q-values to be clipped at a very high value and the bonus term to become excessively large. This resulted in over-optimism, where the agent could not effectively distinguish between good and bad actions until an extremely large number of visits, hindering learning.
*   If `horizon` was set low (e.g., 1.0 or 2.0, suitable for a realistic maximum episodic reward for clipping/bonus scaling), it was then unsuitable for `range(horizon)` in the planning loop, causing TypeErrors or insufficient planning sweeps.

## 2. Decoupling `planning_sweeps` from `max_episode_reward`

The key solution was to decouple these concepts:

*   **`planning_sweeps` (formerly `horizon` in some contexts):**
    *   **Purpose:** Defines the number of iterations for the value iteration (Bellman backup) process within the `plan()` method of model-based algorithms like UCBVI.
    *   **Type:** Integer.
    *   **Typical Value:** Can be set based on the estimated diameter of the state space or the desired depth of planning (e.g., 100 for a 4x4 FrozenLake).

*   **`max_episode_reward`:**
    *   **Purpose:** Represents a realistic upper bound on the cumulative reward achievable in a single episode. This value is used for:
        1.  Clipping optimistic Q-values: `Q_value = min(max_episode_reward, ...)`
        2.  Scaling the exploration bonus: `bonus = c_bonus * max_episode_reward * sqrt(...)`
    *   **Type:** Float.
    *   **Typical Value:** Should be set close to the actual maximum reward the agent can get (e.g., 1.0 for FrozenLake if goal reward is +1 and no other positive rewards accumulate significantly, or slightly higher if there are small positive intermediate rewards).

**Implementation Change:**
The `UCBVI` class and the `ucb_q` function were modified. `UCBVI` now takes `horizon` (for planning sweeps) and `max_episode_reward` as separate arguments. `ucb_q` also takes these distinct parameters and uses them appropriately.

## 3. The `c_bonus` (Exploration Constant)

*   **Purpose:** `c_bonus` scales the exploration bonus. A larger `c_bonus` leads to more optimism and exploration.
*   **Interaction with `max_episode_reward`:** The effective exploration pressure is a result of `c_bonus * max_episode_reward`.
*   **Effective Value:** After `max_episode_reward` was set appropriately (e.g., to 1.0 for FrozenLake), a `c_bonus` value of **1.0** was found to work well. This provided a good balance between exploration and exploitation, allowing the agent to learn effectively. Values significantly higher (e.g., the theoretical default of ~7.0) were detrimental when `max_episode_reward` was also high, but might be reconsidered if `max_episode_reward` is very small (though the primary scaling should come from `max_episode_reward` reflecting actual reward scales).

## 4. Successful Configuration for FrozenLake (Non-Slippery, Wind)

The following parameter configuration for `CtfUCBDriver (UCBVI)` proved effective for the custom FrozenLake (4x4, non-slippery, with wind) environment:

*   `planning_sweeps` (passed as `horizon` to `UCBVI`): `100`
*   `max_episode_reward`: `1.0`
*   `c_bonus`: `1.0`
*   `delta`: `0.1` (failure probability for confidence bounds)

With these settings, the agent demonstrated clear learning, achieving significantly more frequent successes compared to initial configurations.

## 5. General Advice for Other Environments (Lunar Lander, Cartpole Wind)

*   **Crucial Tuning:** The `max_episode_reward` and `c_bonus` parameters are highly sensitive and crucial for good performance. They will likely need to be tuned specifically for each environment based on its reward scale and dynamics.
*   **`max_episode_reward` Estimation:**
    *   For **Lunar Lander:** Consider the maximum possible score. A successful landing gives +100, avoiding crashes gives points, fuel costs points. The range can be wide. A value like 100-250 might be a starting point. With parameters `planning_sweeps=100`, `max_episode_reward=250.0`, `c_bonus=1.0`, and an improved discretization (1728 states including position, velocity, angle, and angular velocity), initial runs up to 1000 episodes show that while the agent explores and total episodic rewards vary (occasionally less negative), consistent positive rewards or successful landings have not yet been achieved. This highlights the significant challenge posed by this environment for tabular UCBVI, likely requiring much more extensive training, further refinement of state discretization, or different algorithmic approaches.
    *   For **Cartpole Wind:** Reward is +1 for every step the pole is balanced. If the max episode length is 200, then `max_episode_reward` could be around 200.
*   **`planning_sweeps`:** Should be an integer, sufficiently large for value propagation across the (discretized) state space. For larger or more complex state spaces, this might need to be higher than 100.
*   **Discretization:** For continuous environments like Lunar Lander and Cartpole, the quality of the state discretization scheme will be paramount *before* even tuning these UCBVI parameters. If the discretization is poor, the agent won't be able to learn a good model or Q-values regardless of UCBVI parameters.

By carefully setting `max_episode_reward` to reflect the true scale of rewards and then tuning `c_bonus` (starting around 1.0), exploration can be guided much more effectively. 

## 6. Initial Configuration for CartPoleWind

A new learning visualization script, `.important/test_cartpole_ucbvi_learning_viz.py`, has been created for the `CartPoleWind` environment. The following initial parameters and discretization strategy are used:

*   **`max_episode_reward`**: `200.0` (derived from `max_episode_steps = 200` in `CartPoleWindSCM`, with a reward of +1 per step).
*   **`N_ACTIONS`**: 2 (Push cart left or right).
*   **`planning_sweeps`** (passed as `horizon` to `UCBVI`): `100`.
*   **`c_bonus`**: `1.0`.
*   **`delta`**: `0.1`.
*   **Discretization Strategy (`discretize_cartpole`)**:
    *   Observation space: `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`.
    *   Cart Position (`x`): 5 bins over `[-2.4, 2.4]`.
    *   Cart Velocity (`x_dot`): 5 bins over `[-2.0, 2.0]`.
    *   Pole Angle (`theta`): 7 bins over `[-0.2095, 0.2095]` radians (approx. +/- 12 degrees).
    *   Pole Angular Velocity (`theta_dot`): 5 bins over `[-2.0, 2.0]` radians/sec.
*   **Resulting `NUM_STATES`**: `5 * 5 * 7 * 5 = 875`.
*   **Initial `N_EPISODES` for testing**: `2000`.

The script will output a GIF of the first episode that successfully runs for the full `MAX_EPISODE_STEPS` and a plot of total episodic rewards. These will help assess the learning performance with this initial configuration. 