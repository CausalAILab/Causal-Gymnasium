# Reinforcement Learning Experiments in CausalGym

This directory contains Python scripts for running reinforcement learning experiments using custom environments within the CausalGym framework.

## `test_frozenlake_ucbvi.py`

### Purpose

This script is designed to test and evaluate the `UCBVI` (Upper Confidence Bound Value Iteration) algorithm on a customized version of the `FrozenLake` environment, referred to as `FrozenLakeSCM` (or `FrozenLakePCH` in some earlier versions/discussions). The environment and agent are highly configurable to allow for experimentation with different causal factors and learning parameters.

### Custom Environment Features (`FrozenLakeSCM`)

The `FrozenLakeSCM` environment extends the classic `FrozenLake` with several key modifications:

1.  **Per-Cell Wind:**
    *   Wind can be configured for each cell of the grid.
    *   `wind_probabilities` in the environment's `__init__` method control the likelihood of `WIND_NONE, WIND_NORTH, WIND_EAST, WIND_SOUTH, WIND_WEST` for each cell, sampled at the start of each episode.
    *   Wind affects the agent's movement:
        *   In non-slippery mode, if wind is present in the agent's current cell, it dictates the agent's movement. Otherwise, the agent's chosen action is performed.
        *   In slippery mode, wind can alter the outcome of an action, adding another layer of stochasticity.
    *   The rendering (`rgb_array` mode) displays wind indicators (arrows or a circle for no wind) on each cell.

2.  **Reward Shaping:**
    *   To facilitate learning in a sparse reward environment, a shaped reward function is implemented:
        *   `+1.0` for reaching the Goal ('G').
        *   `-1.0` for falling into a Hole ('H').
        *   For all other states, a normalized Manhattan distance-based reward is given: `(max_manhattan_dist - current_manhattan_dist_to_goal) / max_manhattan_dist`. This provides a positive incentive for moving closer to the goal.

3.  **Configurable Slipperiness:**
    *   The environment can be initialized with `is_slippery=True` (standard FrozenLake behavior where actions have a 1/3 chance of going in the intended direction and 1/3 chance for each perpendicular direction) or `is_slippery=False` (actions are deterministic unless overridden by wind).

### How to Run

1.  **Prerequisites:**
    *   Ensure you have Python installed with the necessary libraries (e.g., `causalgym`, `numpy`, `matplotlib`, `pygame`). These should be managed by your project's environment (e.g., `venv`).
    *   The script assumes it is run from the root of the `causal2` project directory or that the `PYTHONPATH` is set up correctly to find the `causalgym` modules. The script includes `sys.path` manipulations to aid in this.

2.  **Execution:**
    *   Navigate to the `causal2` directory in your terminal.
    *   Run the script using:
        ```bash
        python causalgym/test/learning/test_frozenlake_ucbvi.py
        ```
    *   The script includes a `CAPTURE_REPLAY_FLAG` (defaults to `True`). If set, and a successful episode occurs, you will be prompted to watch a replay.

### Expected Output

*   **Console Output:**
    *   The script will print diagnostic information during training, including:
        *   Episode numbers.
        *   Cumulative reward per episode.
        *   Average reward per step for the episode.
        *   Notifications when the agent reaches the goal or falls into a hole.
        *   Detailed updates from the `UCBVI` agent's `update` and `plan` methods if `print_diagnostics` is enabled within the script.
    *   A summary at the end, showing the total number of successful episodes and final average rewards.

*   **Plot:**
    *   A PNG image file (e.g., `ucbvi_on_slippery_windy_frozenlake_4x4.png`) will be saved in a `plots/` subdirectory within the `causal2` root directory.
    *   This plot displays:
        *   Cumulative reward per episode and its moving average (primary y-axis).
        *   Average reward per step per episode and its moving average (secondary y-axis).

*   **Episode Replay (Conditional):**
    *   If `CAPTURE_REPLAY_FLAG` in the script is `True` and the agent successfully completes at least one episode:
        *   The script will save the visual frames from the *last* successful episode.
        *   After the plot is generated, you will be prompted in the console: `Watch replay of the last successful episode? (y/n):`.
        *   If you respond with 'y', a new window will open displaying an animation of the agent navigating the environment during that successful run.

### Key Hyperparameters and Configuration (`test_frozenlake_ucbvi.py`)

The script defines several important parameters that can be adjusted for experimentation:

*   `NUM_EPISODES`: Total number of training episodes.
*   `EPISODE_HORIZON`: Maximum number of steps per episode.
*   `PLANNING_SWEEPS`: Number of value iteration sweeps performed by the `UCBVI` agent at the beginning of each episode.
*   `DELTA`: Confidence parameter for the UCB exploration bonus in the `UCBVI` agent. Lower values encourage more exploration.
*   `AGENT_EPSILON`: Probability for epsilon-greedy action selection in the `UCBVI` agent's `act` method.
*   `CAPTURE_REPLAY_FLAG`: Boolean (default `True`) to enable or disable the capture and replay functionality for successful episodes.
*   `ENV_CONFIG`: A dictionary to pass parameters to the `FrozenLakeSCM` environment, including:
    *   `map_name`: e.g., "4x4" or "8x8".
    *   `is_slippery`: `True` or `False`.
    *   `wind_probabilities`: A tuple defining the probability distribution for different wind conditions, e.g., `(0.5, 0.125, 0.125, 0.125, 0.125)` for (None, N, E, S, W).
    *   `seed`: For reproducibility of environment generation and agent initialization.

### UCBVI Algorithm Notes (`causalgym/causal_gym/algorithms/ucbvi.py`)

The `UCBVI` agent used in this script has several key features:

*   **Initialization:** Takes the number of states, actions, horizon, delta, epsilon, and a seed.
*   **`update`:** Updates transition counts (`N`), estimated rewards (`R`), and estimated transition probabilities (`P`).
*   **`plan`:** Performs value iteration.
    *   The Bellman update uses the max V-value of the *next state*.
    *   Terminal states (Hole 'H' or Goal 'G') have their V-values set to 0.
    *   V-values are clipped to `[-H, H]` to handle shaped rewards.
*   **`act`:** Implements epsilon-greedy exploration for the applied action.
*   **`reset_model`:** Resets `N`, `P`, `R`, `Q`, `V` (typically called at the start of a new episode if the model is not persistent across episodes in the test script).

This setup allows for robust testing of the `UCBVI` algorithm under varying conditions of stochasticity (slipperiness, wind) and reward structures.

## Note on Algorithms for Continuous State Spaces (e.g., LunarLander)

While the project has expressed interest in implementing UCBVI and Q-learning algorithms, it's important to note that the tabular UCBVI version detailed here is primarily suited for environments with discrete state spaces (like FrozenLake).

For environments with continuous state spaces (e.g., LunarLander), more advanced algorithms are typically required for effective learning. These may include:
*   **Discretization of the state space:** To adapt tabular methods, though this can be challenging and may lose information, especially in high-dimensional spaces.
*   **Deep Q-Networks (DQN):** Suitable for discrete actions and continuous states, using neural networks for function approximation.
*   **Policy Gradient Methods (e.g., REINFORCE, A2C, PPO, SAC):** Designed for continuous states and can also handle continuous actions.
*   **Continuous UCBVI variants:** More complex research algorithms that extend UCB principles to continuous domains, often involving sophisticated function approximation and uncertainty estimation techniques.

Future work on environments like LunarLander will likely involve exploring these more suitable algorithms, even if initial explorations attempt discretization with tabular methods like UCBVI. 