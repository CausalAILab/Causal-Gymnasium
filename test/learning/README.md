# Reinforcement Learning Experiments in CausalGym

This directory contains Python scripts for running reinforcement learning experiments using custom environments within the CausalGym framework.

## `train_lander_ucbvi.py`

### Purpose
This script trains and evaluates `UCBVI` (Upper Confidence Bound Value Iteration) and `UCBQ` (UCB Q-Learning) algorithms on the `LunarLanderPCH` environment. This environment is a version of the classic LunarLander control problem, adapted to include potential latent confounders (e.g., wind) managed through a PCH (Probabilistic Causal Hierarchy) wrapper. The script focuses on discretizing the continuous state space of LunarLander to make it compatible with these tabular RL methods.

### Custom Environment Features (`LunarLanderPCH`)
*   **PCH Wrapper:** Built on `LunarLanderSCM`, which itself wraps `gymnasium.make("LunarLander-v3")`. The PCH structure allows for interventions and observation of underlying causal variables, though this script primarily uses the standard `step` method.
*   **Latent Confounders:** The `LunarLanderPCH` is designed to potentially include latent variables like wind, which can affect the lander's dynamics. The specific SCM implementation dictates the nature of these confounders.
*   **Continuous State/Discrete Action:** The base LunarLander environment has an 8-dimensional continuous state space and a discrete action space (do nothing, fire left engine, fire main engine, fire right engine).

### Discretization
A `Discretiser` class within the script handles the conversion of the 8D continuous observations from LunarLander into a single integer state, which is required by the tabular `UCBVI` and `UCBQ` agents.
*   It uses `numpy.digitize` based on pre-defined bins for each dimension.
*   The number of bins per dimension is configurable and significantly impacts the total number of discrete states (e.g., `[3,3,2,2,3,2,2,2]` results in 864 states for UCBVI, while `[6,6,4,4,6,4,2,2]` results in 55,296 for UCBQ).
*   Observation space bounds are taken from a temporary `LunarLander-v3` instance.

### How to Run

1.  **Prerequisites:**
    *   Ensure Python, CausalGym, Gymnasium, NumPy, and Matplotlib are installed.
    *   The script uses `sys.path` manipulations to find `LunarLanderPCH` (from `causal_gym.envs.lunar_lander`) and the algorithm implementations (`ucbvi.py`, `ucbq.py` from `causalgym.causal_gym.algorithms`). Ensure these paths are correct relative to your project structure if you modify the script's location.

2.  **Execution:**
    *   Navigate to the project root (e.g., `causal2/`) in your terminal.
    *   Run the script using:
        ```bash
        python causalgym/test/learning/train_lander_ucbvi.py --episodes <N_EPISODES> --algo <ucbvi|ucbq> [OPTIONS]
        ```
    *   Example:
        ```bash
        python causalgym/test/learning/train_lander_ucbvi.py --episodes 1000 --horizon 300 --algo ucbq --plot --gif --verbose
        ```

### Command-Line Arguments
*   `--episodes` (int, default: 500): Number of training episodes.
*   `--horizon` (int, default: 1000): Maximum steps per episode. Also planning horizon for UCBVI.
*   `--delta` (float, default: 0.1): Confidence parameter for UCB algorithms.
*   `--algo` (str, default: "ucbvi", choices: ["ucbvi", "ucbq"]): Algorithm to use.
*   `--seed` (int, default: 0): Random seed.
*   `--gif` (flag): Save a GIF of a trained policy rollout to `lander_rollout.gif`.
*   `--plot` (flag): Save a PNG plot of episode returns to `lander_<algo>_returns.png`.
*   `--verbose` / `-v` (flag): Enable detailed printouts for the first few steps/episodes.
*   `--show-visuals` (flag): Show plots/GIF frames interactively (currently saves them by default).

### Expected Output
*   **Console Output:**
    *   Discretiser self-test status.
    *   Training progress, including episode number, return, and steps per episode (printed periodically).
    *   Verbose output for initial episodes/steps if `--verbose` is used (initial observation, discrete state, actions, rewards, next states).
    *   Notification of "SUCCESSFUL LANDING" if `reward >= 100` upon termination.
    *   Final training time and average returns for the first and last 10% of episodes to indicate learning.
*   **Plot (if `--plot`):**
    *   A PNG file `lander_<algo>_returns.png` (saved in the directory where the script is run) showing cumulative episode returns vs. episode number.
*   **GIF (if `--gif`):**
    *   A GIF file `lander_rollout.gif` (saved in the CWD) showing a rollout of the trained agent.
    *   A PNG file `lander_sample_frames.png` with a few sample frames from the GIF.

### Algorithm Notes
*   **UCBVI:** Uses `num_states`, `n_actions`, `horizon`, `delta`. The script uses a two-step action selection: 1. `x_intended = argmax(V[s,:])`, 2. `a_to_apply = agent.act(s, x_intended)`. Update is `agent.update(s, x_intended, a_to_apply, r, s2)`.
*   **UCBQ:** Uses `n_states`, `n_actions`, `delta`. Action selection is `agent.act(s)`. Update is `agent.update(s, a_to_apply, r, s2)`.

---

## `train_cartpole_ucbvi.py`

### Purpose
This script trains and evaluates `UCBVI` and `UCBQ` algorithms on the `CartPoleWindPCH` environment. This custom environment introduces a latent wind variable to the classic CartPole problem. Similar to the LunarLander trainer, this script discretizes CartPole's continuous state space.

### Custom Environment Features (`CartPoleWindPCH`)
*   **PCH Wrapper:** Built on `CartPoleWindSCM`, which wraps `gymnasium.make("CartPole-v1")`.
*   **Latent Wind:** A horizontal wind force is sampled at the start of each episode and applied as a small additive acceleration to the cart's velocity at each step. The mean and standard deviation of this wind are configurable.
*   **Initial Pole Angle:** The SCM allows setting a mean and standard deviation for the initial pole angle at reset.
*   **Continuous State/Discrete Action:** The base CartPole environment has a 4-dimensional continuous state space and 2 discrete actions (push left, push right).

### Discretization
A `Discretiser` class within the script converts the 4D continuous observations from CartPole into a single integer state.
*   Uses fixed symmetric ranges for each dimension (e.g., position `[-2.4, 2.4]`, angle `[-0.20, 0.20] rad`).
*   The number of bins per dimension (`--bins` argument) is uniform across all dimensions. For example, `--bins 8` results in `8^4 = 4096` states.

### How to Run

1.  **Prerequisites:**
    *   Ensure Python, CausalGym, Gymnasium, NumPy, and Matplotlib are installed.
    *   The script uses `sys.path` manipulations to find `cartpole_wind_pch.py` (assumed in the project root) and the algorithm implementations (`ucbvi.py`, `ucbq.py` from `causalgym.causal_gym.algorithms`).

2.  **Execution:**
    *   Navigate to the project root (e.g., `causal2/`) in your terminal.
    *   Run the script using:
        ```bash
        python causalgym/test/learning/train_cartpole_ucbvi.py --episodes <N> --algo <ucbvi|ucbq> --bins <B> [OPTIONS]
        ```
    *   Example:
        ```bash
        python causalgym/test/learning/train_cartpole_ucbvi.py --episodes 2000 --algo ucbvi --bins 8 --horizon 200 --plot --gif
        ```

### Command-Line Arguments
*   `--episodes` (int, default: 1000): Number of training episodes.
*   `--horizon` (int, default: 200): Maximum steps per episode (planning horizon for UCBVI).
*   `--algo` (str, default: "ucbvi", choices: ["ucbvi", "ucbq"]): Algorithm to use.
*   `--bins` (int, default: 8): Number of bins *per dimension* for the discretizer.
*   `--delta` (float, default: 0.1): Confidence parameter for UCB algorithms.
*   `--epsilon` (float, default: 0.1): Epsilon for epsilon-greedy exploration (used by UCBVI).
*   `--seed` (int, default: 0): Random seed.
*   `--gif` (flag): Save a GIF of a trained policy rollout to `cartpole.gif` in the CWD.
*   `--plot` (flag): Save a PNG plot of cumulative regret to `cartpole_regret.png` in the CWD.

### Expected Output
*   **Console Output:**
    *   Number of discretized states.
    *   Training progress (episode number, return, steps, regret for the episode).
    *   Final training time.
*   **Plot (if `--plot`):**
    *   A PNG file `cartpole_regret.png` showing cumulative regret (sum of `1 - reward_step`) vs. episode number.
*   **GIF (if `--gif`):**
    *   A GIF file `cartpole.gif` showing a rollout of the trained agent.

### Algorithm Notes
*   **UCBVI:** Takes `epsilon` in its constructor. The `plan()` method is called at the start of each episode. The `update` method uses `realised_action` from `info` if available.
*   **UCBQ:** Does not take `epsilon`. The `update` method also uses `realised_action`.
*   **Regret:** The script calculates and plots cumulative regret, where per-step regret is `1.0 - reward_step`. For CartPole, rewards are typically +1 for every step the pole is balanced.

---

## `cartpole_wind_demo.ipynb`

### Purpose
This Jupyter Notebook provides a demonstration of training the `UCBVI` algorithm on the `CartPoleWindPCH` environment. It serves as a more interactive way to explore the concepts in `train_cartpole_ucbvi.py`, showcasing environment setup, agent instantiation (using the `Discretiser` from `train_cartpole_ucbvi.py`), a basic training loop, and plotting of results (episode returns and cumulative regret).

### Key Features
*   **Interactive Environment:** Allows step-by-step execution and inspection of variables.
*   **Uses `Discretiser`:** Imports and uses the `Discretiser` class from `train_cartpole_ucbvi.py` (located in the same directory) to handle CartPole's continuous state space.
*   **UCBVI Agent:** Demonstrates the instantiation and use of the `UCBVI` agent.
*   **Training Loop:** Includes a simplified training loop to run a set number of episodes.
*   **Visualization:** Plots average episode returns (with a moving average) and cumulative regret.

### How to Run
1.  **Prerequisites:**
    *   Ensure Jupyter Notebook or JupyterLab is installed.
    *   Required Python libraries: `numpy`, `matplotlib`, `gymnasium`, `tqdm` (optional, for progress bars).
    *   The notebook relies on:
        *   `cartpole_wind_pch.py` (expected in the project root: `causal2/`).
        *   `ucbvi.py` (expected in `causal2/causalgym/causal_gym/algorithms/`).
        *   `train_cartpole_ucbvi.py` (for the `Discretiser` class, expected in the same `causalgym/test/learning/` directory as the notebook).
    *   The notebook's first cell contains `sys.path` manipulations to help locate these dependencies.

2.  **Execution:**
    *   Navigate to the `causal2/causalgym/test/learning/` directory.
    *   Launch Jupyter Notebook/Lab (e.g., `jupyter notebook` or `jupyter lab`).
    *   Open `cartpole_wind_demo.ipynb`.
    *   Run the cells sequentially.

### Expected Output
*   **Cell 1 (Setup):** Prints messages indicating successful path setup and imports.
*   **Cell 2 (Instantiation):** Prints confirmation of discretizer and agent creation, including the number of discrete states.
*   **Cell 3 (Training):** Prints progress (if `tqdm` is available) and sample returns/regrets after training.
*   **Cell 4 (Plotting):** Displays two plots:
    *   Episode returns (raw and moving average) vs. episode number.
    *   Cumulative regret vs. episode number.

### Configuration (within the notebook)
*   `N_EPISODES`: Number of episodes to train (e.g., 200 for a quick demo).
*   `HORIZON`: Max steps per episode (e.g., 200).
*   `BINS_PER_DIM`: Number of bins for the `Discretiser` (e.g., 8).
*   `DELTA`, `EPSILON`, `SEED`: Parameters for the UCBVI agent and environment.

This notebook is useful for quick tests, parameter tuning visualization, and understanding the interaction between the `CartPoleWindPCH` environment, the `Discretiser`, and the `UCBVI` agent.

---

## `test_frozenlake_ucbvi.py`

### Purpose

This script is designed to test and evaluate the `UCBVI` (Upper Confidence Bound Value Iteration) algorithm on a customized version of the `FrozenLake` environment, referred to as `FrozenLakeSCM`. The environment and agent are highly configurable to allow for experimentation with different causal factors and learning parameters.

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
    *   A PNG image file (e.g., `ucbvi_on_slippery_windy_frozenlake_4x4.png`) will be saved in a `plots/` subdirectory within the `causal2` root directory (Note: this path might need adjustment if the script is run from `causalgym/test/learning/`).
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

## Note on Algorithms for Continuous State Spaces

While the project has expressed interest in implementing UCBVI and Q-learning algorithms, it's important to note that the tabular UCBVI version detailed here is primarily suited for environments with discrete state spaces (like FrozenLake).

For environments with continuous state spaces (e.g., LunarLander, CartPole), more advanced algorithms are typically required for effective learning. These may include:
*   **Discretization of the state space:** As demonstrated in `train_lander_ucbvi.py` and `train_cartpole_ucbvi.py`, this adapts tabular methods but can be challenging for high-dimensional spaces and may lose information. The effectiveness heavily depends on the chosen binning strategy.
*   **Deep Q-Networks (DQN):** Suitable for discrete actions and continuous states, using neural networks for function approximation.
*   **Policy Gradient Methods (e.g., REINFORCE, A2C, PPO, SAC):** Designed for continuous states and can also handle continuous actions.
*   **Continuous UCBVI variants:** More complex research algorithms that extend UCB principles to continuous domains, often involving sophisticated function approximation and uncertainty estimation techniques.

Future work on environments like LunarLander and CartPole will likely involve exploring these more suitable algorithms, even if initial explorations attempt discretization with tabular methods. 