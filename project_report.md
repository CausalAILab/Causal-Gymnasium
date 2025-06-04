# Project Report: Causal Reinforcement Learning Environments and Algorithms

## 1. Introduction

This report details the work completed on developing and testing several reinforcement learning environments with causal properties, along with the implementation and application of specific learning algorithms (UCBVI and UCBQ) designed to leverage these causal structures. The project aims to explore how latent variables can be incorporated into standard RL environments and how algorithms can perform counterfactual reasoning to achieve better performance or understanding.

The primary components of this work include:
*   **Environments:** Frozen Lake, Lunar Lander, and Cartpole Wind, each modified to include a latent variable (typically "wind").
*   **Algorithms:** UCBVI (Upper Confidence Bound Value Iteration) and UCBQ, designed for learning in these causal settings.
*   **Testing:** A suite of tests demonstrating basic environment interaction and the application of the learning algorithms, showcasing their performance through metrics like regret and success rates.

This document will cover the specifics of each environment, the details of the algorithms, and a summary of the experimental results obtained from the learning demonstrations.

## 2. Environments

This section describes the custom reinforcement learning environments developed, focusing on how latent variables were introduced, the resulting causal structures, and the mechanisms for interaction, including counterfactual interventions.

### 2.1 Frozen Lake (Discrete)
*   **Description:**
    The Frozen Lake environment is a classic grid-world problem. The agent navigates a grid representing a frozen lake, aiming to move from a starting state ('S') to a goal state ('G') while avoiding holes ('H'). The surface can be 'Frozen' ('F'), which is safe to traverse. Actions are discrete: Left (0), Down (1), Right (2), Up (3). The environment can be initialized as "slippery," where the agent's chosen action may not lead to the intended movement due to the icy surface, introducing stochasticity into the transitions.

*   **Latent Variable Implementation (Wind):**
    A latent variable, "wind," is introduced to add a layer of causal influence.
    *   The wind is represented by a `wind_map`, a 2D array corresponding to the lake's grid.
    *   At the beginning of each episode (during `reset()`), the `sample_u()` method populates this `wind_map`. Each cell on the grid (that is not a hole or goal) is assigned a wind direction (None, North, East, South, or West) based on predefined `wind_probabilities` (defaulting to 70% chance of no wind, and 7.5% for each of the four directions).
    *   **Effect of Wind:**
        *   If the environment is configured with `is_slippery=False`, the wind in the agent's current cell directly overrides the agent's `intended_action`. For example, if the wind is blowing North, the agent's action will be changed to "Up," regardless of what the agent chose.
        *   If `is_slippery=True`, the wind's overriding effect is not applied directly in this manner. The environment's inherent slipperiness (stochastic transitions from the base FrozenLake-v1) remains the primary source of deviation from the intended action. However, the wind conditions are still logged.

*   **Causal Diagram/Structure:**
    The introduction of wind, particularly in the non-slippery mode, creates a clear causal structure:
    1.  `Agent_Intended_Action` (chosen by policy or intervention)
    2.  `Wind_In_Agent_Cell` (sampled at episode start)
    3.  If `is_slippery=False`: `Wind_In_Agent_Cell` -> `Executed_Action`
    4.  If `is_slippery=True` or `Wind_In_Agent_Cell` is None: `Agent_Intended_Action` -> (Potentially Modified by Slipperiness) -> `Executed_Action`
    5.  `Executed_Action` & `Current_State` -> `Next_State`, `Reward`, `Terminated`

    The key causal link being explored is how `Wind_In_Agent_Cell` can act as a confounder or direct effector on the `Executed_Action`, influencing the outcome.

*   **Usage and Counterfactual `do()` operations:**
    The environment is typically interacted with via the `FrozenLakePCH` (Probabilistic Causal Hacked) wrapper, which in turn uses the `FrozenLakeSCM` (Structural Causal Model).
    *   `reset()`: Resets the environment to an initial state and samples a new `wind_map` for the episode. Returns `(initial_observation, info_dict)`. The `info_dict` contains details like the generated `wind_map` and the wind in the agent's initial cell.
    *   `see()`: This method, part of the PCH, allows the agent to take an action according to its current policy (e.g., a learned policy or a random exploration policy). It returns `(x_int, obs, r_obs, term, trunc, info)`. The environment evolves based on the sampled wind conditions for that episode.
    *   `step(action)`: The underlying SCM method that executes the given `action`. It considers the `wind_map` and `is_slippery` to determine the actual transition.
    *   `do(action)`: This PCH method allows for an interventional action. It calls `self.scm.step(action)`. This means the specified `action` is executed within the SCM, subject to the *currently existing* wind conditions (sampled at `reset()`) and the `is_slippery` logic. This is used to ask "what if I take `action` now, given the current state of the world (including its wind)?".
    *   **Performing Counterfactuals under Specific Wind Conditions:** The current implementation samples wind at `reset()`. To test an action under a *hypothetical* wind condition not currently active, the `wind_map` within the SCM would need to be manually set to the desired configuration before calling `do(action)`. While no direct `set_wind_map(new_wind_map)` method is exposed on the SCM or PCH, this could be achieved by modifying the SCM's `wind_map` attribute directly if needed for specific counterfactual queries beyond what `do(action)` offers with the episode's sampled wind. The `test/test_frozenlake.py` script demonstrates using `do(action)` to evaluate the probability of reaching the goal when a specific action (e.g., 'Right') is consistently taken.

### 2.2 Lunar Lander (Continuous)
*   **Description:**
    This environment is a wrapper around Gymnasium's `LunarLander-v3`. The objective is to control a lunar lander to land safely and softly on a designated landing pad. The observation space is continuous and 8-dimensional, including the lander's position (x, y), linear velocities (vx, vy), angle, angular velocity, and two booleans indicating whether each leg has contact with the ground. Actions are discrete: 0 (do nothing), 1 (fire left orientation engine), 2 (fire main engine), 3 (fire right orientation engine). A significant positive reward is given for a successful landing, a large negative reward for crashing, and small negative rewards for fuel consumption.

*   **Latent Variable Implementation (Wind):**
    A latent "wind" variable introduces a continuous horizontal force affecting the lander.
    *   **Wind Map:** At the start of each episode (during `reset()`), the `sample_u()` method generates a `wind_map`. This map is a 15x20 grid conceptually overlaid on the game world. Each cell in this grid is assigned a wind strength value sampled from a Normal distribution (configurable `wind_mean` and `wind_std`). Positive values represent wind pushing the lander to the left, and negative values push it to the right.
    *   **Wind Application:** In every call to `step(action)` (whether through `see()` or `do()`), the lander's current continuous (x, y) position is used to determine which cell of the `wind_map` it currently occupies. The wind strength from that cell (`self.current_wind`) is then applied as a horizontal force to the lander's physics body (`self._env.unwrapped.lander.ApplyForceToCenter((self.current_wind, 0.0), True)`). This force application occurs *before* the agent's chosen `action` (or the intervened action) is processed by the underlying `LunarLander-v3` environment.

*   **Causal Diagram/Structure (as per `get_graph`):**
    The environment defines its causal graph with nodes: `Wind(U)`, `State(S)`, `Action(X)`, `Reward(Y)`, and `Next_State(S')`.
    *   `Wind(U) -> Next_State(S')`: The wind force directly affects the lander's physics, thus influencing its next state.
    *   `Wind(U) -> Action(X)`: If the agent's policy is wind-aware (i.e., `policy(observation, current_wind)`), the wind can influence the action chosen by the policy.
    *   `State(S) -> Action(X)`: The current state of the lander (observation) influences the action chosen by the policy.
    *   `State(S) -> Reward(Y)`: The current state can directly lead to rewards (e.g., landing or crashing).
    *   `Action(X) -> Reward(Y)`: The chosen action influences the reward (e.g., fuel cost, or contributing to landing/crashing).
    *   `State(S) -> Next_State(S')`: The current state is a primary determinant of the next state.
    *   `Action(X) -> Next_State(S')`: The chosen action also determines the next state.
    The wind acts as an exogenous variable that directly perturbs the system's dynamics and can also inform a sophisticated agent's policy.

*   **Usage and Counterfactual `do()` operations:**
    Interaction occurs via the `LunarLanderPCH` (Probabilistic Causal Hacked) wrapper over `LunarLanderSCM`.
    *   `reset()`: Initializes the `LunarLander-v3` environment and, crucially, calls `sample_u()` to generate a new `wind_map` for the entire episode. Returns `(initial_observation, info_dict)` where `info_dict` includes the `wind_map`.
    *   `see()`: Invokes the agent's policy (`self.env.action()`, which is `self.policy(obs, self.current_wind)`) to get an action. This action is then passed to `self.env.step()`. The wind force (based on the lander's position in the current `wind_map`) is applied before this action is executed by the base environment.
    *   `do(action)`: The specified interventional `action` is passed directly to `self.env.step()`. Similar to `see()`, the wind force from the current `wind_map` is applied based on the lander's position *before* the intervened `action` is executed by the base environment.
    *   **Counterfactual Queries:** `do(action)` allows testing "what if I took `action` right now?" The outcome will be subject to the wind conditions defined by the `wind_map` that was sampled at the beginning of the current episode and the lander's position when `do()` is called. To analyze behavior under a *hypothetical* wind map different from the current episode's, one would need to be able to set the `wind_map` attribute in the SCM before calling `do()`.
    *   **Discretization for Learning:** Since Lunar Lander has a continuous observation space, applying discrete-state algorithms like UCBVI (as seen in `test/test_lander.py` and `test/test_lander_ucbvi.py`) requires discretizing the 8-dimensional observation vector into a finite number of states. This is a practical challenge when bridging continuous environments with such algorithms and is noted in the test scripts (`NUM_STATES = 5*6*5*6`).

### 2.3 Cartpole Wind (Continuous)
*   **Description:**
    This environment adapts Gymnasium's `CartPole-v1`. The setup involves a cart moving horizontally with a pole hinged on top. The agent applies a horizontal force (left or right) to the cart. The goal is to keep the pole balanced upright for as long as possible. The observation space is continuous and 4-dimensional: cart position, cart velocity, pole angle (radians from vertical), and pole angular velocity. Actions are discrete: 0 (push cart left) and 1 (push cart right). A reward of +1 is provided for every time step the pole remains upright.

*   **Latent Variable Implementation (Wind):**
    A latent "wind" variable introduces a fluctuating horizontal force on the cart.
    *   **Per-Step Wind Gust:** Unlike FrozenLake or LunarLander where wind is sampled once per episode, in `CartPoleWindSCM`, the wind (`self.current_wind`) is a scalar value sampled from a Normal distribution (configurable `wind_mean` and `wind_std`) *at every time step*. This is done by calling `sample_u()` at the end of each `step()` method and also during `reset()`.
    *   **Wind Application:** At the beginning of each `step(action)` call (whether from `see()` or `do()`), the `self.current_wind` (which was sampled at the end of the *previous* step or during `reset`) is directly added to the cart's velocity component (`state[1]`) of the environment's internal state. This modification happens *before* the agent's chosen `action` (or the intervened action) is processed by the underlying `CartPole-v1` mechanics.

*   **Causal Diagram/Structure (as per `get_graph`):**
    The `get_graph` property defines a structure identical to that of LunarLander, with nodes: `Wind(U)`, `State(S)`, `Action(X)`, `Reward(Y)`, and `Next_State(S')`.
    *   `Wind(U) -> Next_State(S')`: The per-step wind directly alters the cart's velocity, thereby influencing the next state.
    *   `Wind(U) -> Action(X)`: If the agent's policy is designed to be wind-aware (i.e., `policy(observation, current_wind)`), the current wind gust can influence the action selected by the policy.
    *   `State(S) -> Action(X)`: The current state (observation) is used by the policy to choose an action.
    *   `State(S) -> Reward(Y)`: The state (particularly pole angle) determines if the episode terminates, thus affecting rewards.
    *   `Action(X) -> Reward(Y)`: While actions don't have direct costs, they influence the state trajectory which leads to rewards.
    *   `State(S) -> Next_State(S')`: The current state is a key determinant of the next state.
    *   `Action(X) -> Next_State(S')`: The chosen action also determines the next state.
    The main distinguishing feature is that `U` (Wind) is resampled at each step, making it a rapidly changing exogenous influence.

*   **Usage and Counterfactual `do()` operations:**
    Interactions are managed by `CartPoleWindPCH` wrapping `CartPoleWindSCM`.
    *   `reset()`: Initializes the `CartPole-v1` environment. It also calls `sample_u()` to determine the wind for the very first step. Options exist to set a non-zero initial pole angle (`init_theta_mean`, `init_theta_std`), though it defaults to zero.
    *   `see()`: The agent's policy (`self.env.action()`, which is `self.policy(obs, self.current_wind)`) gets an action `a`. This action is then passed to `self.env.step(a)`. Inside `step()`: 
        1. The wind sampled at the *end of the previous step* (or `reset`) is applied to the cart's velocity.
        2. The action `a` is processed by the base `CartPole-v1` environment.
        3. A *new* wind value is sampled via `sample_u()` for the *next* step.
    *   `do(action)`: The specified interventional `action` is passed to `self.env.step(action)`. The sequence of events within `step()` is the same as for `see()`: current wind applied, intervened action processed, new wind sampled for the next step.
    *   **Counterfactual Queries:** `do(action)` tests the effect of an action given the wind that was determined at the end of the previous time step. Because wind changes every step, counterfactuals about sustained wind conditions would require temporarily overriding the per-step `sample_u()` mechanism to hold wind constant for a sequence of `do()` calls.
    *   **Discretization for Learning:** As noted in your summary ("Continuous, exposes discrete", "discretizations"), CartPoleWind is a continuous environment. For algorithms that operate on discrete states (e.g., UCBVI, UCBQ as tabular methods), the 4-dimensional continuous observation space needs to be discretized. This process involves dividing the range of each observation variable into a set of bins, mapping continuous observations to discrete state indices.

## 3. Algorithms

This section details the learning algorithms implemented and used in this project, focusing on their relevance to causal and counterfactual reasoning.

### 3.1 UCBVI (Upper Confidence Bound Value Iteration)
*   **Description:**
    The `UCBVI` class implements the Counterfactual Upper Confidence Bound Value Iteration algorithm, noted as "Alg. 26 Ctf-UCBVI" in the project summary. This algorithm is designed for model-based reinforcement learning in episodic Markov Decision Processes (MDPs). It aims to learn an optimal policy by maintaining optimistic estimates of Q-values and value functions.

*   **Counterfactual Nature:**
    The core counterfactual aspect of this UCBVI implementation lies in its Q-value and visitation count structures:
    *   `Q[s, x_int, a]`: Represents the optimistic Q-value for being in state `s`, where the *behavioral* or *intended* action was `x_int`, but the algorithm chose to *apply* action `a`.
    *   `N[s, x_int, a]`: Counts the number of times this specific scenario (state `s`, intended `x_int`, applied `a`) has occurred.
    This formulation allows the algorithm to learn the consequences of deviating from a baseline or intended policy. It answers the question: "Given that my current policy/intention is to do `x_int` in state `s`, what is the value if I *instead* choose to do action `a`?"
    The `act(s, x_int)` method then selects the action `a` that maximizes `Q[s, x_int, a]`, effectively choosing the best *actual* action to take, informed by what the intended action was.

*   **Implementation Details:**
    *   **Initialization (`__init__`):
        *   Takes `num_states` (S), `n_actions` (A, representing the size of the action space for both intended and applied actions), `horizon` (H), and `delta` (confidence parameter).
        *   `Q[S, A, A]`: Optimistic Q-values.
        *   `V[S, A]`: Optimistic state-value function, where `V[s, x_int] = max_a Q[s, x_int, a]`.
        *   `N[S, A, A]`: Visitation counts, initialized to 1 (to prevent division by zero for bonuses).
        *   `R[S, A, A]`: Average rewards for `(s, x_int, a)` experiences.
        *   `P[S, A, A, S]`: Transition probabilities `P(s_next | s, x_int, a)`.
    *   **Optimism Bonus (`bonus`):
        *   Calculated as `sqrt(2 * log(1/delta) / N[s, x_int, a])`.
        *   This bonus is added to the Q-value updates to encourage exploration of (state, intended_action, applied_action) combinations that have been tried less frequently.
    *   **Update (`update(s, x_int, a, r, s_next)`):
        *   Called after each environment step where state `s`, intended action `x_int`, applied action `a`, reward `r`, and next state `s_next` are observed.
        *   Increments `N[s, x_int, a]`. 
        *   Updates `R[s, x_int, a]` and `P[s, x_int, a, s_next]` using exponential moving averages based on the new experience.
    *   **Planning (`plan(num_sweeps)`):
        *   Performs `num_sweeps` (or `H` if `num_sweeps` is None) iterations of optimistic backward value iteration.
        *   In each iteration, it updates `Q` and `V`:
            1.  `bonus = self.bonus()`
            2.  `expected_next_value[s, x_int, a] = sum_{s_next} P[s, x_int, a, s_next] * V[s_next, x_int]`
                (Note: The `V` lookup uses `V[s_next, x_int]`, implying the value of the next state `s_next` is considered under the assumption that the *same intended action `x_int`* from the current step would persist or be relevant for valuing `s_next`. This is a specific aspect of this value iteration formulation.)
            3.  `Q[s, x_int, a] = R[s, x_int, a] + bonus[s, x_int, a] + expected_next_value[s, x_int, a]`
            4.  `V[s, x_int] = max_a' Q[s, x_int, a']` (Maximizing over the *applied* action `a'`)
    *   **Action Selection (`act(s, x_int)`):
        *   Given the current state `s` and the intended action `x_int` (typically from a behavioral policy), this method returns the action `a` that maximizes the optimistic Q-value `Q[s, x_int, a]`. This is the action the UCBVI agent decides to actually take.
    *   **Model Reset (`reset_model()`):**
        *   Resets `Q, V, N, R, P` to their initial states, allowing for fresh learning runs.

### 3.2 UCBQ
*   **Description:**
    The `UCBQ` class implements UCB-Q learning, a model-free reinforcement learning algorithm that incorporates the Upper Confidence Bound (UCB) principle for exploration. This aligns with "Standard algorithm 27 from 9.4 of the textbook" mentioned in the project summary. It learns Q-values for state-action pairs directly from experience.

*   **Implementation Details:**
    *   **Initialization (`__init__`):
        *   Takes `n_states` (S), `n_actions` (A), and `delta` (confidence parameter).
        *   `self.Q[S, A]`: The Q-table, storing values for state-action pairs, initialized to zeros.
        *   `self.N_sa[S, A]`: Visitation counts for each state-action pair `(s, a)`, initialized to 1 (to avoid division by zero in bonus calculation and ensure initial exploration).
        *   `self.log_inv_delta`: Stores `log(1/delta)` for computational efficiency.
    *   **Exploration Bonus (`bonus(s, a)`):
        *   Calculates an exploration bonus for a state-action pair `(s, a)` using the formula: `sqrt(2 * self.log_inv_delta / self.N_sa[s, a])`.
        *   This bonus is inversely proportional to the square root of the number of times `(s, a)` has been visited, encouraging the algorithm to try actions in states that it hasn't explored much.
    *   **Update (`update(s, a, r, s_next)`):
        *   This method is called after the agent takes action `a` in state `s`, receives reward `r`, and transitions to state `s_next`.
        *   Increments the visit count `self.N_sa[s, a]`.
        *   The learning rate `alpha` is set to `1.0 / self.N_sa[s, a]`.
        *   The UCB exploration bonus `b` is calculated for the current `(s, a)` pair.
        *   The Q-value update target is: `target = r + max_a' Q[s_next, a'] + b`.
            *   This target consists of the immediate reward `r`.
            *   Plus the maximum Q-value of the next state `s_next` (i.e., `np.max(self.Q[s_next])`), which represents the estimated future reward from `s_next` onwards (standard Q-learning bootstrap).
            *   Plus the exploration bonus `b` for the current `(s,a)` pair. This makes the Q-value update for `Q[s,a]` optimistic.
        *   The Q-value is updated using the formula: `self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * target`.
    *   **Action Selection (`act(s)`):
        *   The agent selects an action greedily based on the current Q-values: `action = argmax_a Q[s, a]`.
        *   Exploration is implicitly handled because the Q-values themselves are optimistic due to the bonus term in their update rule. Thus, acting greedily on these optimistic Q-values naturally leads to exploration of less-certain (higher bonus) actions.

*   **Relation to UCBVI:**
    *   **Model-Free vs. Model-Based:** UCBQ is model-free, learning Q-values directly. UCBVI is model-based, learning a model of the environment (transition probabilities `P` and rewards `R`) and then using planning (value iteration) to compute Q-values.
    *   **Q-Value Structure:** UCBQ uses a standard `Q[s, a]` table. UCBVI uses a more complex `Q[s, x_int, a]` structure to explicitly handle intended vs. applied actions, making it directly suited for counterfactual reasoning in the context of a behavioral policy.
    *   **Bonus Application:** In UCBQ, the bonus for `(s,a)` is added to the TD target when updating `Q[s,a]`. In UCBVI, the bonus for `(s, x_int, a)` is added during the Bellman backup in its planning phase.

### 3.3 Interaction with Environments
*   **Base Policy (Uniform Actions):**
    Your summary notes: "Should use a base policy of uniform actions." This is a common approach for the behavioral policy (`x_int` in UCBVI) or for initial exploration. In this setup:
    *   For UCBVI, the `x_int` provided to `agent.act(s, x_int)` and `agent.update(s, x_int, ...)` would be an action sampled uniformly at random from the available actions in state `s`.
    *   For UCBQ, while it doesn't explicitly use an `x_int`, if it were being compared or used in a context alongside UCBVI where a behavioral policy is relevant, this uniform random action could be what the environment executes if UCBQ's chosen action is not the one applied (though UCBQ itself just learns from `(s,a,r,s')` where `a` is what it chose).

*   **Algorithm Workflow (General Idea, adapted from your summary):
    The typical interaction loop for these learning algorithms, especially fitting the UCBVI model where an intended action `x_int` is part of the process, would be:
    1.  **Observe Current State (`s`):** Get the current state from the environment.
    2.  **Determine Intended Action (`x_int`):** Sample an action from the base behavioral policy (e.g., uniform random action). This is `x_int`.
    3.  **Algorithm Selects Action (`a`):** The learning algorithm (e.g., `ucbvi_agent.act(s, x_int)` or `ucbq_agent.act(s)`) chooses the action `a` to actually execute. For UCBVI, this choice is based on `Q[s, x_int, a]`. For UCBQ, it's based on `Q[s, a]` (where `a` would be the action taken).
    4.  **Execute Action in Environment:** The chosen action `a` is performed in the environment using `env.do(a)` (or `env.step(a)` if `do` is not strictly required by the specific setup for that algorithm's data collection).
    5.  **Observe Outcome:** Receive the next state (`s_next`), reward (`r`), and termination status from the environment.
    6.  **Update Algorithm:** The algorithm updates its internal model/values. 
        *   For `UCBVI`: `ucbvi_agent.update(s, x_int, a, r, s_next)`.
        *   For `UCBQ`: `ucbq_agent.update(s, a, r, s_next)`.
    7.  **Planning (for Model-Based Algos like UCBVI):** Periodically, `ucbvi_agent.plan()` is called to re-compute Q-values based on the updated model.
    8.  Repeat from step 1 until the episode ends or training is complete.

### 3.4 Algorithm Parameter Tuning Insights

A crucial part of applying algorithms like UCBVI effectively involves careful parameter tuning. Initial experiments, particularly with the FrozenLake environment, revealed that certain parameters have a profound impact on learning performance.

*   **Decoupling Planning Horizon from Reward Scaling:**
    A key finding was the necessity to decouple the `horizon` parameter. Originally, a single `horizon` was used for:
    1.  The number of planning sweeps in value iteration (an integer).
    2.  Clipping optimistic Q-values (`Q = min(horizon, ...)`).
    3.  Scaling the exploration bonus (`bonus = c_bonus * horizon * sqrt(...)`).

    This conflation led to issues: a high `horizon` (e.g., 100, for planning) resulted in excessive optimism and poor learning, while a low `horizon` (e.g., 1.0, for reward scaling) was unsuitable for planning sweeps. The solution was to modify `CtfUCBDriver (UCBVI)` and `ucb_q` to accept two distinct parameters:
    *   `planning_sweeps` (integer): For the number of value iteration sweeps (e.g., 100).
    *   `max_episode_reward` (float): For Q-value clipping and bonus scaling, set to reflect the true maximum achievable reward in an episode (e.g., 1.0 for FrozenLake).

*   **`c_bonus` (Exploration Constant):**
    Once `max_episode_reward` was set appropriately to reflect the actual reward scale of the environment, a `c_bonus` value of **1.0** was found to be effective for FrozenLake. This provided a better balance for exploration compared to larger theoretical values when `max_episode_reward` was also large and miscalibrated. The effective exploration pressure results from the interplay of `c_bonus * max_episode_reward`.

*   **Successful Configuration (FrozenLake Example):**
    For the 4x4 non-slippery FrozenLake with wind, the following UCBVI parameters yielded good learning:
    *   `planning_sweeps`: 100
    *   `max_episode_reward`: 1.0
    *   `c_bonus`: 1.0
    *   `delta`: 0.1

*   **Implications for Other Environments:**
    These parameters, especially `max_episode_reward` and `c_bonus`, are highly sensitive and will require specific tuning for other environments like Lunar Lander and Cartpole Wind, based on their unique reward structures and scales. For instance:
    *   **Lunar Lander:** `max_episode_reward` might be in the range of 100-250, given its scoring. The Gymnasium documentation notes: "Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points. If the lander moves away from the landing pad, it loses reward. If the lander crashes, it receives an additional -100 points. If it comes to rest, it receives an additional +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points." This rich, shaped reward structure means the total episodic reward can exceed the +100 from a successful landing alone. Initial runs with `max_episode_reward=250.0` (reflecting this higher potential total reward), `planning_sweeps=100`, `c_bonus=1.0`, and an improved discretization (1728 states including position, velocity, angle, and angular velocity) over 1000 episodes show that while the agent explores and the average total episodic reward shows a slight upward trend from highly negative values, consistent positive rewards or successful landings have not yet been achieved. This underscores the significant challenge of this environment for tabular UCBVI, likely necessitating more extensive training, further refinements in state representation, or different algorithmic approaches.
    *   **Cartpole Wind:** If max episode length is 200 and reward is +1 per step, `max_episode_reward` could be near 200.
    The quality of state discretization for these continuous environments is also a critical precursor to successful learning with tabular UCBVI.

These tuning insights are vital for achieving meaningful learning results and should be considered when applying UCBVI or similar optimistic algorithms to new environments.

## 4. Testing and Learning Demonstrations

This section outlines how the environments and algorithms are tested, including basic interaction examples derived from the test scripts and notebooks, and more complex learning scenarios demonstrating the application of UCBVI and UCBQ.

### 4.1 Basic Environment Interaction

The primary mode of interaction with the custom environments is through their respective PCH (Probabilistic Causal Hacked) wrappers (e.g., `FrozenLakePCH`, `LunarLanderPCH`, `CartPoleWindPCH`). These wrappers provide a consistent interface for observational and interventional steps.

*   **Initialization:**
    *   Environments are initialized by instantiating their PCH class. For example: `env = FrozenLakePCH(is_slippery=True, wind_probabilities=...)` or `env = LunarLanderPCH(wind_mean=0.0, wind_std=0.2)` or `env = CartPoleWindPCH(wind_std=0.01)`.
    *   Key parameters like wind characteristics, slipperiness (for FrozenLake), or max episode steps can be set during initialization.

*   **Resetting an Episode:**
    *   `obs, info = env.reset()`: This command starts a new episode.
    *   It returns an initial observation `obs` and an `info` dictionary.
    *   Crucially, for environments with episode-level latents (FrozenLake, LunarLander), `reset()` triggers the sampling of the latent variable for that episode (e.g., `wind_map`). For CartPoleWind, it samples the initial wind for the first step.
    *   The `info` dictionary often contains the sampled latent(s) (e.g., `info['wind_map']` for FrozenLake and LunarLander).

*   **Observational Steps (`see()`):
    *   `intended_action, obs, reward, terminated, truncated, info = env.see()`:
    *   This method simulates a step where the action is chosen by the environment's internal SCM policy (`self.scm.action()`, which uses `self.scm.policy(observation, current_wind)`).
    *   The SCM's policy might be a simple random policy by default or can be replaced by a more complex learned policy.
    *   The `intended_action` is the action chosen by this internal policy.
    *   The environment then transitions based on this `intended_action` and the current latent variable's state (e.g., wind force application, wind override if not slippery in FrozenLake).
    *   Returns the `intended_action`, new `obs`, `reward`, `terminated` and `truncated` flags, and an `info` dictionary which may contain details like the actual wind effect in that step.

*   **Interventional Steps (`do(action)`):
    *   `obs, reward, terminated, truncated, info = env.do(chosen_action)`:
    *   This method allows the agent/experimenter to force a specific `chosen_action` to be taken in the environment.
    *   The environment transitions based on this `chosen_action` and the current latent variable's state (similar to `see()`, the latent variable's influence is still active during the execution of the `do` action).
    *   This is the primary mechanism for asking counterfactual-style questions like "what happens if I take `action` now, given the current (latent) state of the world?"
    *   Returns the new `obs`, `reward`, `terminated` and `truncated` flags, and an `info` dictionary.

*   **Accessing/Observing Latent Variables:**
    *   **FrozenLake:** The `wind_map` is generated at `reset()` and is available in the `info` dictionary returned by `reset()`. During `step()` (called by `see()` or `do()`), `info['wind_in_cell']` and potentially `info['wind_overrode_action_to']` indicate the wind's effect.
    *   **LunarLander:** The `wind_map` (grid of wind forces) is generated at `reset()` and available in `info['wind_map']`. The specific `current_wind` force applied at each step (derived from the lander's position on the `wind_map`) is used internally within `step()` and influences the transition. The SCM's `policy` can also be made aware of `current_wind`.
    *   **CartPoleWind:** The `current_wind` (a scalar force) is sampled at *every step* (within `step()` and initially at `reset()`). The SCM's policy `self.scm.policy(observation, current_wind)` can use it. The wind's effect (added to cart velocity) is applied internally in `step()`.

*   **Rendering:**
    *   Most environments support `env.render()` for visualization, often with options to show the effects of latent variables (e.g., wind indicators in LunarLander or FrozenLake).

*   **Example Usage from Test Scripts:**
    *   `test/test_frozenlake.py`, `test/test_lander.py`, and `test/test_cartpole.ipynb` all demonstrate loops that call `env.reset()` at the start of episodes, and then either `env.see()` or `env.do(specific_action)` for a certain horizon or until termination. They collect statistics like success rates (FrozenLake, LunarLander) or average rewards (CartPole) to compare observational outcomes versus interventional outcomes.

### 4.2 Learning with Algorithms

This subsection details how the UCBVI and UCBQ algorithms are applied to the custom environments, drawing from the provided learning test scripts.

*   **Setup for Learning Experiments:**
    *   **Environments:** The PCH versions of the environments are used (e.g., `FrozenLakePCH`, `LunarLanderPCH`, `CartPoleWindPCH`).
    *   **Agents:** Instances of `UCBVI` or `UCBQ` are created with parameters appropriate for the environment (number of states, number of actions, horizon).
    *   **Discretization:** For continuous environments (LunarLander, CartPoleWind) when used with these tabular algorithms, a discretization step is necessary. This involves defining a way to map the continuous observation vector to a discrete state index. For example, `agent.discretize = lambda obs: ...` is used to provide this mapping to the agent.
        *   FrozenLake: Observations are already discrete, so `discretize` is `lambda obs: int(obs)`.
        *   LunarLander (`test_lander_ucbvi.py`): A specific discretization scheme (e.g., `NUM_STATES = 5*6*5*6`) is implied and would be implemented in an agent-specific `discretize` method.
        *   CartPoleWind (`test_cartpole_algos.ipynb`): Similarly, requires a discretization strategy if UCBVI/UCBQ are applied.
    *   **Behavioral Policy (for UCBVI):** UCBVI learns a counterfactual policy. The `env.see()` method is used to get an `intended_action (x_int)` from the environment SCM's default policy (often a uniform random policy). This `x_int` is then fed into `agent.act(s, x_int)` and `agent.update(s, x_int, a, r, s_next)`.

*   **Performance Metrics:**
    *   **Success Rate / Average Reward:** Depending on the environment, performance is often measured as the proportion of episodes where the goal is successfully reached (e.g., FrozenLake, LunarLander landing) or the average total reward per episode (e.g., CartPole).
    *   **Cumulative Regret:** This is a key metric for UCB-style algorithms. It measures the difference in performance (e.g., success or reward) between the learning agent and an optimal or baseline policy over time. 
        *   Typically, a baseline performance (`best_p` or `best_reward`) is established (e.g., by running a fixed interventional policy known to be good, or an oracle policy).
        *   Per-episode regret = `baseline_performance - agent_performance_in_episode`.
        *   Cumulative regret is the sum of per-episode regrets.
    *   **Time to Success:** Some analyses might look at how many episodes it takes for the agent to consistently achieve successful outcomes.

*   **Results and Observations for Specific Environments:**

    *   **Frozen Lake with UCBVI (from `.important/test_frozenlake_ucbvi.py`):
        *   **Objective:** Train UCBVI to find a policy that maximizes the chance of reaching the 'G' (goal) state.
        *   **Process:**
            1.  An instance of `FrozenLakePCH` (e.g., `is_slippery=True`) is created.
            2.  A `UCBVI` agent is initialized (`num_states=16` for 4x4 map, `n_actions=4`).
            3.  In each episode, within the horizon:
                a.  `x_int, obs_t, ... = env.see()`: The environment SCM's policy (e.g., random) provides an `intended_action (x_int)`.
                b.  `s = agent.discretize(obs_t)`.
                c.  `a = agent.act(s, x_int)`: UCBVI chooses the counterfactually optimal action `a`.
                d.  `obs_next, r, done, _ = env.do(a)`: Action `a` is executed.
                e.  `s_n = agent.discretize(obs_next)`.
                f.  `agent.update(s, x_int, a, r, s_n)`: UCBVI's model (N, R, P) is updated.
            4.  `agent.plan()`: After each episode, UCBVI re-plans (value iteration) using its updated model.
        *   **Metrics & Outputs:**
            *   A baseline success probability (`p_int`) is calculated by running a fixed interventional policy (e.g., always `do(RIGHT)`).
            *   The success rate of UCBVI (`p_ucb`) is calculated.
            *   Cumulative regret (`cum_regret = sum (p_int - UCBVI_episode_success)`) is tracked and plotted.
            *   The script outputs the baseline success rate, UCBVI success rate, and the final cumulative regret, saving a plot of the regret curve.
        *   **Expected Outcome:** UCBVI is expected to learn a policy that improves its success rate over time, and the cumulative regret should ideally grow sub-linearly, indicating learning.

    *   **Lunar Lander with UCBVI (from `.important/test_lander_ucbvi.py`):
        *   **Objective:** Train UCBVI to successfully land the lunar module.
        *   **Discretization Challenge:** LunarLander has a continuous 8-dimensional observation space. To apply the tabular UCBVI algorithm, a `discretize(obs)` function is implemented:
            *   It considers the first four observation variables: cartesian coordinates (x, y) and linear velocities (vx, vy).
            *   Each variable is clipped to a specific range and then binned using `np.digitize` (e.g., x into 5 bins, y into 6 bins, vx into 5 bins, vy into 6 bins).
            *   This results in a total of `5 * 6 * 5 * 6 = 900` discrete states.
            *   This `discretize` function is assigned to `agent.discretize`.
        *   **Process:**
            1.  An instance of `LunarLanderPCH` is created.
            2.  A `UCBVI` agent is initialized (`num_states=900`, `n_actions=4`, `horizon=400`, specific `delta`).
            3.  The learning loop is similar to FrozenLake's UCBVI:
                a.  Get `intended_action (x_int)` via `env.see()`.
                b.  Discretize current observation to state `s`.
                c.  UCBVI agent chooses `a = agent.act(s, x_int)`.
                d.  Execute `a` via `env.do(a)`.
                e.  Discretize next observation to `s_n`.
                f.  `agent.update(s, x_int, a, r, s_n)`.
            4.  `agent.plan(num_sweeps=PLAN_SWEEPS)` is called periodically (e.g., every `PLAN_PERIOD = 3` episodes) to update Q-values.
        *   **Metrics & Outputs:**
            *   A baseline success probability (`p_int`) is established by always taking a fixed action (e.g., `NOOP=0`) interventionally (`env.do(0)`).
            *   **Current Status & Expected Outcome:** UCBVI is expected to learn a policy that improves its success rate. The LunarLander environment provides shaped rewards: positive for approaching the pad, slowing down, and leg contact; negative for fuel usage, plus a terminal +100 for safe landing or -100 for crashing. "Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points... Solved is 200 points." (Gymnasium documentation). However, experiments with up to 1000 episodes, using a 1728-state discretization (including x, y, vx, vy, theta, omega) and tuned parameters (`max_episode_reward=250.0`, `c_bonus=1.0`), show that while the average total episodic reward trends slightly upwards from very negative values, the agent still predominantly crashes and has not achieved consistent positive rewards or successful landings. This indicates that more sophisticated state representations, significantly more training episodes, or alternative algorithms may be needed for this complex environment.

    *   **Cartpole Wind with UCBVI/UCBQ (from `test/test_cartpole_algos.ipynb`):**
        *   **(Note: This notebook was previously missing/corrupted. The following describes the intended test setup based on inferences and common practices.)**
        *   **Objective:** Train UCBVI/UCBQ to balance the pole for as long as possible.
        *   **Process:**
            1.  An instance of `CartPoleWindPCH` is created.
            2.  A `UCBVI` agent is initialized with parameters appropriate for the environment.
            3.  The learning loop is similar to FrozenLake's UCBVI:
                a.  Get `intended_action (x_int)` via `env.see()`.
                b.  Discretize current observation to state `s`.
                c.  UCBVI agent chooses `a = agent.act(s, x_int)`.
                d.  Execute `a` via `env.do(a)`.
                e.  Discretize next observation to `s_n`.
                f.  `agent.update(s, x_int, a, r, s_n)`.
            4.  `agent.plan(num_sweeps=PLAN_SWEEPS)` is called periodically (e.g., every `PLAN_PERIOD = 3` episodes) to update Q-values.
        *   **Metrics & Outputs:**
            *   The script collects statistics like the average total reward per episode.
            *   The script outputs the final average total reward.
        *   **Expected Outcome:** UCBVI/UCBQ is expected to learn a policy that improves its average total reward over time.