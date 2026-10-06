import numpy as np
from gymnasium import spaces

from causal_gym import SCM, PCH
from causal_gym.core import ObsType, Graph

MOVES = np.array(
  [
    [0, 0], # hover
    [-1, 0], # up
    [0, -1], # left
    [1, 0], # down
    [0, 1] # right
  ],
  dtype = np.int64,
)

WIND_VALUES = np.array(
  [
    [0, 0],
    [-1, 0],
    [0, -1],
    [1, 0],
    [0, 1],
  ],
  dtype=np.int64,
)

WIND_PROBS = np.array(
  [0.6, 0.1, 0.1, 0.1, 0.1],
  dtype=np.float64,
)


class RescueDroneSCM(SCM):

  def __init__(self, wind_probs=WIND_PROBS, grid_size=5, 
               drone_start=(0, 0), victim_start=(0, 2), 
               random_start=False, max_steps=2
               ):
    super().__init__()

    self.grid_size = grid_size
    self.max_steps =  max_steps

    self.action_space = spaces.Discrete(len(MOVES))
    self.observation_space = spaces.MultiDiscrete([self.grid_size] * 4)

    self.drone_start = np.array(drone_start, dtype = np.int64)
    self.victim_start = np.array(victim_start, dtype = np.int64)

    self.drone_pos = self.drone_start.copy()
    self.victim_pos = self.victim_start.copy()

    self.u = None
    self.steps = 0
    self.rescued = False
    self._policy = self._natural_policy
    self.wind_probs = np.array(wind_probs, dtype=float) # parametrize wind probability
    self.random_start = random_start


  def _natural_policy(self, observation, u):
    drone_pos = observation[:2]
    victim_pos = observation[2:]

    candidate_pos = drone_pos + MOVES + u
    candidate_pos = np.clip(candidate_pos, 0, self.grid_size - 1, )

    dist = np.abs(candidate_pos - victim_pos).sum(axis=1)

    return int(np.argmin(dist))


  def sample_u(self):
    wind_index = self.np_random.choice(
      len(WIND_VALUES),
      p=self.wind_probs,
    )

    return WIND_VALUES[wind_index].copy()

  # parametrize the step size.
  # Also the early stop etc.
  def reset(self, *, seed=None, options=None):
    super().reset(seed = seed, options = options)

    if self.random_start:
      self.drone_pos = self.np_random.integers(self.grid_size, size=2)

      # Match the initial Manhattan distance to the step budget.
      distance = self.max_steps
      offsets = np.array([
        [dx, dy]
        for dx in range(-distance, distance + 1)
        for dy in range(-distance, distance + 1)
        if abs(dx) + abs(dy) == distance
      ])

      candidates = self.drone_pos + offsets
      inside = ((candidates >= 0) & (candidates < self.grid_size)).all(axis=1)
      candidates = candidates[inside]

      if len(candidates) == 0:
        raise ValueError(
          f"No victim position at Manhattan distance {distance} "
          f"from drone position {self.drone_pos.tolist()} "
          f"in a {self.grid_size}x{self.grid_size} grid."
        )

      index = self.np_random.integers(len(candidates))
      self.victim_pos = candidates[index].copy()
    else:
      self.drone_pos = self.drone_start.copy()
      self.victim_pos = self.victim_start.copy()

    self.steps = 0
    self.rescued = False
    self.u = self.sample_u()

    return self.observation(), {}
  

  def observation(self):
    return np.concatenate([self.drone_pos, self.victim_pos]).astype(np.int64)


  def action(self):
    if self.u is None:
        raise RuntimeError("Call reset() before action().")

    return int(self._policy(self.observation(), self.u, ))


  def step(self, action):
    if self.u is None:
        raise RuntimeError("Call reset() before step().")

    if not self.action_space.contains(action):
        raise ValueError(f"Invalid action: {action}")

    action = int(action)

    self.drone_pos = np.clip(
        self.drone_pos + MOVES[action] + self.u,
        0,
        self.grid_size - 1,
    )

    self.steps += 1
    self.rescued = np.array_equal(
        self.drone_pos,
        self.victim_pos,
    )

    reward = 1.0 if self.rescued else -0.01
    terminated = self.rescued
    truncated = (
        self.steps >= self.max_steps
        and not terminated
    )

    info = {
        "action": action,
        "steps": self.steps,
        "rescued": self.rescued,
    }

    if terminated or truncated:
      self.u = None
    else:
      self.u = self.sample_u()

    return (
        self.observation(),
        reward,
        terminated,
        truncated,
        info,
    )

  @property
  def get_graph(self):
    nodes = [
      {"name": "S_1", "label": "Initial state"},
    ]
    edges = []

    for i in range(1, self.max_steps + 1):
      state = f"S_{i}"
      action = f"X_{i}"
      next_state = f"S_{i + 1}"
      reward = f"Y_{i}"

      nodes.extend([
        {"name": action, "label": f"Stage {i} action"},
        {"name": next_state, "label": f"State after stage {i}"},
        {"name": reward, "label": f"Stage {i} reward"},
      ])

      edges.extend([
        {"from_": state, "to_": action, "type_": "directed"},
        {"from_": state, "to_": next_state, "type_": "directed"},
        {"from_": action, "to_": next_state, "type_": "directed"},
        {"from_": next_state, "to_": reward, "type_": "directed"},
        {"from_": action, "to_": next_state, "type_": "bidirected"},
      ])

    return Graph(nodes=nodes, edges=edges)


class RescueDronePCH(PCH):

  def __init__(self, wind_probs=WIND_PROBS, grid_size=5, 
               drone_start=(0, 0), victim_start=(0, 2), 
               random_start=False, max_steps=2, **kwargs):
    self.env = RescueDroneSCM(
      wind_probs=wind_probs,
      grid_size=grid_size,
      drone_start=drone_start,
      victim_start=victim_start,
      random_start=random_start,
      max_steps=max_steps,
      )
    super().__init__(**kwargs)


  def see(self, see_policy=None):
    if self.env.u is None:
      raise RuntimeError("Call reset() before see().")
    
    state = self.env.observation()
    stage_u = self.env.u.copy()

    if see_policy is None:
      action = self.env.action()
    else:
      action = int(see_policy(state, stage_u))

    next_observation, reward, terminated, truncated, info = (
      self.env.step(action)
    )

    info["natural_action"] = action
    info["u"] = stage_u

    return next_observation, reward, terminated, truncated, info
  

  def do(self, do_policy):
    if self.env.u is None:
      raise RuntimeError("Call reset() before do().")

    state = self.env.observation()
    action = int(do_policy(state))

    next_observation, reward, terminated, truncated, info = (
      self.env.step(action)
    )

    info["action"] = action

    return next_observation, reward, terminated, truncated, info
  

  def ctf_do(self, ctf_policy):
    if self.env.u is None:
      raise RuntimeError("Call reset() before ctf_do().") # double check

    state = self.env.observation()
    natural_action = self.env.action()

    action = int(
      ctf_policy(state, natural_action)
    )

    next_observation, reward, terminated, truncated, info = (
      self.env.step(action)
    )

    info["natural_action"] = natural_action
    info["action"] = action

    return next_observation, reward, terminated, truncated, info  

  # default ctf policy
