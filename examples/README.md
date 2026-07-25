# Example notebooks

## Notebook audit

Last audited: **2026-07-25**. Each notebook was executed from its first cell in
an independent `causal-gymnasium` kernel with headless SDL and a 180-second
per-cell timeout. `Execution` only reports whether the notebook completes.
`Output check` separately reports whether the result matches a stated theory
target or, for visual/training examples, whether the generated behavior is
reasonable. Random estimates use sampling tolerances rather than exact equality.
Stored notebook outputs are not used as evidence; use **Restart/Run All** to
refresh them in your local environment.

| Notebook | Execution | Output check | Requirements | Known issue |
| --- | --- | --- | --- | --- |
| [AntMaze](test_antmaze.ipynb) | Pass | Flow pass; policy quality not evaluated | MuJoCo, OGBench, Stable-Baselines3 | Full training is intentionally outside this quick example |
| [CartPole](test_cartpole.ipynb) | Pass | Pass (qualitative comparison) | Gymnasium classic-control, Matplotlib | — |
| [CartPole visual](test_cartpole_visual.ipynb) | Pass | Visual behavior reasonable | Gymnasium classic-control, Matplotlib | — |
| [DTR](test_dtr.ipynb) | Pass | Pass (two-stage mechanics and qualitative results) | Core, NumPy, Matplotlib | No strict numeric target is stated |
| [FrozenLake](test_frozenlake.ipynb) | Pass | Visual/action behavior reasonable | Gymnasium toy-text, pygame | — |
| [Highway](test_highway.ipynb) | Pass | Visual behavior reasonable; GIF generated | highway-env, pygame, Pillow | — |
| [Highway single-step](test_highway_single_step.ipynb) | Pass | State/action behavior reasonable | highway-env, Matplotlib | — |
| [Lava](test_lava.ipynb) | Pass | Visual transitions reasonable | MiniGrid, pygame, Matplotlib | — |
| [LunarLander](test_lunar_lander.ipynb) | Pass | Episodes advance and animations render | Box2D, pygame, Matplotlib | — |
| [MAB (Chapter 7)](<test_mab (Ch 7).ipynb>) | Pass | **Incorrect** | NumPy, Matplotlib, pandas, NetworkX | Public `MABSCM` reward mechanism differs from the textbook structural equation; deferred to a semantic branch |
| [Masked Atari](test_masked_atari.ipynb) | **Blocked** | Not checked | ALE and a legally supplied Pong ROM | `pong.bin` is not installed |
| [MDP (Chapter 7)](<test_mdp (Ch 7).ipynb>) | Pass | Pass (numeric, tolerance-based) | NumPy, Matplotlib, pandas, NetworkX | `MDPSCM` returns NumPy booleans; the notebook casts before array indexing |
| [MNIST](test_mnist.ipynb) | Pass | Pass (mechanics; no strict numeric target) | PyTorch, torchvision, MNIST data, Matplotlib | First use may need a dataset download |
| [MuJoCo random-friction Ant](test_mujoco_random_friction_ant.ipynb) | Pass | Render/rollout mechanics reasonable | MuJoCo, Matplotlib | Full algorithm performance is not evaluated |
| [Race](test_race.ipynb) | Pass | Visual behavior reasonable; HTML animation generated | highway-env, pygame, Matplotlib | Upstream `racetrack-v0` emits a version deprecation warning |
| [Windy MiniGrid](test_windyminigrid.ipynb) | Pass | Visual transitions reasonable | MiniGrid, pygame, Matplotlib | Uses `tostring_rgb`, supported by the pinned Matplotlib version but deprecated upstream |

Detailed evidence and deferred issues are recorded in
[`reports/maintenance/notebook_example_audit_2026-07-25.md`](../reports/maintenance/notebook_example_audit_2026-07-25.md).

Each entry below names the environment, points to the source module, lists the primary classes, states the Gymnasium registration (when available), and summarises the causal twist it introduces. Environments are grouped by the type of simulator or abstraction they target.

### Classic Control & MuJoCo
- **CartPoleWind** (`causal_gym/envs/cartpole_wind.py`)
  - Classes: `CartPoleWindSCM`, `CartPoleWindPCH`
  - Registration: `causal_gym/CartPoleWind-v0`
  - Description: standard cartpole with a per-step wind latent; info dict exposes natural actions for counterfactual comparisons.
  - Examples: `examples/test_cartpole.ipynb`, `examples/test_cartpole_visual.ipynb`

- **LunarLander Wind Field** (`causal_gym/envs/lunar_lander.py`)
  - Classes: `LunarLanderSCM`, `LunarLanderPCH`
  - Registration: `causal_gym/LunarLanderPCH-v0`
  - Description: samples a spatial wind map each episode and applies forces within Box2D; rendering overlays wind and natural actions.
  - Examples: `examples/test_lunar_lander.ipynb`, `examples/test_lander.py`

- **Random Friction Ant** (`causal_gym/envs/random_friction_ant.py`)
  - Classes: `RandomFrictionAntMujocoSCM`, `RandomFrictionAntMujocoPCH`
  - Registration: `causal_gym/RandomFrictionAntPCH-v0`
  - Description: randomises MuJoCo geom friction sets at reset; optionally concatenates sampled frictions to observations for identification studies.
  - Examples: `examples/test_mujoco_random_friction_ant.ipynb`

- **Random Mass Hopper Wrapper** (`causal_gym/envs/random_mass_hopper.py`)
  - Classes: `MassHopper` (Gymnasium wrapper)
  - Registration: helper only (not registered).
  - Description: utility wrapper that resamples Hopper body masses and optionally reveals them; combine with SCMs to induce latent dynamics shifts.
  - Examples: _no dedicated notebook yet_

- **AdroitHandDoor** (`causal_gym/envs/adroit_hand_door.py`)
  - Classes: `AdroitHandDoorSCM`, `AdroitHandDoorPCH`
  - Registration: not pre-registered; instantiate `AdroitHandDoorPCH` directly.
  - Description: wraps `AdroitHandDoor-v1` from `gymnasium_robotics`, providing causal hooks around a dexterous manipulation task (graph structure placeholder).
  - Examples: _no dedicated notebook yet_

### Highway & Driving
- **Highway Sequential Driving** (`causal_gym/envs/highway.py`)
  - Classes: `HighwaySCM`, `HighwayPCH`
  - Registration: not pre-registered.
  - Description: builds on `highway-env` with latent fog, lane misperception, and dashboard cues; pygame overlays highlight latent vs. observed factors.
  - Examples: `examples/test_highway.ipynb`

- **Highway Single-Step** (`causal_gym/envs/highway_single_step.py`)
  - Classes: `HighwaySingleStepSCM`, `HighwaySingleStepPCH`
  - Registration: not pre-registered.
  - Description: single decision about braking/accelerating under latent tail-light signals and weather; great for partial observability and counterfactual demos.
  - Examples: `examples/test_highway_single_step.ipynb`

<!-- - **Highway MDP Variant** (`causal_gym/envs/highway_mdp.py`)
  - Classes: `HighwayMDPSCM`, `HighwayMDPPCH`
  - Registration: not pre-registered.
  - Description: multi-step highway scenario with explicit logging of latent variables and confounded rewards; richer sequential structure for RL baselines.
  - Examples: _no dedicated notebook yet_ -->

- **Race Track Driving** (`causal_gym/envs/race.py`)
  - Classes: `RaceSCM`, `RacePCH`
  - Registration: not pre-registered.
  - Description: extends `racetrack-v0` with latent driver impairment, fog, and dashboard warnings; rewards promote lane-centred, safe driving.
  - Examples: `examples/test_race.ipynb`

### Minigrid
Turn any grid world into a windy one!
- **Custom LavaCrossing** (`causal_gym/envs/lava_minigrid.py`)
  - Classes: `CustomCrossingEnv`
  - Registration: `Custom-LavaCrossing-{easy,hard,extreme,maze,maze-complex}-v0`
  - Description: Minigrid layouts with lava corridors, optional coins, and wind distributions from `wind_dist.py`; suited for navigation with safety constraints.
  - Examples: `examples/test_lava.ipynb`

- **Windy MiniGrid** (`causal_gym/envs/windy_minigrid.py`)
  - Classes: `WindyMiniGridSCM`, `WindyMiniGridPCH`
  - Registration: `causal_gym/WindyGridWorld-v0`
  - Description: wraps MiniGrid environments with location-dependent winds and optional icon overlays. This enables you to modify **any** MiniGrid environment to be a windy one! Simply follow the similar set up in `examples/test_lava.ipynb` to start building your customized windy MiniGrid.
  - Examples: `examples/test_windyminigrid.ipynb`

### Tabular & Structured Causal Examples
- **Multi-Armed Bandit (Chapter 7)** (`causal_gym/envs/mab.py`)
  - Classes: `MABSCM`, `MABPCH`
  - Registration: not pre-registered.
  - Description: reproduce the two-arm bandit with optional continuous confounding from the textbook; highlights differences between logged observational data and `do` interventions.
  - Examples: `examples/test_mab (Ch 7).ipynb`

- **MDP Example (Chapter 7)** (`causal_gym/envs/mdp.py`)
  - Classes: `MDPSCM`, `MDPPCH`
  - Registration: `causal_gym/MDPExample-v0`
  - Description: confounded binary MDP from the Causal AI textbook with three exogenous variables driving transitions and rewards.
  - Examples: `examples/test_mdp (Ch 7).ipynb`

- **Dynamic Treatment Regime (DTR) (Chapter 8)** (`causal_gym/envs/dtr.py`)
  - Classes: `DTRSCM`, `DTRPCH`
  - Registration: not pre-registered.
  - Description: reproduce exmaple 8.1 in the book, a two-stage medical decision process with latent confounders; ideal for staged interventions and policy evaluation when ignorability is violated.
  - Examples: `examples/test_dtr.ipynb`

- **WhereDo Example (Chapter 9)** (`causal_gym/envs/wheredo_example.py`)
  - Classes: `ExampleSCM_9_5`, `ExamplePCH_9_5`
  - Registration: not pre-registered.
  - Description: implements the instrument-variable example 9.5 in the book with paired actions (`X1`, `X2`) in a single step, useful for counterfactual consistency exercises.
  - Examples: _no dedicated notebook yet_

- **Robot Walk** (`causal_gym/envs/robowalk.py`)
  - Classes: `RobotWalkSCM`, `RobotWalkPCH`
  - Registration: `causal_gym/RobotWalk-v0`
  - Description: 1-D hallway traversal with stability latents and confounded transitions; includes `PolicyMapping` helper for visualising learned policies.
  - Examples: _no dedicated notebook yet_

- **FrozenLake Wind Map** (`causal_gym/envs/frozen_lake.py`)
  - Classes: `FrozenLakeSCM`, `FrozenLakePCH`
  - Registration: `causal_gym/FrozenLakePCH-v0`
  - Description: adds per-cell wind directions, and enhanced rendering to `FrozenLake-v1`; exposes generated wind map via the info dict.
  - Examples: `examples/test_frozenlake.ipynb`, `examples/test_frozenlake.py`

### Atari
- **Masked Atari** (`causal_gym/envs/masked_atari.py`)
  - Classes: `MaskedAtariSCM`, `MaskedAtariPCH`
  - Registration: `causal_gym/Masked{EnvName}-v0`
  - Description: programmatically masks sections of Atari frames to emulate missing information; compares policies using masked vs. full observations.
  - Supported games: Pong, Amidar, Asterix, Boxing, Breakout, ChopperCommand, Gopher, KungFuMaster, MsPacman, Qbert, RoadRunner, Seaquest
  - Examples: `examples/test_masked_atari.ipynb`

<!-- ### Vision
- **MNIST Causal Classifier** (`causal_gym/envs/mnist.py`)
  - Classes: `MNISTSCM`, `MNISTPCH`
  - Registration: not pre-registered.
  - Description: generates digit observations conditioned on treatment and latent patient type; models perception confounding in a simple binary decision setting.
  - Examples: `examples/test_mnist.ipynb` -->
