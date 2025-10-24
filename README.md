# CausalGym: Causal Reinforcement Learning Playground

CausalGym is a Gymnasium-compatible suite for experimenting with structural causal models in causal reinforcement learning. Each environment exposes both the usual step/reset loop and the Pearl Causal Hierarchy (PCH) interface so you can collect observational rollouts, perform interventions, and query counterfactuals inside a single simulation.

## Highlights
- Builds causal RL simulators on top of Gymnasium, highway-env, Minigrid, ALE, MuJoCo, and custom tabular domains.
- Provides unified `SCM` and `PCH` abstractions so behaviour policies (`see`), interventions (`do`), and counterfactual queries (`ctf_do`) share the same environment instance.
- Ships with causal graphs, exogeneous variable sampling hooks, and reusable wrappers to support strucutred causal reasoning.
- Bundles ready-to-run notebooks and scripts illustrating causal RL workflows across grid worlds, classic control, Atari games, driving simulators, and high-dimensional tasks.

## Core Causal Concepts Explained
![CausalGym repository overview](examples/repo.png)

**Environment - Structural Causal Models (SCMs)** – Each environment subclasses `causal_gym.core.SCM`, which you can think of as a Gymnasium `Env` augmented with structural equations. An SCM specifies:
- *Endogenous variables* (state, action, reward, next state, perception, etc.) updated step-by-step via deterministic functions.
- *Exogenous variables* sampled through `sample_u()`. These latents inject stochasticity into the system (wind gusts, random friction, etc.) and are the levers behind counterfactual reasoning.
- *Graph structure* returned by `get_graph`, capturing how the variables causally influence one another. The graph is more than documentation. It grounds the rules that make interventions and counterfactuals well-defined.

From an RL researcher’s perspective, an SCM decomposes the usual Markovian transition `p(s', r | s, a)` in an MDP into a set of assignments that reveal which randomness is policy-dependent and which comes from the environment (Note that we also support general decision making problems that are NOT MDPs!). By exposing the latents and graph, CausalGym expands RL learning modalities to all three levels of PCH: collecting purely observational data, actively perturbing the system, and querying counterfactuals.

**Pearl’s Causal Hierarchy (PCH)** – The companion `PCH` wrapper is the control panel for navigating the three interaction modalities with an SCM:
- *Level 1 – Observing (`see`)*: The environment evolves under its built-in behaviour policy. Calling `see()` corresponds to passively logging trajectories; the info dict records the “natural action” that was taken.
- *Level 2 – Intervening (`do`)*: The `do` operator breaks incoming edges into the action node. You supply a `do_policy` (mapping observation → action) that samples an action w.r.t the input observation. This is similar to `step()` interface in the standard Gymnasium setup.
- *Level 3 – Counterfactual (`ctf_do`)*: Counterfactuals ask “what would have happened under a different action *given* what we just observed?” The wrapper first lets the behaviour policy act, then calls your `ctf_policy(observation, natural_action)` so you can deviate from the natural choice while keeping the sampled exogenous variables fixed.

**Bridge to Potential Outcomes (PO)** – If you are used to notation such as \(Y_x\) or \(Y_{x'} - Y_x\):
- `see()` produces observational data \( (U, X, Y) \) with the action determined by the behaviour policy. Such data is collected passively and is close to concepts like standard logged bandit feedback in the literature.
- `do(policy)` realises \(Y_{x}\) by forcing `X` to whatever the policy outputs. Different calls with different forced actions mimic Randomized Controlled Trials (RCT).
- `ctf_do` retrieves a single-world intervention graph (SWIG) view: the natural action gives you the “factual” world, and the counterfactual policy lets you evaluate contrasts such as \(Y_{x'} - Y_x\) conditioned on the realised latents. Because the SCM keeps track of the sampled `U`, two consecutive calls within the same episode remain coupled in the potential-outcome sense.

**Tasks and permissions** – The `Task` object configures which levels you are allowed to access (`LearningRegime.see`, `do`, `ctf_do`, or mixtures like `see_do`). This gives you a precise way to state assumptions: for example, an off-policy evaluation project can restrict itself to `see`, whereas algorithmic recourse or counterfactual policy evaluation requires `ctf_do`.

For a deeper dive into the underlying theory, see the [Causal Artificial Intelligence](https://causalai-book.net/) textbook from our [Causal AI Lab](https://causalai.net). The README sections below reference these concepts when describing each environment and its causal affordances.

## Installation
```bash
pip install -e .
```
The editable install pulls in Gymnasium, highway-env, pygame, networkx and other dependencies defined in `setup.py`. Some environments download additional assets on first use (e.g. `MNISTSCM` fetches the MNIST dataset via `torchvision` and the Atari wrapper requires ALE ROMs).

## Quickstart: observational, interventional, and counterfactual rollouts
```python
from causal_gym.envs import CartPoleWindPCH
from causal_gym.core.task import Task, LearningRegime

# Enable all interaction modes (see + do + ctf_do)
# ctf_do regime has access to all modalities
task = Task(learning_regime=LearningRegime.ctf_do)

# Control the pole when there is a wind!
env = CartPoleWindPCH(task=task, wind_std=0.05)
obs, info = env.reset(seed=7)

# 1) Observe the behaviour policy under natural dynamics
obs, reward, terminated, truncated, info = env.see()

# 2) Intervene on the action: always push right
obs, reward, terminated, truncated, info = env.do(lambda observation: 1)

# 3) Counterfactual: act with respect to both the observation and the natural action
def counterfactual_policy(observation, natural_action):
    return 0 if natural_action == 1 else natural_action

obs, reward, terminated, truncated, info = env.ctf_do(counterfactual_policy)

# Access the causal graph if you need it for non-parametric analysis or visualisation
graph = env.env.get_graph

env.close()
```
You can also rely on Gymnasium's registry (`gymnasium.make("causal_gym/CartPoleWind-v0")`) when you only need the PCH wrapper but still want compatibility with RL baselines.

## Examples and testing aids
The `examples/` directory contains interactive notebooks (`test_cartpole.ipynb`, `test_frozenlake.ipynb`, `test_highway.ipynb`, `test_mujoco_random_friction_ant.ipynb`, etc.) and lightweight scripts (`interactive_play.py`, `test_frozenlake.py`, `test_lander.py`) that demonstrate how to step through `SCM`/`PCH` APIs, render interventions, and benchmark policies. 

## Repository Structure
- `causal_gym/`
  - `core/`
    - `scm.py`: base class that augments Gymnasium envs with structural equations, latent sampling, and causal graphs.
    - `pch.py`: wrapper implementing Pearl’s hierarchy (`see`, `do`, `ctf_do`) and task-aware access control.
    - `task.py`: defines learning regimes, structural assumptions, and reward semantics used by the wrappers.
    - `graph*.py`, `object_utils.py`, `set_utils.py`: utilities for building and manipulating causal diagrams.
    - `wrappers/`: mixins that adjust observations, actions, or policies before they reach the underlying SCM/PCH.
  - `envs/`
    - `*.py`: domain-specific SCM/PCH implementations (CartPole, FrozenLake, Highway, Atari, MuJoCo, etc.) with optional assets.
    <!-- - `__init__.py`: registers Gymnasium IDs and wires SCM↔PCH pairs into the public namespace. -->
  <!-- - `__init__.py`: re-exports everything under `causal_gym` for `import causal_gym as cg`. -->
- `examples/`
  - notebooks and scripts (`test_*.ipynb`, `interactive_play.py`) demonstrating interventions, counterfactuals, and training loops.
- `setup.py`: packaging metadata and dependency pins.

## Environment Suite
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
