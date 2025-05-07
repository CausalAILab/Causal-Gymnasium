import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Ball, Wall, Lava, Goal
from minigrid.core.actions import Actions
from minigrid.core.constants import COLORS
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    point_in_circle,
    rotate_fn,
)
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS

from causal_gym.core import PolicyType, ActType, ObsType, SCM, PCH
from causal_gym.envs import WindyMiniGridSCM, WindyMiniGridPCH
from causal_gym.envs import MDPExampleSCM, MDPExamplePCH
from causal_gym.envs import DTRExampleSCM, DTRExamplePCH
from causal_gym.envs import MABExampleSCM, MABExamplePCH

# We would import this if it existed
# from causal_gym.envs import MABExampleSCM, MABExamplePCH

class ExtendedWindyGridSCM(WindyMiniGridSCM):
    """
    An extended WindyGrid environment that integrates with different decision models.
    
    This environment maintains the physics of WindyMiniGridSCM while allowing for
    integration with different decision models (MDP, MAB, DTR) from causal_gym.
    
    It maps the 4-directional grid navigation to the appropriate actions in each model,
    and handles the reward calculation and state transitions accordingly.
    
    The wind mechanics from the original WindyMiniGridSCM are preserved, maintaining
    the causal influence of wind on navigation.
    """
    
    def __init__(self, env: MiniGridEnv, decision_type='MDP', policy=None, show_wind=False, wind_dist=None, gamma=0.9):
        """
        Initialize the extended windy grid environment
        
        Args:
            env: Base MiniGrid environment
            decision_type: Type of decision model ('MDP', 'MAB', or 'DTR')
            policy: Behavioral policy for the environment
            show_wind: Whether to display wind direction in rendering
            wind_dist: Wind distribution function or tuple
            gamma: Discount factor for rewards
        """
        super().__init__(env, policy, show_wind, wind_dist)
        self.gamma = gamma
        self.decision_type = decision_type
        
        # Initialize the decision model based on type
        if decision_type == 'MDP':
            self.decision_model = MDPExampleSCM()
        elif decision_type == 'MAB':
            self.decision_model = MABExampleSCM()
        elif decision_type == 'DTR':
            # For DTR, we need to customize the state and reward functions
            self.decision_model = DTRExampleSCM(
                s1_function=lambda: self._map_grid_to_dtr_s1(),
                s2_function=lambda: self._map_grid_to_dtr_s2(),
                y_function=lambda: self._calculate_dtr_reward()
            )
            # Customize the DTR policies
            self.decision_model.change_policy1(lambda s1, u: self._dtr_policy1(s1))
            self.decision_model.change_policy2(lambda s1, x1, s2, u: self._dtr_policy2(s1, x1, s2))
        else:
            raise ValueError(f"Unknown decision type: {decision_type}")
        
        # Initialize DTR-specific tracking
        self.dtr_complete = False
        self.dtr_episode_reward = 0.0
        
        # Trajectory tracking for policy evaluation
        self.episode_history = []
        self.current_trajectory = []
        self.steps_in_episode = 0
        self.cumulative_reward = 0.0
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment and the decision model"""
        # Reset trajectory tracking
        if self.current_trajectory:
            self.episode_history.append(self.current_trajectory)
            self.current_trajectory = []
        
        self.steps_in_episode = 0
        self.cumulative_reward = 0.0
        
        # Reset DTR-specific tracking
        self.dtr_complete = False
        self.dtr_episode_reward = 0.0
        
        # Reset the decision model
        self.decision_model.reset(seed=seed)
        
        # Reset the grid environment
        obs, info = super().reset(seed=seed, options=options)
        
        # Record initial state
        self._record_state()
        
        return obs, info
    
    def _record_state(self):
        """Record the current state of the environment"""
        state = {
            'grid_pos': tuple(self.agent_pos),
            'grid_dir': self.agent_dir,
            'wind_dir': self.wind_dir,
            'steps': self.steps_in_episode,
            'decision_state': self._map_grid_to_decision_state(),
            'decision_type': self.decision_type,
            'cumulative_reward': self.cumulative_reward
        }
        
        # Add to current trajectory
        self.current_trajectory.append(state)
    
    def _map_grid_to_decision_state(self):
        """
        Map the grid world state to a state for the decision model
        
        Returns:
            decision_state: State representation for the decision model
        """
        if self.decision_type == 'MDP':
            # For MDP, use a combination of position and wind direction
            width = self._env.unwrapped.width
            pos = self.agent_pos
            return (pos[0] * width + pos[1]) * 5 + self.wind_dir
            
        elif self.decision_type == 'MAB':
            # MAB doesn't use state
            return None
            
        elif self.decision_type == 'DTR':
            # For DTR, this is handled by the custom functions
            if self.decision_model.stage == 0:
                return self._map_grid_to_dtr_s1()
            else:
                return self._map_grid_to_dtr_s2()
        
        # Default fallback
        return tuple(self.agent_pos)
    
    def _map_grid_to_dtr_s1(self):
        """Map grid state to DTR S1 (first stage state)"""
        # Use x-coordinate as S1 state
        return self.agent_pos[0] % 2
    
    def _map_grid_to_dtr_s2(self):
        """Map grid state to DTR S2 (second stage state)"""
        # Use y-coordinate as S2 state
        return self.agent_pos[1] % 2
    
    def _calculate_dtr_reward(self):
        """Calculate the reward for DTR based on grid state"""
        # Check if we've reached a goal or hit lava
        pos = self.agent_pos
        if isinstance(self.grid.get(*pos), Goal):
            return 1  # Success
        elif isinstance(self.grid.get(*pos), Lava):
            return 0  # Failure
        
        # Default: check if we're closer to goal
        goal_pos = None
        for i in range(self._env.unwrapped.width):
            for j in range(self._env.unwrapped.height):
                if isinstance(self.grid.get(i, j), Goal):
                    goal_pos = (i, j)
                    break
            if goal_pos:
                break
        
        if goal_pos:
            # Calculate Manhattan distance to goal
            distance = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
            
            # Scale distance to [0, 1] range
            max_distance = self._env.unwrapped.width + self._env.unwrapped.height
            normalized_distance = distance / max_distance
            
            # Invert so closer = higher reward
            return 1 - normalized_distance
        
        return 0.5  # Neutral reward
    
    def _dtr_policy1(self, s1):
        """Default first-stage policy for DTR"""
        # For the grid, we need 4 possible actions
        # Convert binary s1 to one of 4 actions
        if s1 == 0:
            return np.random.choice([0, 1])  # left or right
        else:
            return np.random.choice([2, 3])  # forward or stay
    
    def _dtr_policy2(self, s1, x1, s2):
        """Default second-stage policy for DTR"""
        # Try to make a different action than first stage
        if x1 <= 1:  # If first action was left/right
            return np.random.choice([2, 3])  # Do forward/stay
        else:
            return np.random.choice([0, 1])  # Do left/right
    
    def _map_decision_to_grid_action(self, decision_action):
        """
        Map a decision model action to a grid action
        
        Args:
            decision_action: Action from the decision model
            
        Returns:
            grid_action: Corresponding action in the grid world
        """
        # All models use 4 action space mapped to grid actions:
        # 0: Left
        # 1: Right
        # 2: Forward
        # 3: Stay (do nothing)
        mapping = {
            0: Actions.left,
            1: Actions.right,
            2: Actions.forward,
            3: Actions.done
        }
        
        return mapping.get(decision_action % 4, Actions.done)
    
    def action(self):
        """
        Sample action from the behavioral policy
        
        Returns:
            grid_action: Action to take in the grid environment
        """
        # If custom policy is provided, use it
        if self._policy is not None:
            grid_action = self._policy(self._internal_state, self.wind_dir)
            return grid_action
        
        # Otherwise, get action from decision model
        if self.decision_type == 'MDP':
            decision_state = self._map_grid_to_decision_state()
            decision_action = self.decision_model.action(decision_state)
            
        elif self.decision_type == 'MAB':
            decision_action = self.decision_model.action()
            
        elif self.decision_type == 'DTR':
            if self.dtr_complete:
                # If DTR episode is complete, reset it
                self.decision_model.reset()
                self.dtr_complete = False
                self.dtr_episode_reward = 0.0
            
            # Get action from DTR model
            decision_action = self.decision_model.action()
            
        else:
            # Default to random action
            decision_action = np.random.randint(0, 4)
        
        # Map to grid action
        grid_action = self._map_decision_to_grid_action(decision_action)
        
        return grid_action
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Grid action to take
            
        Returns:
            next_state, reward, terminated, truncated, info: Step results
        """
        # Increment step counter
        self.steps_in_episode += 1
        
        # Take step in grid environment
        next_state, grid_reward, terminated, truncated, info = super().step(action)
        
        # Map the grid action to decision action
        decision_action = self._map_grid_to_decision_action(action)
        
        # Take step in decision model
        if self.decision_type == 'MDP':
            _, decision_reward, _, _, _ = self.decision_model.step(decision_action)
            
        elif self.decision_type == 'MAB':
            _, decision_reward, _, _, _ = self.decision_model.step(decision_action)
            
        elif self.decision_type == 'DTR':
            if not self.dtr_complete:
                try:
                    _, decision_reward, dtr_terminated, _, dtr_info = self.decision_model.step(decision_action)
                    if dtr_terminated:
                        self.dtr_complete = True
                        self.dtr_episode_reward = decision_reward
                except ValueError:
                    # DTR might be already complete
                    self.decision_model.reset()
                    self.dtr_complete = False
                    decision_reward = 0.0
            else:
                # DTR is complete, use the stored reward
                decision_reward = self.dtr_episode_reward

        # Check if decision_reward is a function and call it if so
        if callable(decision_reward):
            decision_reward = decision_reward()
        # print('grid_reward ', grid_reward)
        # print('decision reward ', decision_reward)
        # Combine grid and decision rewards
        combined_reward = grid_reward + decision_reward
        
        # Update cumulative reward
        self.cumulative_reward += combined_reward
        
        # Record transition
        self._record_transition(action, decision_action, next_state, grid_reward, decision_reward, combined_reward, terminated, truncated, info)
        
        return next_state, combined_reward, terminated, truncated, info
    
    def _map_grid_to_decision_action(self, grid_action):
        """
        Map a grid action to a decision model action
        
        Args:
            grid_action: Action in the grid world
            
        Returns:
            decision_action: Corresponding action in the decision model
        """
        # Map grid actions to 4 decision actions
        if grid_action == Actions.left:
            return 0
        elif grid_action == Actions.right:
            return 1
        elif grid_action == Actions.forward:
            return 2
        else:
            return 3
    
    def _record_transition(self, grid_action, decision_action, next_state, grid_reward, decision_reward, combined_reward, terminated, truncated, info):
        """
        Record a transition in the current trajectory
        
        Args:
            grid_action: Action in the grid world
            decision_action: Action in the decision model
            next_state: Resulting state
            grid_reward: Reward from the grid environment
            decision_reward: Reward from the decision model
            combined_reward: Combined reward
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Additional information
        """
        transition = {
            'grid_pos': tuple(self.agent_pos),
            'grid_dir': self.agent_dir,
            'wind_dir': self.wind_dir,
            'grid_action': grid_action,
            'decision_action': decision_action,
            'grid_reward': grid_reward,
            'decision_reward': decision_reward,
            'combined_reward': combined_reward,
            'terminated': terminated,
            'truncated': truncated,
            'steps': self.steps_in_episode,
            'decision_state': self._map_grid_to_decision_state(),
            'decision_type': self.decision_type,
            'cumulative_reward': self.cumulative_reward
        }
        
        # Add DTR-specific info
        if self.decision_type == 'DTR':
            transition['dtr_stage'] = self.decision_model.stage
            transition['dtr_complete'] = self.dtr_complete
            if hasattr(self.decision_model, 's1'):
                transition['dtr_s1'] = self.decision_model.s1
            if hasattr(self.decision_model, 'x1'):
                transition['dtr_x1'] = self.decision_model.x1
            if hasattr(self.decision_model, 's2'):
                transition['dtr_s2'] = self.decision_model.s2
        
        # Add to current trajectory
        self.current_trajectory.append(transition)


class ExtendedWindyGridPCH(WindyMiniGridPCH):
    """
    PCH wrapper for ExtendedWindyGridSCM that enables policy evaluation methods
    
    This wrapper provides methods for:
    - Observing the environment (see)
    - Intervening in the environment (do)
    - Policy evaluation using IPW, DP, and RCT methods
    
    It integrates with different decision models (MDP, MAB, DTR) and adapts
    the policy evaluation methods accordingly.
    """
    
    def __init__(self, env: MiniGridEnv, decision_type='MDP', policy=None, show_wind=False, wind_dist=None, gamma=0.9):
        """
        Args:
            env: Base MiniGrid environment
            decision_type: Type of decision model ('MDP', 'MAB', or 'DTR')
            policy: Behavioral policy for the environment
            show_wind: Whether to display wind direction in rendering
            wind_dist: Wind distribution function or tuple
            gamma: Discount factor for rewards
        """
        # Create the SCM
        self.env = ExtendedWindyGridSCM(env, decision_type, policy, show_wind, wind_dist, gamma)
        
        # Initialize state space from the underlying environment
        self.state_space = [env.unwrapped.width, env.unwrapped.height]
        
        # Initialize PCH
        PCH.__init__(self)
        
        # Store gamma for reward discounting
        self.gamma = gamma
        self.decision_type = decision_type
    
    def see(self, bpolicy=None):
        """
        Args:
            bpolicy: Optional custom behavioral policy
        Returns:
            action, next_state, reward, terminated, truncated, info
        """
        # Store original policy
        original_policy = None
        if bpolicy is not None:
            original_policy = self.env._policy
            self.env._policy = bpolicy
        
        try:
            # Get action from behavioral policy
            action = self.env.action()
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            return action, next_state, reward, terminated, truncated, info
        
        finally:
            # Restore original policy
            if original_policy is not None:
                self.env._policy = original_policy
    
    def do(self, action):
        """
        Returns:
            next_state, reward, terminated, truncated, info
        """
        return self.env.step(action)
    
    def collect_trajectories(self, num_episodes, behavior_policy=None):
        """
        Collect trajectories using the behavioral policy
        
        Args:
            num_episodes: Number of episodes to collect
            behavior_policy: Optional custom behavioral policy
            
        Returns:
            trajectories: List of collected trajectories
        """
        # Clear existing history
        self.env.episode_history = []
        
        for _ in range(num_episodes):
            self.env.reset()
            done = False
            
            while not done:
                # Take step using see()
                action, next_state, reward, terminated, truncated, info = self.see(bpolicy=behavior_policy)
                done = terminated or truncated
        
        return self.env.episode_history
    
