### frozenlake.py
import gymnasium as gym
import numpy as np
import pygame
import os

# Attempt relative imports for normal package use
try:
    from ..core import SCM, PCH
    from ..core.types import PolicyType, ObsType, ActType
except ImportError:
    # Fallback for direct script execution
    import sys
    # Construct the absolute path to the 'causal2' directory (or project root)
    # Assuming this script is in 'causal2/causalgym/causal_gym/envs/'
    # Path needs to go up three levels to reach 'causal2'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Now attempt the import again, this time it should work if causalgym is in project_root
    from causal_gym.core import SCM, PCH
    from causal_gym.core.types import PolicyType, ObsType, ActType


# Define some colors for rendering (RGB)
COLOR_BG = np.array([128, 128, 128]) 
COLOR_FROZEN = np.array([230, 230, 230]) # Lighter gray for ice
COLOR_HOLE = np.array([0, 0, 0])      
COLOR_GOAL = np.array([0, 200, 0])    # Darker Green
COLOR_START = np.array([200, 200, 0]) # Yellow for Start
COLOR_AGENT = np.array([200, 0, 0])   # Red for Agent

TILE_SIZE = 32 # Global tile size in pixels, used for rgb_array rendering with sprites

# Define wind directions
WIND_NONE = 0
WIND_NORTH = 1
WIND_EAST = 2
WIND_SOUTH = 3
WIND_WEST = 4
WIND_DIRECTIONS = [WIND_NONE, WIND_NORTH, WIND_EAST, WIND_SOUTH, WIND_WEST]

class FrozenLakeSCM(SCM[PolicyType, ObsType, ActType]):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(self,
                 seed: int = 0,
                 map_name: str = "4x4",
                 is_slippery: bool = True,
                 wind_probabilities: tuple[float, float, float, float, float] = (0.7, 0.075, 0.075, 0.075, 0.075), # WIND_NONE, N, E, S, W
                 render_mode: str | None = None):
        super().__init__()
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.wind_probabilities = wind_probabilities
        assert sum(wind_probabilities) == 1.0, "Wind probabilities must sum to 1."
        assert len(wind_probabilities) == 5, "Wind probabilities must have 5 elements."

        # Base Gymnasium environment (primarily for P, desc, action/observation spaces)
        # We set its render_mode to "rgb_array" to avoid it trying to create its own Pygame window
        # if our SCM is in "human" mode. Our SCM will manage its own Pygame window if needed.
        self.env = gym.make(f"FrozenLake-v1", desc=None, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array")
        
        self.nrow, self.ncol = self.env.unwrapped.nrow, self.env.unwrapped.ncol
        self.initial_state_distrib = self.env.unwrapped.initial_state_distrib
        self.P = self.env.unwrapped.P
        self.desc = self.env.unwrapped.desc # Ensure desc is set early

        self.observation_space = self.env.unwrapped.observation_space
        self.action_space = self.env.unwrapped.action_space
        # self.reward_range = self.env.unwrapped.reward_range # This will change due to shaping
        # Max Manhattan distance for reward shaping
        self.max_manhattan_dist = (self.nrow - 1) + (self.ncol - 1)
        if self.max_manhattan_dist == 0 and self.nrow == 1 and self.ncol == 1: # For 1x1 grid, ensure it's not treated as 0 if not G/H
             # If it's G, reward is 1. If H, -1. If S (which is G), 1.
             # This is mostly a conceptual edge case, FrozenLake maps are larger.
             # Setting it to 1 just to avoid potential division by zero if a 1x1 'F' map was made.
            self.max_manhattan_dist = 1

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.cell_size = 50  # Cell size for "human" mode rendering (can be different from TILE_SIZE)
        
        # Ensure Pygame modules are initialized if any rendering will occur.
        # image.load() might need display module to be initialized even for rgb_array.
        if self.render_mode in ["human", "rgb_array"]:
            if not pygame.get_init():
                print("[DEBUG] Initializing Pygame...")
                pygame.init() # General Pygame init
            if not pygame.display.get_init(): # Ensure display module is also initialized
                print("[DEBUG] Initializing Pygame display module...")
                pygame.display.init()
            
            if self.render_mode == "human": # Specific display.set_mode for human mode
                self.window_size = (self.ncol * self.cell_size, self.nrow * self.cell_size)
                self.window_surface = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption(f"FrozenLake - {self.map_name}")
                self.clock = pygame.time.Clock()
        
        self.wind_map = np.zeros((self.nrow, self.ncol), dtype=int)
        self.agent_pos = None

        # Wind indicator configurations
        self.wind_indicator_size_factor = 0.55 # Increased from 0.4 to make arrows larger
        self.wind_indicator_margin_factor = 0.05 
        self.wind_indicator_color = (0, 0, 255, 220) # Blue, semi-transparent for arrows      
        self.no_wind_indicator_color = (0, 0, 255, 220) # Changed to blue, same as arrows
        self.no_wind_symbol_radius_factor = 0.25 # Reduced from 0.35 for a smaller circle
        self.wind_arrow_thickness_factor = 0.1 # Reverted from 0.18 for thinner, potentially clearer arrows
        self.wind_arrow_head_size_factor = 0.25 
        
        # Find goal position
        self.goal_pos_rc = None
        for r in range(self.nrow):
            for c in range(self.ncol):
                cell_char = self.desc[r, c].item()
                if isinstance(cell_char, bytes): cell_char = cell_char.decode('utf-8')
                if cell_char == 'G':
                    self.goal_pos_rc = (r, c)
                    break
            if self.goal_pos_rc:
                break
        if self.goal_pos_rc is None:
            raise ValueError("Goal 'G' not found in map description.")

        self.sample_u()

    def _to_rc(self, s_idx: int) -> tuple[int, int]:
        """Converts a 1D state index to 2D row, col coordinates."""
        return s_idx // self.ncol, s_idx % self.ncol

    def _manhattan_distance(self, pos1_rc: tuple[int, int], pos2_rc: tuple[int, int]) -> int:
        """Calculates Manhattan distance between two (row, col) positions."""
        return abs(pos1_rc[0] - pos2_rc[0]) + abs(pos1_rc[1] - pos2_rc[1])

    def sample_u(self):
        self.wind_map = np.full((self.nrow, self.ncol), WIND_NONE, dtype=int)
        for r in range(self.nrow):
            for c in range(self.ncol):
                cell_type = self.desc[r, c].item()
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode('utf-8')
                
                if cell_type in ['F', 'S']:
                    self.wind_map[r, c] = self.np_random.choice(
                        WIND_DIRECTIONS,
                        p=self.wind_probabilities
                    )
        return {"u_wind_map": self.wind_map.copy()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) 
        obs, info = self.env.reset(seed=self._np_random_seed) 
        
        u_info = self.sample_u() 
        info['wind_map_generated'] = True 
        agent_r, agent_c = self._to_rc(obs)
        info['initial_agent_cell_wind'] = self.wind_map[agent_r, agent_c].item() if self.wind_map is not None else WIND_NONE
        
        self.agent_pos = obs
        
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        transitions = self.P[self.agent_pos][action]
        
        current_row, current_col = self._to_rc(self.agent_pos)
        wind_in_cell = self.wind_map[current_row, current_col]

        intended_action = action
        if wind_in_cell != WIND_NONE:
            if not self.is_slippery:
                if wind_in_cell == WIND_NORTH: intended_action = 3
                elif wind_in_cell == WIND_EAST: intended_action = 2
                elif wind_in_cell == WIND_SOUTH: intended_action = 1
                elif wind_in_cell == WIND_WEST: intended_action = 0
            else: 
                pass 

        if not self.is_slippery and wind_in_cell != WIND_NONE:
             actual_transitions = self.P[self.agent_pos][intended_action]
        else: 
             actual_transitions = self.P[self.agent_pos][action]

        i = self.np_random.choice(len(actual_transitions), p=[t[0] for t in actual_transitions])
        p_prob, s_next, r_original, terminated = actual_transitions[i]
        
        # current_pos_rc = self._to_rc(self.agent_pos) # Not needed for new reward logic
        next_pos_rc = self._to_rc(s_next)
        
        final_reward = 0.0
        # Determine character of the next state for reward calculation
        next_state_char_bytes = self.desc[next_pos_rc] # It's a numpy array of bytes
        next_state_char = next_state_char_bytes.item().decode('utf-8') if isinstance(next_state_char_bytes.item(), bytes) else next_state_char_bytes.item()

        if next_state_char == 'G':
            final_reward = 1.0
            # Gymnasium's FrozenLake-v1 sets r_original=1 and terminated=True for Goal
        elif next_state_char == 'H':
            final_reward = -1.0
            # Gymnasium's FrozenLake-v1 sets r_original=0 and terminated=True for Hole
        else:  # 'S' or 'F'
            dist_to_goal = self._manhattan_distance(next_pos_rc, self.goal_pos_rc)
            if self.max_manhattan_dist > 0:
                # Reward is higher for smaller distances (closer to 1), normalized.
                # Example: (6-0)/6 = 1 (at goal, but handled above), (6-1)/6 = 0.83, (6-6)/6 = 0 (furthest)
                final_reward = (self.max_manhattan_dist - dist_to_goal) / self.max_manhattan_dist
            else: # Should ideally not happen for valid FrozenLake maps (e.g. 1x1 'F' map)
                final_reward = 0.0 
            # terminated remains False for S or F states unless set by PCH logic later

        self.agent_pos = s_next
        self.lastaction = action 

        truncated = False 
        info = {"prob": p_prob, "action_was": action}
        if not self.is_slippery and wind_in_cell != WIND_NONE:
            info["wind_overrode_action_to"] = intended_action
        info["wind_in_cell"] = wind_in_cell
        info["agent_pos_rc"] = self._to_rc(self.agent_pos)
        info["original_reward"] = r_original # Keep for debugging if needed

        if self.render_mode == "human":
            self.render(mode="human")

        return (int(s_next), final_reward, terminated, truncated, info)

    def action(self):
        return self.np_random.integers(self.action_space.n)

    def observation(self):
        raise NotImplementedError

    @staticmethod
    def _render_individual_wind_arrow(wind_direction: int, tile_s: int) -> pygame.Surface | None:
        # This method remains, as it uses pygame.draw and is independent of sprites
        if tile_s < 10: tile_s = 10
        arrow_canvas_size = tile_s
        arrow_surface = pygame.Surface((arrow_canvas_size, arrow_canvas_size), pygame.SRCALPHA)
        arrow_surface.fill((0,0,0,0)) 

        center_x = arrow_canvas_size // 2
        center_y = arrow_canvas_size // 2
        
        shaft_length_half = arrow_canvas_size // 4
        head_width_half = arrow_canvas_size // 5
        head_length = arrow_canvas_size // 4 

        arrow_color = (0, 0, 200, 255) # Dark Blue arrow, fully opaque

        if wind_direction == WIND_NONE:
            radius = max(1, arrow_canvas_size // 8)
            pygame.draw.circle(arrow_surface, arrow_color, (center_x, center_y), radius)
            return arrow_surface

        # Arrow drawing logic (North, East, South, West) ...
        if wind_direction == WIND_NORTH:
            start_shaft = (center_x, center_y + shaft_length_half)
            end_shaft = (center_x, center_y - shaft_length_half)
            pygame.draw.line(arrow_surface, arrow_color, start_shaft, end_shaft, 1)
            pts = [
                (center_x, end_shaft[1] - head_length), 
                (center_x - head_width_half, end_shaft[1]), 
                (center_x + head_width_half, end_shaft[1])  
            ]
            pygame.draw.polygon(arrow_surface, arrow_color, pts)
        elif wind_direction == WIND_EAST:
            start_shaft = (center_x - shaft_length_half, center_y)
            end_shaft = (center_x + shaft_length_half, center_y)
            pygame.draw.line(arrow_surface, arrow_color, start_shaft, end_shaft, 1)
            pts = [
                (end_shaft[0] + head_length, center_y), 
                (end_shaft[0], center_y - head_width_half), 
                (end_shaft[0], center_y + head_width_half)  
            ]
            pygame.draw.polygon(arrow_surface, arrow_color, pts)
        elif wind_direction == WIND_SOUTH:
            start_shaft = (center_x, center_y - shaft_length_half)
            end_shaft = (center_x, center_y + shaft_length_half)
            pygame.draw.line(arrow_surface, arrow_color, start_shaft, end_shaft, 1)
            pts = [
                (center_x, end_shaft[1] + head_length), 
                (center_x - head_width_half, end_shaft[1]), 
                (center_x + head_width_half, end_shaft[1])  
            ]
            pygame.draw.polygon(arrow_surface, arrow_color, pts)
        elif wind_direction == WIND_WEST:
            start_shaft = (center_x + shaft_length_half, center_y)
            end_shaft = (center_x - shaft_length_half, center_y)
            pygame.draw.line(arrow_surface, arrow_color, start_shaft, end_shaft, 1)
            pts = [
                (end_shaft[0] - head_length, center_y), 
                (end_shaft[0], center_y - head_width_half), 
                (end_shaft[0], center_y + head_width_half)  
            ]
            pygame.draw.polygon(arrow_surface, arrow_color, pts)
        else: # Should not happen if WIND_NONE is handled
            return None 
        
        return arrow_surface

    def _render_wind_indicator(self, wind_direction_in_cell: int, indicator_size: int):
        indicator_surface = pygame.Surface((indicator_size, indicator_size), pygame.SRCALPHA)
        indicator_surface.fill((0, 0, 0, 0))  # Fully transparent background

        center_x, center_y = indicator_size // 2, indicator_size // 2
        
        # Use factors for dynamic sizing based on indicator_size
        arrow_length = int(indicator_size * 0.35) 
        line_thickness = max(1, int(indicator_size * self.wind_arrow_thickness_factor))
        head_len = max(2, int(indicator_size * self.wind_arrow_head_size_factor)) # Ensure arrowhead is visible

        if wind_direction_in_cell == WIND_NONE:
            radius = int(indicator_size * self.no_wind_symbol_radius_factor)
            pygame.draw.circle(
                indicator_surface,
                self.no_wind_indicator_color,
                (center_x, center_y),
                radius,
                width=line_thickness 
            )
        else:
            arrow_color = self.wind_indicator_color
            
            start_pos, end_pos = None, None

            if wind_direction_in_cell == WIND_NORTH:
                start_pos = (center_x, center_y + arrow_length // 2)
                end_pos = (center_x, center_y - arrow_length // 2)
            elif wind_direction_in_cell == WIND_SOUTH:
                start_pos = (center_x, center_y - arrow_length // 2)
                end_pos = (center_x, center_y + arrow_length // 2)
            elif wind_direction_in_cell == WIND_EAST:
                start_pos = (center_x - arrow_length // 2, center_y)
                end_pos = (center_x + arrow_length // 2, center_y)
            elif wind_direction_in_cell == WIND_WEST:
                start_pos = (center_x + arrow_length // 2, center_y)
                end_pos = (center_x - arrow_length // 2, center_y)

            if start_pos and end_pos:
                pygame.draw.line(indicator_surface, arrow_color, start_pos, end_pos, line_thickness)

                # Draw arrowhead
                if wind_direction_in_cell == WIND_NORTH:
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] - head_len // 2, end_pos[1] + head_len // 2), line_thickness)
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] + head_len // 2, end_pos[1] + head_len // 2), line_thickness)
                elif wind_direction_in_cell == WIND_SOUTH:
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] - head_len // 2, end_pos[1] - head_len // 2), line_thickness)
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] + head_len // 2, end_pos[1] - head_len // 2), line_thickness)
                elif wind_direction_in_cell == WIND_EAST:
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] - head_len // 2, end_pos[1] - head_len // 2), line_thickness)
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] - head_len // 2, end_pos[1] + head_len // 2), line_thickness)
                elif wind_direction_in_cell == WIND_WEST:
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] + head_len // 2, end_pos[1] - head_len // 2), line_thickness)
                    pygame.draw.line(indicator_surface, arrow_color, end_pos, (end_pos[0] + head_len // 2, end_pos[1] + head_len // 2), line_thickness)
        return indicator_surface

    def render(self, mode=None):
        current_render_mode = mode if mode else self.render_mode

        if current_render_mode is None:
            return None

        if current_render_mode == "rgb_array":
            # Ensure Pygame is initialized for surface creation and drawing
            if not pygame.get_init(): 
                pygame.init()
            if not pygame.display.get_init(): # display.init() might be needed for some operations
                pygame.display.init()

            canvas = pygame.Surface((self.ncol * TILE_SIZE, self.nrow * TILE_SIZE))
            canvas.fill((255, 255, 255))  # White background

            # Draw terrain tiles (procedurally, with special Goal shape)
            for r_idx in range(self.nrow):
                for c_idx in range(self.ncol):
                    cell_char = self.desc[r_idx, c_idx].decode('utf-8')
                    rect_x, rect_y = c_idx * TILE_SIZE, r_idx * TILE_SIZE
                    rect = pygame.Rect(rect_x, rect_y, TILE_SIZE, TILE_SIZE)
                    
                    if cell_char == 'G':
                        # Draw a diamond for the Goal
                        diamond_points = [
                            (rect_x + TILE_SIZE // 2, rect_y + TILE_SIZE // 10),  # Top point
                            (rect_x + TILE_SIZE - TILE_SIZE // 10, rect_y + TILE_SIZE // 2), # Right point
                            (rect_x + TILE_SIZE // 2, rect_y + TILE_SIZE - TILE_SIZE // 10), # Bottom point
                            (rect_x + TILE_SIZE // 10, rect_y + TILE_SIZE // 2)  # Left point
                        ]
                        pygame.draw.polygon(canvas, COLOR_GOAL, diamond_points)
                    else:
                        color = COLOR_FROZEN # Default for 'F'
                        if cell_char == 'S': color = COLOR_START
                        elif cell_char == 'H': color = COLOR_HOLE
                        # No explicit 'G' here as it's handled above; F is the other main type
                        pygame.draw.rect(canvas, color, rect)
            
            # Draw agent (as a centered circle)
            if self.agent_pos is not None:
                agent_r, agent_c = self._to_rc(self.agent_pos)
                center_x = agent_c * TILE_SIZE + TILE_SIZE // 2
                center_y = agent_r * TILE_SIZE + TILE_SIZE // 2
                agent_radius = TILE_SIZE // 4 # Smaller circle for agent
                pygame.draw.circle(canvas, COLOR_AGENT, (center_x, center_y), agent_radius)

            # Draw wind indicators (scaled to TILE_SIZE for rgb_array)
            indicator_actual_size = int(TILE_SIZE * self.wind_indicator_size_factor)
            margin_actual_size = int(TILE_SIZE * self.wind_indicator_margin_factor)
            for r_idx in range(self.nrow):
                for c_idx in range(self.ncol):
                    wind_in_cell = self.wind_map[r_idx, c_idx]
                    cell_char = self.desc[r_idx, c_idx].decode('utf-8')
                    if cell_char in ('H', 'G'): continue 
                    if wind_in_cell != WIND_NONE or cell_char in ('S', 'F'):
                        indicator_surface = self._render_wind_indicator(wind_in_cell, indicator_actual_size)
                        blit_pos_x = c_idx * TILE_SIZE + margin_actual_size
                        blit_pos_y = r_idx * TILE_SIZE + margin_actual_size
                        canvas.blit(indicator_surface, (blit_pos_x, blit_pos_y))
            
            img_array = pygame.surfarray.array3d(canvas)
            return np.transpose(img_array, (1, 0, 2))

        elif current_render_mode == "human":
            if self.window_surface is None: # Should have been initialized in __init__ if human mode
                pygame.init()
                pygame.display.init()
                self.window_size = (self.ncol * self.cell_size, self.nrow * self.cell_size)
                self.window_surface = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption(f"FrozenLake - {self.map_name}")
                self.clock = pygame.time.Clock()

            canvas = self.window_surface
            canvas.fill((255, 255, 255)) # White background

            # --- Procedural rendering for human mode (as before) ---
            for r_idx in range(self.nrow):
                for c_idx in range(self.ncol):
                    cell_char = self.desc[r_idx, c_idx].decode('utf-8')
                    rect = pygame.Rect(c_idx * self.cell_size, r_idx * self.cell_size, self.cell_size, self.cell_size)
                    color = COLOR_FROZEN
                    if cell_char == 'S': color = COLOR_START
                    elif cell_char == 'H': color = COLOR_HOLE
                    elif cell_char == 'G': color = COLOR_GOAL
                    pygame.draw.rect(canvas, color, rect)
            
            # Draw wind indicators (scaled to self.cell_size for human mode)
            indicator_actual_size = int(self.cell_size * self.wind_indicator_size_factor)
            margin_actual_size = int(self.cell_size * self.wind_indicator_margin_factor)
            for r_idx in range(self.nrow):
                for c_idx in range(self.ncol):
                    wind_in_cell = self.wind_map[r_idx, c_idx]
                    cell_char = self.desc[r_idx, c_idx].decode('utf-8')
                    if cell_char in ('H', 'G'): continue
                    if wind_in_cell != WIND_NONE or cell_char in ('S', 'F'):
                        indicator_surface = self._render_wind_indicator(wind_in_cell, indicator_actual_size)
                        blit_pos_x = c_idx * self.cell_size + margin_actual_size
                        blit_pos_y = r_idx * self.cell_size + margin_actual_size
                        canvas.blit(indicator_surface, (blit_pos_x, blit_pos_y))

            # Draw agent (procedural for human mode)
            if self.agent_pos is not None:
                agent_r, agent_c = self._to_rc(self.agent_pos)
                agent_rect = pygame.Rect(
                    agent_c * self.cell_size + self.cell_size // 4, 
                    agent_r * self.cell_size + self.cell_size // 4,
                    self.cell_size // 2, 
                    self.cell_size // 2)
                pygame.draw.rect(canvas, COLOR_AGENT, agent_rect)

            pygame.event.pump()
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"]) # Use metadata for fps
            return None
        
        return None # Should not be reached

    def close(self):
        self.env.close()

class FrozenLakePCH(PCH[PolicyType, ObsType, ActType, PolicyType, ObsType, ActType]):
    metadata = FrozenLakeSCM.metadata

    def __init__(self, **kwargs):
        self.env = FrozenLakeSCM(**kwargs)
        super().__init__(env=self.env)

    def see(self):
        a = self.env.action()
        obs, r, terminated, truncated, info = self.env.step(a)
        return a, obs, r, terminated, truncated, info

    def do(self, action: ActType):
        obs, r, terminated, truncated, info = self.env.step(action)
        return obs, r, terminated, truncated, info

if __name__ == "__main__":
    import time

    print("Testing unified FrozenLakePCH with wind and human rendering...")
    
    wind_cycle = [WIND_NONE, WIND_NORTH, WIND_EAST, WIND_SOUTH, WIND_WEST] 

    # Test with is_slippery = False
    print("\nTesting with is_slippery = False")
    # Ensure render_mode="human" for __main__ tests that expect a window
    env_nonslippery = FrozenLakePCH(render_mode="human", map_name="4x4", is_slippery=False, 
                                    wind_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2))
    obs, info = env_nonslippery.reset(seed=42)
    if env_nonslippery.env.render_mode == "human": env_nonslippery.render() 
    time.sleep(0.5)
    print(f"Initial non-slippery info: {info}")
    if env_nonslippery.env.wind_map is not None:
        print("Initial wind_map (non-slippery):\n", env_nonslippery.env.wind_map)

    for i in range(5): # Shortened loop for quicker testing
        current_agent_state = env_nonslippery.env.env.unwrapped.s 
        agent_r, agent_c = current_agent_state // env_nonslippery.env.ncol, current_agent_state % env_nonslippery.env.ncol
        current_cell_wind = "N/A"
        if env_nonslippery.env.wind_map is not None and 0 <= agent_r < env_nonslippery.env.nrow and 0 <= agent_c < env_nonslippery.env.ncol:
            current_cell_wind = env_nonslippery.env.wind_map[agent_r, agent_c].item()
        
        action = env_nonslippery.action_space.sample()
        print(f"Step {i}: Action: {action}, Agent at ({agent_r},{agent_c}), Wind in cell: {current_cell_wind}")
        
        obs, reward, terminated, truncated, info = env_nonslippery.do(action)
        if env_nonslippery.env.render_mode == "human": env_nonslippery.render()
        
        print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env_nonslippery.reset(seed=42 + i + 1)
            if env_nonslippery.env.render_mode == "human": env_nonslippery.render()
            if env_nonslippery.env.wind_map is not None:
                print("New wind_map after reset:\n", env_nonslippery.env.wind_map)

        if env_nonslippery.env.render_mode == "human":
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    env_nonslippery.close()
                    pygame.quit() 
                    exit()
        time.sleep(0.3)

    print("Closing non-slippery environment.")
    env_nonslippery.close()
    time.sleep(1)

    # Test with is_slippery = True
    print("\nTesting with is_slippery = True")
    env_slippery = FrozenLakePCH(render_mode="human", map_name="8x8", is_slippery=True,
                                 wind_probabilities=(0.1,0.1,0.3,0.2,0.3)) 
    obs, info = env_slippery.reset(seed=123)
    if env_slippery.env.render_mode == "human": env_slippery.render()
    time.sleep(0.5)
    print(f"Initial slippery info: {info}")
    if env_slippery.env.wind_map is not None:
        print("Initial wind_map (slippery):\n", env_slippery.env.wind_map)

    for i in range(5): # Shortened loop
        current_agent_state_slip = env_slippery.env.env.unwrapped.s
        agent_r_slip, agent_c_slip = current_agent_state_slip // env_slippery.env.ncol, current_agent_state_slip % env_slippery.env.ncol
        current_cell_wind_slip = "N/A"
        if env_slippery.env.wind_map is not None and 0 <= agent_r_slip < env_slippery.env.nrow and 0 <= agent_c_slip < env_slippery.env.ncol:
            current_cell_wind_slip = env_slippery.env.wind_map[agent_r_slip, agent_c_slip].item()

        action = env_slippery.action_space.sample()
        print(f"Step {i}: Action: {action}, Agent at ({agent_r_slip},{agent_c_slip}), Wind in cell: {current_cell_wind_slip}")

        obs, reward, terminated, truncated, info = env_slippery.do(action)
        if env_slippery.env.render_mode == "human": env_slippery.render()
        
        print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env_slippery.reset(seed=123 + i + 1)
            if env_slippery.env.render_mode == "human": env_slippery.render()
            if env_slippery.env.wind_map is not None:
                print("New wind_map after reset:\n", env_slippery.env.wind_map)

        if env_slippery.env.render_mode == "human":
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    env_slippery.close()
                    pygame.quit() 
                    exit()
        time.sleep(0.2) 

    print("Closing slippery environment.")
    env_slippery.close()

    print("\n__main__ block complete. Pygame should quit if not exited manually.")
    pygame.quit()