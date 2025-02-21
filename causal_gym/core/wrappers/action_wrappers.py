from __future__ import annotations
import numpy as np
from PIL import Image
from enum import IntEnum
from typing import Any, SupportsFloat
from gymnasium import spaces
# from causal_gym.core import ActionPCHWrapper
# from causal_gym.envs import WindyMiniGridPCH
from gymnasium.core import WrapperActType, ActType, ObsType, Env, Wrapper
from minigrid.core.constants import COLORS
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    point_in_circle,
    rotate_fn,
    point_in_line,
)
from minigrid.core.world_object import WorldObj, Ball, Wall, Lava, Goal
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

from .. import PolicyType, ActType, ObsType, SCM, PCH, ActionPCHWrapper
from ...envs import WindyMiniGridPCH
from ...envs.constants import WIND_ICONS, COIN_IMG, FLAG_IMG, ROBO_IMG, ACT_TO_DIR
from ...envs.lava_minigrid import Coin


class Actions(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3
    still = 4

class MiniGridActionRemapWrapper(ActionPCHWrapper):

    def __init__(self, env: WindyMiniGridPCH):
        """Constructor for the action wrapper.

        Args:
            env: Environment to be wrapped.
        """
        assert isinstance(env, WindyMiniGridPCH),f"This only works with WindyMiniGridPCH, not '{type(env)}'"
        ActionPCHWrapper.__init__(self, env)
        self.env = env
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            return self.env.__getattr__(name)
        
    @property
    def agent_dir(self,):
        return self.env.agent_dir
    
    @agent_dir.setter
    def agent_dir(self, new_dir):
        self.env.agent_dir = new_dir

    @property
    def agent_pos(self,):
        return self.env.agent_pos
    
    @agent_pos.setter
    def agent_pos(self, new_pos):
        self.env.agent_pos = new_pos

    @property
    def wind_dir(self,):
        return self.env.wind_dir
    
    @wind_dir.setter
    def wind_dir(self, new_dir):
        self.env.wind_dir = new_dir

    def see(self, bpolicy=None) -> tuple[WrapperActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`see` action using :meth:`self.wrap_action`."""
        if bpolicy is not None:
            unwrapped_bpolicy = lambda state, wind: self.unwrap_action(bpolicy(state, wind))
        else:
            unwrapped_bpolicy = None
        prev_dir = self.env.agent_dir
        action, observation, reward, terminated, truncated, info = self.env.see(bpolicy=unwrapped_bpolicy)
        new_dir = self.env.agent_dir
        return self.wrap_action(action, prev_dir, new_dir), observation, reward, terminated, truncated, info

    def do(self, action: WrapperActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.do` using the modified ``action`` from :meth:`self.unwrap_action`."""
        return self.env.do(self.unwrap_action(action))

    def wrap_action(self, action: ActType, prev_dir: int, new_dir: int) -> WrapperActType:
        """
        Map the actions from the MiniGrid system of direction + three way moves to four-way moves

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        if action == 6:
            action = self.actions.still
        elif new_dir == 0:
            action = self.actions.right
        elif new_dir == 1:
            action = self.actions.down
        elif new_dir == 2:
            action = self.actions.left
        elif new_dir == 3:
            action = self.actions.up
        return action

    def unwrap_action(self, action: WrapperActType) -> ActType:
        """
        Map the actions from four-way moves to the MiniGrid system of direction + three way moves

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        if action == self.actions.left:
            self.env.agent_dir = 2
            action = 2
        elif action == self.actions.right:
            self.env.agent_dir = 0
            action = 2
        elif action == self.actions.up:
            self.env.agent_dir = 3
            action = 2
        elif action == self.actions.down:
            self.env.agent_dir = 1
            action = 2
        elif action == self.actions.still:
            action = 6
        return action
    

    def render_tile(
        self,
        obj: Union[WorldObj, None],
        render_agent: bool,
        wind_dist: tuple = None,
        policy_dir: int = None,
        cur_action: int = None,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 2,
        policy_color: Union[str, tuple] = 'grey',
    ) -> np.ndarray:
        """
        Render the wind dir tile
        """
        # grid background
        img = np.ones(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )*200

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Draw obj if there is any
        if obj is not None:
            if isinstance(obj, Coin):
                if policy_dir is None:
                    # only draw coin when we are not drawing policy mappings
                    coin = COIN_IMG
                    coin = (coin * 255).astype(np.uint8)
                    # 3/4 size grid agent icon
                    overlay = Image.fromarray(coin, 'RGBA').resize((int(np.ceil(tile_size * subdivs * .8)), int(np.ceil(tile_size * subdivs * .8))), Image.Resampling.LANCZOS)
                    base = Image.fromarray(img)
                    base.paste(overlay, box=(int(np.ceil(tile_size * subdivs * .1)), int(np.ceil(tile_size * subdivs * .1))), mask=overlay)
                    # update tile patch
                    img = np.array(base)
            elif isinstance(obj, Lava):
                c = (255, 148, 40)
                # Background color
                fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c)
                # Little waves
                for i in range(3):
                    ylo = 0.3 + 0.2 * i
                    yhi = 0.4 + 0.2 * i
                    fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
                    fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
                    fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
                    fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))
                if not render_agent:
                    # Lava grid is the only thing we need to draw
                    img = downsample(img, subdivs)
                    return img
            elif isinstance(obj, Goal):
                flag = FLAG_IMG
                flag = (flag * 255).astype(np.uint8)
                overlay = Image.fromarray(flag, 'RGBA').resize((int(np.ceil(tile_size * subdivs * .8)), int(np.ceil(tile_size * subdivs * .8))), Image.Resampling.LANCZOS)
                base = Image.fromarray(img)
                base.paste(overlay, box=(int(np.ceil(tile_size * subdivs * .1)), int(np.ceil(tile_size * subdivs * .1))), mask=overlay)
                # update tile patch
                img = np.array(base)
                if not render_agent:
                    # No more drawing at Goal grid
                    img = downsample(img, subdivs)
                    return img
            else:
                obj.render(img)


        # Overlay the agent on top if needed
        if render_agent and policy_dir is None:
            robo_head = ROBO_IMG
            robo_head = (robo_head * 255).astype(np.uint8)
            # 3/4 size grid agent icon
            overlay = Image.fromarray(robo_head, 'RGBA').resize((int(np.ceil(tile_size * subdivs * .8)), int(np.ceil(tile_size * subdivs * .8))), Image.Resampling.LANCZOS)
            base = Image.fromarray(img)
            base.paste(overlay, box=(int(np.ceil(tile_size * subdivs * .1)), int(np.ceil(tile_size * subdivs * .1))), mask=overlay)
            # update tile patch
            img = np.array(base)

        # Draw policy mappings
        if policy_dir is not None:
            if isinstance(policy_color, str):
                assert policy_color in list(COLORS.keys()), f'color str (input: {policy_color}) must be one of these: {COLORS.keys()}.'
                policy_color = COLORS[policy_color]
            if policy_dir != 4:
                tri_fn = point_in_triangle(
                    (0.15, 0.25),
                    (0.85, 0.50),
                    (0.15, 0.75),
                )

                exclude_tri_fn = point_in_triangle(
                    (0.15, 0.25),
                    (0.29, 0.50),
                    (0.15, 0.75),
                )

                combined_fn = lambda x, y: tri_fn(x, y) and not exclude_tri_fn(x, y)

                # Rotate the policy dir based on its direction
                combined_fn = rotate_fn(combined_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * policy_dir)
                fill_coords(img, combined_fn, policy_color)
            else:
                # still
                fill_coords(img, point_in_circle(0.5, 0.5, 0.21), policy_color)

        # Draw action to do at lower left corner
        if cur_action is not None:
            if isinstance(policy_color, str):
                assert policy_color in list(COLORS.keys()), \
                    f'color str (input: {policy_color}) must be one of these: {COLORS.keys()}.'
                policy_color = COLORS[policy_color]
            policy_dir = ACT_TO_DIR[cur_action]
            if policy_dir != 4:
                # initial triangle pointing toward right
                tri_fn = point_in_triangle(
                    (0.05, 0.75),
                    (0.28, 0.84),
                    (0.05, 0.92),
                )

                exclude_tri_fn = point_in_triangle(
                    (0.05, 0.75),
                    (0.09, 0.84),
                    (0.05, 0.92),
                )

                combined_fn = lambda x, y: tri_fn(x, y) and not exclude_tri_fn(x, y)

                # Rotate the policy dir based on its direction
                combined_fn = rotate_fn(combined_fn, cx=0.1667, cy=0.8334, theta=0.5 * np.pi * policy_dir)
                fill_coords(img, combined_fn, policy_color)
            else:
                # still
                fill_coords(img, point_in_circle(0.1667, 0.8334, 0.1), policy_color)

        # Draw wind arrows
        if wind_dist is not None:
            if np.all(np.array(wind_dist[0:3]) > 0):
                if wind_dist[3] >= .5:
                    # circle, cross
                    wind_to_draw = [WIND_ICONS[4], WIND_ICONS[5]]
                else:
                    wind_to_draw = [WIND_ICONS[5], WIND_ICONS[4]]
            else:
                wind_dir_to_draw = sorted(range(len(WIND_ICONS) - 1), key=lambda x:wind_dist[x], reverse=True)
                wind_to_draw = [WIND_ICONS[d] for d in wind_dir_to_draw if wind_dist[d] > 0]
        
            for i, icon in enumerate(wind_to_draw):
                # each wind icon takes about 1/3 * 1/3 space
                icon = (icon * 255).astype(np.uint8)
                overlay = Image.fromarray(icon, 'RGBA').resize((tile_size * subdivs // 3, tile_size * subdivs // 3), Image.Resampling.LANCZOS)
                base = Image.fromarray(img)
                base.paste(overlay, box=(tile_size * subdivs // 3 * (3 - len(wind_to_draw) + i), tile_size * subdivs // 3 * 2), mask=overlay)
                # update tile patch
                img = np.array(base)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        return img
        

    def render(
        self,       
        action_todo: int = None,
        show_policy: bool = True,
        show_wind: bool = True
    ) -> np.ndarray:
        '''
        Given a grid world and a policy mapping from state to directions, 
        rerender the obs with the policy.
        '''
        tile_size = self.tile_size
        agent_pos = self.agent_pos
        width_px = self.env.state_space[0] * tile_size
        height_px = self.env.state_space[1] * tile_size
        init_obs = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)


        # render wind, policy mapping and redraw objs for each state
        for state in np.ndindex(tuple(self.state_space)):
            # Redo everything except walls
            if not (self.grid.get(*state) is None or isinstance(self.grid.get(*state), (Coin, Lava, Goal))):
                continue
            
            # Always render the wind direction
            # Let wind_dist be a one-hot vec when there is wind
            if isinstance(self.wind_dist, Callable):
                wind_dir = np.random.choice(5, p=self.wind_dist(state))
            else:
                wind_dir = np.random.choice(5, p=self.wind_dist)
                
            w_dist = np.zeros(5)
            w_dist[wind_dir] = 1.0
            if state == agent_pos:
                # Let wind_dist be a one-hot vec when there is wind
                w_dist = np.zeros(5)
                # get the current env wind dir
                w_dist[self.wind_dir] = 1.0
                # only render the action indicator within the agent's grid
                act_to_render = action_todo
            else:
                # w_dist = None
                act_to_render = None

            # replace the tile image with prettier ones and add wind/policy
            tile_img = self.render_tile(
                obj=self.grid.get(*state),
                render_agent=(state == agent_pos),
                wind_dist=None if not show_wind else w_dist,
                policy_dir=None,
                cur_action=None if not show_policy else act_to_render,
                tile_size=tile_size,
                policy_color=(255, 60, 60),
                subdivs=1
            )
            i, j = state
            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            init_obs[ymin:ymax, xmin:xmax, :] = tile_img


        # cut the surrounding grey walls
        output = init_obs[tile_size//4*3:tile_size*(self.state_space[0]-1)+tile_size//4, tile_size//4*3:tile_size*(self.state_space[1]-1)+tile_size//4]
        
        return output


    def render_map(
        self,
        mappings: dict = None,
        policy_color: Union[str, tuple] = 'grey',
        draw_wind: bool = False,
        draw_policy: bool = False,
    ) -> np.ndarray:
        '''
        Given a grid world representation and a mapping from state to directions, 
        render direction arrows to the map. 
        '''

        agent_pos = (1, 1)
        tile_size = self.tile_size
        width_px = self.state_space[0] * tile_size
        height_px = self.state_space[1] * tile_size
        init_obs = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # render wind, policy mapping and redraw objs for each state
        for state in np.ndindex(tuple(self.env.state_space)):
            # Redo everything except walls
            if not (self.grid.get(*state) is None or isinstance(self.grid.get(*state), (Coin, Lava, Goal))):
                continue
            
            # whether to draw wind at this state
            if isinstance(self.wind_dist, Callable) and draw_wind:
                w_dist = self.wind_dist(state)
            elif isinstance(self.wind_dist, tuple) and draw_wind:
                w_dist = self.wind_dist
            else:
                w_dist = None

            # whether to draw policy dir at this state
            if draw_policy and mappings is not None and state in mappings.keys():
                action = ACT_TO_DIR[mappings[state]]
            else:
                action = None

            # replace the tile image with prettier ones and add wind/policy
            tile_img = self.render_tile(
                obj=self.grid.get(*state),
                render_agent=(state == agent_pos),
                wind_dist=w_dist,
                policy_dir=action,
                cur_action=None,
                tile_size=tile_size,
                policy_color=policy_color
            )
            i, j = state
            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            init_obs[ymin:ymax, xmin:xmax, :] = tile_img

        # cut the surrounding grey walls
        output = init_obs[tile_size//4*3:tile_size*(self.state_space[0]-1)+tile_size//4, tile_size//4*3:tile_size*(self.state_space[1]-1)+tile_size//4]
        return output

