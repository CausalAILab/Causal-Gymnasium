import os
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum


#  Wrapped & Simplified action system 
class WrappedActions(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3
    still = 4

# translate our action direction into the direction system in minigrid
ACT_TO_DIR = {
    WrappedActions.up: 3,
    WrappedActions.down: 1,
    WrappedActions.left: 2,
    WrappedActions.right: 0, 
    WrappedActions.still: 4
}

package_path = os.path.dirname(os.path.abspath(__file__))
# returned result is four channel, the forth channel represents transparency, 
# 0 for fully transparent, 1 for fully opaque
COIN_IMG = plt.imread(os.path.join(package_path, 'assets', 'coin.png'))
ROBO_IMG = plt.imread(os.path.join(package_path, 'assets', 'robohead.png'))
FLAG_IMG = plt.imread(os.path.join(package_path, 'assets', 'flag.png'))
cross = plt.imread(os.path.join(package_path, 'assets', 'cross.png'))
circle = plt.imread(os.path.join(package_path, 'assets', 'still.png'))
up_arrow = plt.imread(os.path.join(package_path, 'assets', 'arrow.png'))
left_arrow = np.rot90(up_arrow, k=1, axes=(0,1))
down_arrow = np.rot90(up_arrow, k=2, axes=(0,1))
right_arrow = np.rot90(up_arrow, k=3, axes=(0,1))
WIND_ICONS = [right_arrow, down_arrow, left_arrow, up_arrow, circle, cross]