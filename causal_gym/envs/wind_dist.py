def maze_wind_dist(state):
    if state[0] == 1:
        # downward or still
        return (0., .5, 0., 0., .5)
    elif state[1] == 7:
        # right wind
        return (.5, 0., 0., 0., .5)
    else:
        # all other states
        return (.15, .15, .15, .15, .4)
    
def maze2_wind_dist(state):
    if (state[0] == 1 and state[1] <= 3) or tuple(state) in [(2,3), (3,3)]:
        # mainly still, slight down
        return (0., .2, 0., 0., .8)
    elif state[0] == 1 and state[1] > 3:
        # downward or still
        return (0., .5, 0., 0., .5)
    elif state[0] == 7 or tuple(state) in [(4,5),]:
        # strong left
        return (.0, .0, .8, 0., .2)
    elif state[1] >= 4 and state[1] < 7 and state[0] < 7:
        # strong downward, slight left, slight still
        return (.0, .8, .1, 0., .1)
    elif state[1] == 7 or tuple(state) in [(2,2), (3,2)]:
        # strong right wind
        return (.8, .0, 0., 0., .2)
    else:
        # all other states
        return (.15, .15, .15, .15, .4)

# AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
WIND_DIST = {
    'Custom-LavaCrossing-easy-v0': (0, .8, 0, 0, .2),
    'Custom-LavaCrossing-hard-v0': (0, .8, 0, 0, .2),
    'Custom-LavaCrossing-extreme-v0': (0, .7, 0, 0, .3),
    'Custom-LavaCrossing-maze-v0': maze_wind_dist,
    'Custom-LavaCrossing-maze-complex-v0': maze2_wind_dist,
}