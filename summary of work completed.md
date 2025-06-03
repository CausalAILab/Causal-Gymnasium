## Summary of Work Completed

# Environments:
**Frozen Lake (discrete)**
Grid environment, comparable to exiting minigrid envs.
Latent variable: wind (changes on each action)
Causal diagram?

**Lunar Lander (continuous)**
Continuous environment
Exposes a discrete interface for learning algos
Implements wind on a grid basis

**Cartpole Wind (continuous)**
Continuous, exposes discrete
Latent wind variable changes on each action (step) the algo takes

# Algos
UCBVI
UCBQ

Standard algorithms 26 and 27 from 9.4 of the textbook.
Intercepts next action with env.action(), determines next action w/ ucbq, takes the action w/ env.do(), updates ucbq.
Existing action can be anything, as per the textbook, but should cover all actions. Uniform distribution assists w/ exploration, ok and simple to use.
**Key Finding:** Crucial parameters for UCBVI (planning horizon vs. reward scaling horizon, `c_bonus`) were identified and tuned, significantly improving learning on FrozenLake. These will need careful per-environment tuning for Cartpole and Lunar Lander.

# Testing

## Regular
Exists for Frozen Lake, Lunar Lander, and Cartpole Wind
Shows a basic non-learning algorithm
Shows how to interact with the environments (similar to test_lava)

## Learning
Uses UCBVI to learn an optimal action for the environment

**Frozen Lake**
Trains algo
Shows a successful run, when the run was first successful, regret over time, etc

**Lunar Lander**
Tests different discretizations of the env
Outputs successful run example
Shows regret
Difficult b/c of continuous env, can take a while to learn

**Cartpole Wind**
Also tests discretizations, successful run, regret


Summary

UCB algo
3 envs, 2 tests each (one basic, one learning w/ algo)

TODO:
Show this work.
Algos (ucbvi, and ucbq) are in .important. Need to ensure UCBVI is properly counterfactual. Should follow steps listed above.
Tests are for learning w/ algos.
Should use a base policy of uniform actions.

I'll need to:

Write up a report containing details on each environment, how a latent variable was added, how this creates a causal graph (what? isn't it just wind -> move? see what other envs do if they show a graph), and how to use the environment, especially to do ctf-do. If needed, we can easily implement a CTF-do function.

We can report on how each demo file does it's demo (they're all notebooks, or should be). Basic stuff, like how to set up the environment, how to take an action, how to activate the latent variable, etc.

We can then talk about learning. Should also be pretty straightforward. UCBQ is the most complex part, but it's filled out when learning. We see the next action, look up the best action to take w/ ucbq, take that action, then fill out a reward.

We should show examples of the algo being successful. How long it takes to start succeeding regularly, expectations of reward and regret, changes to the base algorithm, etc.