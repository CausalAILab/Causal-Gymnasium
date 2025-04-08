# causal-gym
This repository contains the simulator code for Causal AI Gym, which includes a set of potentially confounded environments and general environments beyond (PO)MDPs. We adopt the formulation of Strucutural Causal Models (SCMs). For more details on SCM definitions and do-calculus please see: [link](https://en.wikipedia.org/wiki/Causality_(book)).

## Installation
Download this repository to your local machine and at the project root directory, open your terminal, type in 
```
pip install .
```

## Supported Environments
- **Confounded MiniGrid**: We built a windy grid world environment wrapper upon the [MiniGrid](https://minigrid.farama.org) codebase. With this easily configurable grid world environments set, you can define your own grid world with confounders to suite your research needs!  
See ``test/test_windyminigrid.ipynb`` for use cases and examples!
- **Confounded Inventory Control**: We adopt the inventory contorl problem introduced in [Algorithms in Reinforcement Learning](https://sites.ualberta.ca/~szepesva/rlbook.html) Example 1 and The Causal Inference Book Chap. 7 Example 7.2.  
See ``test/test_mdp_example.ipynb`` for use cases and examples!

## Notes
- In installation instructions, should use "pip install -e ." to avoid the environment image assets not showing up in the causalgym package in Conda