from causal_gym.envs import MABExampleSCM, MABExamplePCH
from causal_gym.envs.mdp_example import MDPExampleSCM, MDPExamplePCH
from multiprocess import Process
from tqdm import tqdm
import numpy as np
import multiprocess as mp
import pandas as pd
import matplotlib.pyplot as plt

def run_episode(policy=None, seed=None, GAMMA=0.9, MAX_STEPS=10000, env=None):
    #Run a single episode in the MDP environment.
    

    # Initialize environment
    environment = None
    if env is None:
        environment = MDPExamplePCH(max_step=MAX_STEPS)
    else:
        environment = env
    
    # Reset environment with seed
    s, _ = environment.reset(seed=seed)
    
    # Track states, actions, rewards
    states = [s]
    actions = []
    rewards = []
    
    done = False
    discounted_reward = 0
    step = 0
    
    # Run episode
    while not done:
        if policy is None:
            # Use behavioral policy ('see' mode)
            x, s, y, _, done, _ = environment.see()
        else:
            # Use provided policy ('do' mode)
            s, y, _, done, _ = environment.do(policy(s))
        
        # Update tracking variables
        if policy is not None:
            actions.append(policy(states[-1]))
        else:
            actions.append(x)
        states.append(s)
        rewards.append(y)
        
        # Update discounted reward
        discounted_reward = discounted_reward * GAMMA + y
        
        step += 1
        if step >= MAX_STEPS:
            break
    
    return discounted_reward, states, actions, rewards

# Function to evaluate a policy using parallel processing
def evaluate_policy(policy, policy_name, N_SAMPLES=2000, env=None):
    manager = mp.Manager()
    results = manager.list()
    processes = []
    
    # Create and start processes
    for i in range(N_SAMPLES):
        p = Process(target=lambda: results.append(run_episode(policy, seed=i, env=env)))
        processes.append(p)
        p.start()
        
    # Wait for processes to complete
    for p in tqdm(processes, desc=f"Evaluating {policy_name}"):
        p.join()
        
    # Process results
    rewards = []
    daily_profits = []
    for result in results:
        discounted_reward, states, actions, rewards_list = result
        rewards.append(discounted_reward)
        daily_profits.extend(rewards_list)
      
    return {
        'name': policy_name,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'daily_profit': np.mean(daily_profits),
        'rewards': rewards,
        'daily_profits': daily_profits
    }

def collect_interventional_data_mdp(n_samples=1000, MAX_STEPS=10000):
    '''
    Collect data on transitions and rewards using atomic interventions.
    
    This function performs do(X_i=x_i) for different state-action pairs and collects the
    resulting transitions and rewards.
    
    Returns:
        Dictionary containing interventional transition probabilities and expected rewards
    '''
    # Initialize data structures to store transition counts and reward sums
    transition_counts = np.zeros((2, 2, 2))  # [s, x, s']
    reward_sums = np.zeros((2, 2))  # [s, x]
    counts = np.zeros((2, 2))  # [s, x]
    
    # Iterate through all possible state-action pairs
    for init_state in [0, 1]:
        for action in [0, 1]:
            # Perform n_samples interventions for each state-action pair
            for sample in tqdm(range(n_samples), desc=f"Collecting interventional data for S={init_state}, X={action}"):
                # environment
                env = MDPExamplePCH(max_step=MAX_STEPS)
                
                # environment and set initial state
                s, _ = env.reset(seed=sample)
                
                # Find an episode where the state matches our target initial state
                steps = 0
                while s != init_state and steps < 100:  # Limit steps to avoid infinite loops
                    _, s, _, _, _, _ = env.see()
                    steps += 1
                
                if s == init_state:  # Only proceed if we found the target state
                    # Perform atomic intervention do(X=action)
                    next_s, reward, _, _, _ = env.do(action)
                    
                    # Update counts and sums
                    transition_counts[init_state, action, next_s] += 1
                    reward_sums[init_state, action] += reward
                    counts[init_state, action] += 1
    
    # Calculate transition probabilities and expected rewards
    transition_probs = np.zeros((2, 2, 2))
    expected_rewards = np.zeros((2, 2))
    
    for s in range(2):
        for a in range(2):
            if counts[s, a] > 0:
                # P(S'|S, do(X))
                for next_s in range(2):
                    transition_probs[s, a, next_s] = transition_counts[s, a, next_s] / counts[s, a]
                
                # E[Y|S, do(X)]
                expected_rewards[s, a] = reward_sums[s, a] / counts[s, a]
    
    return {
        'transition_probs': transition_probs,
        'expected_rewards': expected_rewards,
        'counts': counts
    }

def collect_observational_data_mdp(n_samples=1000, MAX_STEPS=10000):
    '''
    Collect observational data on transitions and rewards.
    
    This function observes the environment without intervention and collects
    the resulting transitions and rewards.
    
    Returns:
        Dictionary containing observational transition probabilities and expected rewards
    '''
    # Initialize data structures to store transition counts and reward sums
    transition_counts = np.zeros((2, 2, 2))  # [s, x, s']
    reward_sums = np.zeros((2, 2))  # [s, x]
    counts = np.zeros((2, 2))  # [s, x]
    
    # Collect data from multiple episodes
    for episode in tqdm(range(n_samples), desc="Collecting observational data"):
        # Initialize environment
        env = MDPExamplePCH(max_step=MAX_STEPS)
        
        # Reset environment
        s, _ = env.reset(seed=episode)
        
        # Run episode with behavioral policy
        done = False
        steps = 0
        
        while not done and steps < 100:  # Limit steps to avoid excessive data
            # Use behavioral policy ('see' mode)
            a, next_s, r, _, done, _ = env.see()
            
            # Update counts and sums
            transition_counts[s, a, next_s] += 1
            reward_sums[s, a] += r
            counts[s, a] += 1
            
            s = next_s
            steps += 1
    
    # Calculate transition probabilities and expected rewards
    transition_probs = np.zeros((2, 2, 2))
    expected_rewards = np.zeros((2, 2))
    
    for s in range(2):
        for a in range(2):
            if counts[s, a] > 0:
                # P(S'|S, X)
                for next_s in range(2):
                    transition_probs[s, a, next_s] = transition_counts[s, a, next_s] / counts[s, a]
                
                # E[Y|S, X]
                expected_rewards[s, a] = reward_sums[s, a] / counts[s, a]
    
    return {
        'transition_probs': transition_probs,
        'expected_rewards': expected_rewards,
        'counts': counts
    }

# Function to collect observational data
def collect_observational_data_dtr(env, num_episodes, seed=42):
    data = []
    for _ in tqdm(range(num_episodes), desc="Collecting observational data"):
        # Reset the environment
        s1, _ = env.env.reset()
        
        # First stage
        x1, s2, _, _, _, info1 = env.see()
        
        # Second stage
        x2, _, y, terminated, _, info2 = env.see()
        
        # Record the trajectory
        data.append((s1, x1, s2, x2, y))
    
    # Convert to DataFrame
    return pd.DataFrame(data, columns=['S1', 'X1', 'S2', 'X2', 'Y'])

# Implement the IPW estimator for policy evaluation
def ipw_policy_evaluation(data, policy1, policy2, propensity_x1, propensity_x2):
    """  
    Args:
        data: DataFrame of observational data
        policy1: Function mapping S1 to X1
        policy2: Function mapping (S1,X1,S2) to X2
        propensity_x1: Dictionary mapping S1 to P(X1|S1)
        propensity_x2: Dictionary mapping (S1,X1,S2) to P(X2|S1,X1,S2)
    Returns:
        policy_value: Estimated expected reward for the policy
    """
    # sum of weighted rewards
    weighted_rewards = 0
    
    # iterate through all trajectories (observed sequence of events in environment)
    for _, row in data.iterrows():
        s1, x1, s2, x2, y = row['s1'], row['x1'], row['s2'], row['x2'], row['y']
        
        # compute indicator functions for if x1 = 0 and x2 = 1
        target_prob_x1 = 1.0 if policy1 == x1 else 0.0
        target_prob_x2 = 1.0 if policy2 == x2 else 0.0
        
        # Get behavior policy probabilities (propensity scores)
        behavior_prob_x1 = propensity_x1[s1][x1]
        behavior_prob_x2 = propensity_x2[(s1, x1, s2)][x2]
        
        # Skip if propensity score is 0 (avoid division by zero)
        if behavior_prob_x1 == 0 or behavior_prob_x2 == 0:
            continue
        
        # Compute importance weight
        importance_weight = (target_prob_x1 / behavior_prob_x1) * (target_prob_x2 / behavior_prob_x2)
        
        # Add weighted reward
        weighted_rewards += y * importance_weight
    
    # Normalize by number of trajectories
    policy_value = weighted_rewards / len(data)
    
    return policy_value

# Compute propensity scores from the observational data
def compute_propensity_scores(data):
    """ 
    Args:
        data: DataFrame of observational data
    Returns:
        propensity_x1: Dictionary mapping S1 to P(X1|S1)
        propensity_x2: Dictionary mapping (S1,X1,S2) to P(X2|S1,X1,S2)
    """
    # Compute P(X1=1|S1)
    propensity_x1 = {}
    for s1 in [0, 1]:
        mask = data['s1'] == s1
        if sum(mask) > 0:
            propensity_x1[s1] = {
                0: 1 - data.loc[mask, 'x1'].mean(),
                1: data.loc[mask, 'x1'].mean()
            }
    
    # Compute P(X2=1|S1,X1,S2)
    propensity_x2 = {}
    for s1 in [0, 1]:
        for x1 in [0, 1]:
            for s2 in [0, 1]:
                mask = (data['s1'] == s1) & (data['x1'] == x1) & (data['s2'] == s2)
                if sum(mask) > 0:
                    propensity_x2[(s1, x1, s2)] = {
                        0: 1 - data.loc[mask, 'x2'].mean(),
                        1: data.loc[mask, 'x2'].mean()
                    }
    
    return propensity_x1, propensity_x2

# Define different policies for evaluation
def always_zero_policy(state):
    return 0

def always_one_policy(state):
    return 1

def match_state_policy(state):
    return state

def opposite_state_policy(state):
    return 1 - state

#############################################################
# The below is for the extended windy grid environment test
#############################################################

# run one pass of the environment with the ext_windy_lavagrid pch class
def run_one_pass(ext_windy_lavagrid, behavior_policy1, behavior_policy2, SEED):
    # Reset for demonstration
    observation, info = ext_windy_lavagrid.reset(seed=SEED)
    print(f"Initial position: {ext_windy_lavagrid.env.agent_pos}")
    print(f"Initial wind: {info['wind']}")
    print(f"DTR stage: {ext_windy_lavagrid.env.decision_model.stage}")
    print(f"S1 (derived from position): {ext_windy_lavagrid.env._map_grid_to_dtr_s1()}")
    init_obs = ext_windy_lavagrid.render()

    # First step - use 'see' with our first behavioral policy
    action, next_state, reward, terminated, truncated, info = ext_windy_lavagrid.see(bpolicy=behavior_policy1)
    print(f"\nStep 1:")
    print(f"Action taken: {action}")
    print(f"New position: {ext_windy_lavagrid.env.agent_pos}")
    print(f"Wind direction: {info['wind']}")
    print(f"DTR stage: {ext_windy_lavagrid.env.decision_model.stage}")
    print(f"S2 (derived from position): {ext_windy_lavagrid.env._map_grid_to_dtr_s2()}")
    print(f"Reward: {reward}")
    step1_obs = ext_windy_lavagrid.render()

    # Second step - use 'see' with our second behavioral policy
    action, next_state, reward, terminated, truncated, info = ext_windy_lavagrid.see(bpolicy=behavior_policy2)
    print(f"\nStep 2:")
    print(f"Action taken: {action}")
    print(f"New position: {ext_windy_lavagrid.env.agent_pos}")
    print(f"Wind direction: {info['wind']}")
    print(f"DTR stage: {ext_windy_lavagrid.env.decision_model.stage}")
    print(f"DTR complete: {ext_windy_lavagrid.env.dtr_complete}")
    print(f"Reward: {reward}")
    step2_obs = ext_windy_lavagrid.render()

    # Now DTR is complete, we must reset to start a new episode
    print("\nResetting DTR model for new episode (keeping grid position)...")
    # Keep track of current grid position
    current_pos = ext_windy_lavagrid.env.agent_pos
    current_dir = ext_windy_lavagrid.env.agent_dir
    current_wind = ext_windy_lavagrid.env.wind_dir

    # Reset just the DTR model to start a new episode
    ext_windy_lavagrid.env.decision_model.reset()
    ext_windy_lavagrid.env.dtr_complete = False
    ext_windy_lavagrid.env.dtr_episode_reward = 0.0
    print(f"New DTR stage after reset: {ext_windy_lavagrid.env.decision_model.stage}")

    # Third step - start of new DTR episode
    action, next_state, reward, terminated, truncated, info = ext_windy_lavagrid.see(bpolicy=behavior_policy1)
    print(f"\nStep 3 (First step of new DTR episode):")
    print(f"Action taken: {action}")
    print(f"New position: {ext_windy_lavagrid.env.agent_pos}")
    print(f"Wind direction: {info['wind']}")
    print(f"DTR stage: {ext_windy_lavagrid.env.decision_model.stage}")
    print(f"Reward: {reward}")
    step3_obs = ext_windy_lavagrid.render()

    # Fourth step
    action, next_state, reward, terminated, truncated, info = ext_windy_lavagrid.see(bpolicy=behavior_policy2)
    print(f"\nStep 4 (Second step of new DTR episode):")
    print(f"Action taken: {action}")
    print(f"New position: {ext_windy_lavagrid.env.agent_pos}")
    print(f"Wind direction: {info['wind']}")
    print(f"DTR stage: {ext_windy_lavagrid.env.decision_model.stage}")
    print(f"DTR complete: {ext_windy_lavagrid.env.dtr_complete}")
    print(f"Reward: {reward}")
    step4_obs = ext_windy_lavagrid.render()
    # Visualize the steps
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(init_obs)
    axes[0].axis('off')
    axes[0].set_title('Initial State (S1)')
    axes[1].imshow(step1_obs)
    axes[1].axis('off')
    axes[1].set_title('After First Step (S2)')
    axes[2].imshow(step2_obs)
    axes[2].axis('off')
    axes[2].set_title('After Second Step (DTR Complete)')
    axes[3].imshow(step4_obs)
    axes[3].axis('off')
    axes[3].set_title('After Fourth Step (New DTR Episode)')

    plt.tight_layout()
    plt.show()


def run_multiple_passes_extended_windy_mingrid(num_passes, Goal, ext_windy_lavagrid, behavior_policy1, behavior_policy2, SEED):
    # Run 100 passes and collect data
    results = []

    for pass_idx in range(num_passes):
        # Reset environment for new pass
        observation, info = ext_windy_lavagrid.reset(seed=SEED + pass_idx)
        
        # Initialize tracking variables for this pass
        pass_rewards = 0.0
        steps_taken = 0
        dtr_episodes_completed = 0
        grid_goals_reached = 0
        
        # Continue until terminated or max steps reached
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps_taken < 30:  # 30 steps max per pass
            # Check if we need to reset DTR
            if ext_windy_lavagrid.env.dtr_complete:
                # Reset just the DTR model
                ext_windy_lavagrid.env.decision_model.reset()
                ext_windy_lavagrid.env.dtr_complete = False
                ext_windy_lavagrid.env.dtr_episode_reward = 0.0
                dtr_episodes_completed += 1
            
            # Choose appropriate policy based on DTR stage
            if ext_windy_lavagrid.env.decision_model.stage == 0:
                # First stage of DTR
                action, next_state, reward, terminated, truncated, info = ext_windy_lavagrid.see(bpolicy=behavior_policy1)
            else:
                # Second stage of DTR
                action, next_state, reward, terminated, truncated, info = ext_windy_lavagrid.see(bpolicy=behavior_policy2)
            
            # Update tracking
            pass_rewards += reward
            steps_taken += 1
            
            # Check if grid goal reached
            if terminated and isinstance(ext_windy_lavagrid.env.grid.get(*ext_windy_lavagrid.env.agent_pos), Goal):
                grid_goals_reached += 1
        
        # If DTR was complete at end, count it
        if ext_windy_lavagrid.env.dtr_complete:
            dtr_episodes_completed += 1
        
        # Store results for this pass
        results.append({
            'pass_idx': pass_idx,
            'total_reward': pass_rewards,
            'steps_taken': steps_taken,
            'dtr_episodes_completed': dtr_episodes_completed,
            'grid_goals_reached': grid_goals_reached
        })
    
    return results

def behavior_policy1(ext_windy_lavagrid, state, wind, EXPLORATION_RATE=0.2):
    """First stage policy with exploration (S1 -> X1)"""
    s1 = ext_windy_lavagrid.env._map_grid_to_dtr_s1()
    # Base policy: Forward if S1=1
    base_action = 1 if s1 == 0 else 2
    # With probability EXPLORATION_RATE, take a random action
    if np.random.random() < EXPLORATION_RATE:
        # Choose randomly from available actions: 0=right, 1=forward, 2=left, 3=Stay
        action = np.random.choice([0, 1, 2, 3])
    else:
        action = base_action
    return ext_windy_lavagrid.env._map_decision_to_grid_action(action)

def behavior_policy2(ext_windy_lavagrid, state, wind, EXPLORATION_RATE=0.2):
    """Second stage policy with exploration (S1, X1, S2 -> X2)"""
    s2 = ext_windy_lavagrid.env._map_grid_to_dtr_s2()
    # Base policy: Right if S2=1
    base_action = 0 if s2 == 0 else 3 
    # With probability EXPLORATION_RATE, take a random action
    if np.random.random() < EXPLORATION_RATE:
        # Choose randomly from available actions: 0=right, 1=forward, 2=left, 3=Stay
        action = np.random.choice([0, 1, 2, 3])
    else:
        action = base_action
    return ext_windy_lavagrid.env._map_decision_to_grid_action(action)

# Let's also create alternative policies that are different but reasonable
def alternative_policy1(ext_windy_lavagrid, EXPLORATION_RATE,  state, wind):
    """Alternative policy 1: Prefers different actions than base policy"""
    s1 = ext_windy_lavagrid.env._map_grid_to_dtr_s1()
    # Opposite preferences: Forward if S1=0, Right if S1=1
    action = 2 if s1 == 0 else 1
    return ext_windy_lavagrid.env._map_decision_to_grid_action(action)

def alternative_policy2(ext_windy_lavagrid, state, wind):
    """Alternative policy 2: Prefers different actions than base policy"""
    s2 = ext_windy_lavagrid.env._map_grid_to_dtr_s2()
    # Opposite preferences: Stay if S2=0, Left if S2=1
    action = 3 if s2 == 0 else 0
    return ext_windy_lavagrid.env._map_decision_to_grid_action(action)