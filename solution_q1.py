import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

# Function to choose an action based on the epsilon-greedy policy
def choose_action(state, Q, epsilon):
    # Only use the first element of the state which is the actual state tuple
    state_tuple = state[0] if isinstance(state, tuple) else state
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore: random action
    else:
        return np.argmax(Q[state_tuple])  # Exploit: best action based on current Q-values

# Initialize the Blackjack environment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Initialize Q-values (action-value function) to zero for all state-action pairs
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 1.0  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 50000  # Total number of training episodes

# Q-learning algorithm
for episode in range(num_episodes):
    # Start a new episode and get the initial state
    current_state = env.reset()
    
    # Only use the first element of the state which is the actual state tuple
    current_state = current_state[0] if isinstance(current_state, tuple) else current_state
    
    done = False
    while not done:
        # Select an action using the policy derived from Q (epsilon-greedy)
        action = choose_action(current_state, Q, epsilon)
        
        # Take the action and observe the new state and reward
        step_result = env.step(action)

        # Unpack the step_result with flexibility for different lengths of return values
        next_state = step_result[0]
        reward = step_result[1]
        done = step_result[2]
        
        # Only use the first element of the next state which is the actual state tuple
        next_state = next_state[0] if isinstance(next_state, tuple) else next_state
        
        # Get the best Q-value for the next state
        next_best = np.max(Q[next_state])
        
        # Calculate the Q-learning target value
        td_target = reward + gamma * next_best
        
        # Update the Q-value for the current state and action
        Q[current_state][action] += alpha * (td_target - Q[current_state][action])
        
        # Transition to the next state
        current_state = next_state

# Close the environment
env.close()

# Print out the first few entries in our Q-table (state-action values)
for state, actions in list(Q.items())[:5]:
    print(f"State: {state}, Actions: {actions}")

