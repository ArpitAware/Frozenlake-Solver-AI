"""
===========================================
  Reinforcement Learning: Q-Learning Agent
  Environment: FrozenLake (4x4 Grid World)
===========================================

PROBLEM:
  An agent starts at S (Start) and must reach G (Goal)
  while avoiding H (Holes) on a frozen lake.
  
  Grid:
  S F F F      S = Start (safe)
  F H F H      F = Frozen (safe)
  F F F H      H = Hole (game over)
  H F F G      G = Goal (+1 reward)

GOAL:
  Train the agent using Q-Learning so it learns
  the best path to reach G without falling into H.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# STEP 1: CREATE THE ENVIRONMENT
env = gym.make("FrozenLake-v1", is_slippery=False)  # is_slippery=False for easier learning

n_states  = env.observation_space.n   # 16 states (4x4 grid)
n_actions = env.action_space.n        # 4 actions: Left, Down, Right, Up

print(f"States : {n_states}")
print(f"Actions: {n_actions}  (0=Left, 1=Down, 2=Right, 3=Up)")
print()

# STEP 2: INITIALIZE THE Q-TABLE
# Q-Table shape: [16 states x 4 actions] → all zeros at start
Q = np.zeros((n_states, n_actions))

print("Initial Q-Table (all zeros):")
print(Q)
print()

# STEP 3: SET HYPERPARAMETERS
EPISODES        = 2000    # Total training episodes
ALPHA           = 0.8     # Learning rate  (how fast to learn)
GAMMA           = 0.95    # Discount factor (value of future rewards)
EPSILON         = 1.0     # Exploration rate (start: explore 100%)
EPSILON_DECAY   = 0.001   # How fast epsilon decreases
EPSILON_MIN     = 0.01    # Minimum exploration rate

# STEP 4: TRAINING LOOP (Q-LEARNING)
rewards_per_episode = []

print("Training the agent...\n")

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:

        #  Epsilon-Greedy Action Selection 
        # With probability EPSILON  → explore  (random action)
        # With probability 1-EPSILON → exploit (best known action)
        if np.random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()          # EXPLORE
        else:
            action = np.argmax(Q[state, :])             # EXPLOIT

        # ── Take Action & Observe Result ─────────────────────
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Table Update (Bellman Equation) 
        #
        #  Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        #
        #  r              = reward received
        #  γ * max Q(s')  = discounted best future reward
        #
        Q[state, action] = Q[state, action] + ALPHA * (
            reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    # Decay epsilon (explore less as agent learns more)
    EPSILON = max(EPSILON_MIN, EPSILON - EPSILON_DECAY)

    rewards_per_episode.append(total_reward)

    if (episode + 1) % 200 == 0:
        avg = np.mean(rewards_per_episode[-200:])
        print(f"  Episode {episode+1:>4} | Avg Reward (last 200): {avg:.3f} | Epsilon: {EPSILON:.3f}")

# STEP 5: RESULTS
print("\nTraining Complete!\n")
print("Learned Q-Table:")
print(np.round(Q, 3))

# Policy extracted from Q-Table
actions = ["←", "↓", "→", "↑"]
policy  = [actions[np.argmax(Q[s])] for s in range(n_states)]

print("\n Learned Policy (4x4 Grid):")
print("┌──┬──┬──┬──┐")
for row in range(4):
    cells = " | ".join(policy[row*4 : row*4+4])
    print(f"│ {cells} │")
print("└──┴──┴──┴──┘")
print("(H = Hole  | G = Goal)")

# STEP 6: EVALUATE THE TRAINED AGENT
print("\n Evaluating trained agent over 100 test episodes...")
wins = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state, :])   # Pure exploitation
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward == 1.0:
            wins += 1
            break

print(f"   Win Rate: {wins}/100 = {wins}%")

# STEP 7: PLOT TRAINING PROGRESS
# Smooth rewards using a rolling window
window = 100
smoothed = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 4))
plt.plot(rewards_per_episode, alpha=0.3, color='steelblue', label='Raw Reward')
plt.plot(range(window-1, len(rewards_per_episode)), smoothed, color='darkblue', linewidth=2, label=f'{window}-ep Average')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning on FrozenLake — Training Progress")
plt.legend()
plt.tight_layout()
plt.savefig("rl_training_progress.png", dpi=150)
plt.close()
print("\n Training plot saved as 'rl_training_progress.png'")

env.close()