import pygame
import numpy as np
import random
import pickle
import os
import matplotlib.pyplot as plt

# Initialize PyGame
pygame.init()

# Screen dimensions
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Reinforcement Learning Example')

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

# Parameters
agent_size = 20
target_size = 20
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration factor

# Initialize agent and target positions
agent_pos = np.array([width // 4, height // 2])
#target_pos = np.array([2.5 * width // 4, height // 2])
target_pos = np.array([width -20, height -20 ])
# Define file paths for saving and loading
q_table_file = 'q_table.pkl'

# Load the Q-table if it exists
if os.path.exists(q_table_file):
    with open(q_table_file, 'rb') as f:
        q_table = pickle.load(f)
else:
    q_table = np.zeros((width // agent_size, height // agent_size, 4))

# Functions to move the agent
def move_agent(pos, action):
    if action == 0:
        pos[1] -= agent_size
    elif action == 1:
        pos[1] += agent_size
    elif action == 2:
        pos[0] -= agent_size
    elif action == 3:
        pos[0] += agent_size
    pos[0] = np.clip(pos[0], 0, width - agent_size)
    pos[1] = np.clip(pos[1], 0, height - agent_size)
    return pos

# Monitoring learning progress
rewards_per_episode = []

# Main loop
num_episodes = 1000
for episode in range(num_episodes):
    total_reward = 0
    agent_pos = np.array([width // 4, height // 2])  # Reset agent position

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                with open(q_table_file, 'wb') as f:
                    pickle.dump(q_table, f)
                plt.plot(rewards_per_episode)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Learning Progress')
                plt.show()
                exit()

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            state = (agent_pos[0] // agent_size, agent_pos[1] // agent_size)
            action = np.argmax(q_table[state])

        # Move the agent
        next_pos = move_agent(agent_pos.copy(), action)

        # Calculate the reward
        if np.array_equal(next_pos, target_pos):
            reward = 100
            next_pos = np.array([width // 4, height // 2])  # Reset position
            done = True
        else:
            reward = -1
            done = False

        total_reward += reward

        # Update Q-table
        state = (agent_pos[0] // agent_size, agent_pos[1] // agent_size)
        next_state = (next_pos[0] // agent_size, next_pos[1] // agent_size)
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state]))

        agent_pos = next_pos

        # Drawing
        screen.fill(black)
        pygame.draw.rect(screen, green, (*target_pos, target_size, target_size))
        pygame.draw.rect(screen, red, (*agent_pos, agent_size, agent_size))
        pygame.display.flip()
        pygame.time.wait(100)

        if done:
            break

    rewards_per_episode.append(total_reward)

# Save the Q-table
with open(q_table_file, 'wb') as f:
    pickle.dump(q_table, f)

# Plot the rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Progress')
plt.show()

pygame.quit()
