import pygame
import numpy as np
import pickle
import os

# Initialize PyGame
pygame.init()

# Screen dimensions
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Reinforcement Learning Example')

# Colors
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

# Parameters
agent_size = 20
target_size = 20

# Initialize agent and target positions
agent_pos = np.array([width // 4, height // 2])
#target_pos = np.array([3 * width // 4, height // 2])
target_pos = np.array([width -20, height -20 ])

# Load the Q-table
q_table_file = 'q_table.pkl'
if os.path.exists(q_table_file):
    with open(q_table_file, 'rb') as f:
        q_table = pickle.load(f)
else:
    raise FileNotFoundError("Q-table file not found!")

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

# Run the agent using the learned policy
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = (agent_pos[0] // agent_size, agent_pos[1] // agent_size)
    action = np.argmax(q_table[state])
    agent_pos = move_agent(agent_pos.copy(), action)

    # Check if agent reaches the target
    if np.array_equal(agent_pos, target_pos):
        print("Target reached!")
        running = False

    # Drawing
    screen.fill(black)
    pygame.draw.rect(screen, green, (*target_pos, target_size, target_size))
    pygame.draw.rect(screen, red, (*agent_pos, agent_size, agent_size))
    pygame.display.flip()
    pygame.time.wait(100)

pygame.quit()
