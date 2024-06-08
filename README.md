# simple_rl
![plot](img/reinfo_banner.png?raw=true "Screenshot")

This project demonstrates a simple reinforcement learning agent (a red square) learning to reach a target (a green square) using Q-learning in a PyGame environment.

## Requirements

- Python 3.x
- PyGame
- NumPy
- Matplotlib

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/MarcelloMorettoni/simple_rl.git
   cd simple_rl

Files
train.py: Contains the code for training the agent using Q-learning.
test.py: Contains the code for testing the agent using the learned Q-table.
q_table.pkl: The file where the learned Q-table is saved (created after running train.py).
README.md: This file.
How It Works
Q-learning Algorithm
The agent learns to reach the target by interacting with the environment and updating its Q-table based on the rewards received. The Q-learning algorithm is a type of reinforcement learning where the agent learns the value of taking specific actions in specific states.

Environment Setup
The agent (red square) starts at a fixed position.
The target (green square) is at a fixed position but can be made dynamic.
The agent receives a reward of +100 for reaching the target and -1 for each step taken.
Training Loop
The agent selects an action using an epsilon-greedy policy.
The agent moves based on the selected action.
The Q-table is updated based on the received reward and the new state.
The environment is rendered using PyGame.
Testing Loop
The agent selects actions based on the learned Q-table.
The agent moves towards the target using the optimal policy.
The environment is rendered using PyGame.
Learning Progress
The total rewards per episode are tracked and plotted using Matplotlib to visualize the agent's learning progress. Over time, the agent should learn to reach the target more efficiently, resulting in higher total rewards per episode.

Example Output
After training, you can expect the agent to learn an optimal path to the target. The learning progress can be visualized in a plot showing the total rewards per episode.


Contributing
Feel free to submit issues or pull requests if you have suggestions for improvements or find any bugs.

License
This project is licensed under the MIT License. See the LICENSE file for details.
