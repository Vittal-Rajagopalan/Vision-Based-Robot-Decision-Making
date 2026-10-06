# Vision-Based-Robot-Decision-Making
Train and test an agent to reach a goal while avoiding a moving obstacle on a preset grid.

This repository holds all the files necessary to train an AI agent using Reinforcement Learning (RL), record training statistics, and compare against a baseline, rule based model. 

Requirements
Python 3.9+
Packages: gymnasium, numpy, matplotlib
pip install gymnasium numpy matplotlib

How to Run
1. Download 3 Python files in a single folder
2. Run main script (safe_robot_rl.py)
3. Follow menu instructions to train, then test or evaluate
4.   Ensure a model has been trained before testing or evaluating

Environment default values:
GRID_SIZE = 8
MAX_STEPS = 60
STARTING_ROBOT_POSITION = np.array([0, 0])
STARTING_COW_POSITION = np.array([4, 4])
STARTING_GOAL_POSITION = np.array([7, 7])
COW_MOVE_PROB = 0.7
ACTION_NAMES = {0: "right", 1: "up", 2: "left", 3: "down", 4: "stay"}

Hyperparameter default values:
EPISODES - (Total tests): 15,000
ALPHA - (Learning rate): 0.1
GAMMA - (long-term thinking): 0.95
EPSILON - (Starting exploration amount): 1.0
EPSILON_MIN - (minimum value for epsilon): 0.05
EPSILON_DECAY - (rate of epsilon reduction per episode): 0.99995
