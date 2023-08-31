import numpy as np

# Save directory, includes model and replay memory.
save_path = "./save"
backup_path = "./backup_save"

HOST = "127.0.0.1"
PORT = 11225


##### Hyperparameters below control the training of the agent. #####

width = 144
height = 81

# How many SECONDS of updates the agent keeps in its stack
stack_size = 5  # Phi; amount of frames the agent sees e.g. stack_size 4 means the agent sees the
# last 4 frames.
state_size = 6 # Length of the state array received from the game.

'''Inputs (all -1 to +1):
 - Pitch
 - Roll
 - Throttle
 - Fire'''
num_actions = 4 # Number of possible actions


# The epsilon-greedy slope stops changing after this many frames.
explore_count = 100000
start_epsilon = 1           # The epsilon slope begins at this float.
end_epsilon = 0.1           # The epsilon slope stops at this float.

# Discount factor. A higher discount factor determines how much the agent
# should care about prioritizing the future vs. the present.
discount = 0.99

learning_rate = 0.000001   # AKA step size.

memory_size = 100000  # The size of the replay memory.
batch_size = 128     # The mini-batch size used for a gradient descent step.
priority_ratio = 0.125

# ============================================== #

# The reward for shooting down the enemy.
reward_kill = 1

# The punishment for dying.
reward_death = -1

max_reward_angle = 0.005
min_reward_angle = -0.010
angle_threshold = 180

#============================================== #

# Autosave after this many episodes.
autosave_period = 50

cam_memtype = np.ubyte
memtype = np.float32

# Disable epsilon, always take the best action.
# Use to evaluate the best possible performance.
performance_mode = False

#============================================== #
