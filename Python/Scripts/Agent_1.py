'''F1 Agent v1: Double DQN'''

# Import packages.
import queue
import random
import math
import time
import tensorflow as tf

# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
from . import config as cfg
from . import Memory_1 as mem


class agent:
    '''This creates an agent that plays and learns to dogfight. All of the functions
       used for training are contained in this class.'''

    def __init__(self):
        self.model = mem.load_models()
        self.train_data, self.stats = mem.load_data()
        self.states_memory, self.action_memory, self.reward_memory, self.transitions_memory = mem.load_memory()

        self.model.summary()


    def init_phi(self, state1, state2):
        '''Initializes the frame stack (phi) with the starting state.'''

        # Newest frames to the end of the queue, oldest to the front.
        self.phi1 = queue.deque()
        self.phi2 = queue.deque()
        # Initialize phi with screen.
        for _ in range(cfg.stack_size):
            self.phi1.append(state1)
        for _ in range(cfg.stack_size):
            self.phi2.append(state2)

        self.last_prediction1, self.last_prediction2 = (None, None)

    def save_all(self, path):
        mem.save_models(path, self.model)
        mem.save_data(path, self.train_data, self.stats)
        mem.save_memory(path, self.states_memory, self.action_memory,
                        self.reward_memory, self.transitions_memory)


    def update_memory(self, states, action, reward, transitions):
        '''Creates a new experience in the replay memory. Each experience is stored between 4
           seperate NumPy arrays, at the same index.

           Function Parameters:
           states <np.ndarray; shape=(game_size^2 * stack_size)> = the state/frame stack for the experience
           action <int> = the action taken
           reward <int> = the reward received for taking said action in state
           transitions <np.ndarray; shape=(game_size^2 * stack_size)> = the frame stack of the frame AFTER taking said action'''

        # Start replacing experiences in the memory from the beginning again.
        if self.train_data["update_index"] >= cfg.memory_size:
            self.train_data["update_index"] = 0

        # Insert the experience into replay memory.
        self.states_memory[self.train_data["update_index"]] = states
        self.action_memory[self.train_data["update_index"]] = action
        self.reward_memory[self.train_data["update_index"]] = reward
        self.transitions_memory[self.train_data["update_index"]] = transitions

        self.train_data["update_index"] += 1
        # Keep track of how much of the memory has been filled.
        if self.train_data["filled_memory"] < cfg.memory_size:
            self.train_data["filled_memory"] += 1

    def actions_from_gaussians(self, means, stdevs):
        # Get samples from Gaussian
        actions = np.random.default_rng().standard_normal(cfg.num_actions)
        # Convert to action probabilities
        actions = actions * stdevs + means
        return actions

    def get_actions(self, plane_number):
        '''Draws action inputs (continous) based on mean and stdev
        values output by the actor.
        
        Returns action inputs sampled from Gaussain. Float[]'''

        # Call np.expand_dims so that it can be used as input to the model.
        if plane_number == 1:
            stack = np.expand_dims(self.phi1, axis=0)
        elif plane_number == 2:
            stack = np.expand_dims(self.phi2, axis=0)

        # Get mean and stdev values from model. Ignores value function.
        prediction = self.model(stack, training=False)
        means = prediction[0].numpy()[0]
        stdevs = prediction[1].numpy()[0]
        value = prediction[2].numpy()[0]
        actions = self.actions_from_gaussians(means, stdevs)

        if plane_number == 1:
            self.last_prediction1 = np.concatenate((means, stdevs, value))
            # print(value)
        elif plane_number == 2:
            self.last_prediction2 = np.concatenate((means, stdevs, value))


        if math.isnan(actions[0]):
            foo = 1
            print("prediction is NaN")
            pass

        return actions, means

    def reward_transition(self, plane_number, reward, transition):
        # Keep the previous frame stack for updating memory.
        if plane_number == 1:
            phi_last = list(self.phi1)
            # Update the frame stack with the latest frame.
            self.phi1.append(transition)
            # Remove oldest update, add newest
            self.phi1.popleft()
            phi_current = self.phi1

            # Update the memory with the last state, the action taken in the last state, the reward for doing so,
            # and the resulting state.
            # The plane # does NOT matter since they share a memory.
            self.update_memory(phi_last, self.last_prediction1, reward, phi_current)
        elif plane_number == 2:
            phi_last = list(self.phi2)
            self.phi2.append(transition)
            self.phi2.popleft()
            phi_current = self.phi2
            self.update_memory(phi_last, self.last_prediction2, reward, phi_current)


    def get_batch_indices(self):
        '''Gets a list of experiences from the replay memory. In reality,
           returns a list of indexes that are used to access the parts of
           each experience in each array.'''

        indices = np.random.randint(0, self.train_data["filled_memory"], size=cfg.batch_size)

        return indices

    def learn(self):
        '''This function is also called by the game mainloop.
           As the name suggests, it performs a gradient descent step, and also resets the
           target Q-network if needed (single DQN only).'''

        # Create the mini-batch of experiences.
        indices = self.get_batch_indices()
        states = self.states_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        transitions = self.transitions_memory[indices]

        # Set the target Q-network value to zero if the state is terminal.
        dones = np.ndarray((cfg.batch_size))
        for t in range(cfg.batch_size):
            if rewards[t] == -1:
                dones[t] = 1.0
            else:
                dones[t] = 0.0
        dones = tf.cast(tf.convert_to_tensor(dones), tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Predictions for S
            prediction_s = self.model(states, training=False)
            # Predictions for S'
            prediction_t = self.model(transitions, training=False)

            # Array of values S and S'. Masks by the "dones" array for terminal states;
            # terminal states have dones[] = 0, so the TD error is just the reward.
            value_s = tf.squeeze(prediction_s[2])
            value_t = tf.squeeze(prediction_t[2])
            # Wraps tensors in a tf.Variable so they are treated as constants by the tape.
            value_s_const = tf.Variable(value_s)
            value_t_const = tf.Variable(value_t)

            expected_value = rewards + cfg.discount * value_t_const * (1 - dones)
            td_error = expected_value - value_s_const
            critic_loss = td_error * tf.keras.losses.Huber()(expected_value, value_s)

            td_error_expanded = tf.reshape(tf.repeat(td_error, repeats=cfg.num_actions), [cfg.batch_size, cfg.num_actions])
            means_s = prediction_s[0]
            stdevs_s = prediction_s[1]
            means_s_untracked = tf.Variable(means_s)
            stdevs_s_untracked = tf.Variable(stdevs_s)

            means_eligibility_vector = means_s / means_s_untracked
            stdevs_eligibility_vector = stdevs_s / stdevs_s_untracked
            actor_means_loss = -tf.reduce_sum(td_error_expanded * means_eligibility_vector)
            actor_stdevs_loss = -tf.reduce_sum(td_error_expanded * stdevs_eligibility_vector)

            loss = actor_means_loss + actor_stdevs_loss + critic_loss

        temp1 = prediction_s[0].numpy()
        temp2 = value_s.numpy()
        temp3 = td_error.numpy()
        temp4 = critic_loss.numpy()
        temp5 = means_s.numpy()
        tempA = means_eligibility_vector.numpy()
        tempB = td_error_expanded.numpy()
        temp6 = actor_means_loss.numpy()
        temp7 = loss.numpy()

        gradients = tape.gradient(loss, self.model.trainable_variables)

        for x in gradients:
            n = tf.math.is_nan(x)
            if not tf.reduce_sum(tf.cast(n, tf.float32)) == 0:
                foo = 1

        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        # self.stats["loss"].append(loss.numpy())

        # If the learning rate is too large and causes the model to diverge to infinity, let the user know.
        if math.isnan(prediction_s[0][0][0]):
            print("NaN")
        if math.isinf(prediction_s[0][0][0]):
            print("inf")
        # ================================================================================ #

    def add_debug(self):
        self.stats["score"].append(self.game.score)
