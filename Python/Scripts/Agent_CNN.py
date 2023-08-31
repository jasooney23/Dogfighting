'''Agent v2: uses '''

# Import packages.
import queue
import random
import math
import time
import tensorflow as tf

# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
from . import config as cfg
from . import Memory_CNN as mem


class agent:
    '''This creates an agent that plays and learns to dogfight. All of the functions
       used for training are contained in this class.'''

    def __init__(self):
        self.model = mem.load_models()
        self.train_data, self.stats = mem.load_data()
        self.state_img_memory, self.state_inst_memory, self.action_memory, self.reward_memory, self.transition_img_memory, self.transition_inst_memory = mem.load_memory()

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

        self.last_actions1, self.last_actions2 = (None, None)

    def save_all(self, path):
        mem.save_models(path, self.model)
        mem.save_data(path, self.train_data, self.stats)
        mem.save_memory(path, self.state_img_memory, self.state_inst_memory, self.action_memory,
                        self.reward_memory, self.transition_img_memory, self.transition_inst_memory)


    def update_memory(self, state, action, reward, transition):
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

        index = self.train_data["update_index"]

        # Insert the experience into replay memory.
        self.state_img_memory[index] = state[0]
        self.state_inst_memory[index] = state[1]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.transition_img_memory[index] = transition[0]
        self.transition_inst_memory[index] = transition[1]

        self.train_data["update_index"] += 1
        # Keep track of how much of the memory has been filled.
        if self.train_data["filled_memory"] < cfg.memory_size:
            self.train_data["filled_memory"] += 1

    def stack_phi(self, phi, expand=False):
        imgs = []
        datas = []
        for x in range(len(phi)):
            imgs.append(phi[x][0] / 255)
            datas.append(phi[x][1])

        imgs = np.array(imgs)
        datas = np.array(datas)
        if expand:
            return (np.expand_dims(imgs, axis=0), np.expand_dims(datas, axis=0))
        else:
            return (imgs, datas)

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
            stack = self.stack_phi(self.phi1, expand=True)
        elif plane_number == 2:
            stack = self.stack_phi(self.phi2, expand=True)

        # Determine epsilon
        explore_count_percent = self.train_data["frames_played"] / \
            cfg.explore_count
        if explore_count_percent > 1:
            explore_count_percent = 1
        current_epsilon = cfg.start_epsilon - \
            ((cfg.start_epsilon - cfg.end_epsilon) * explore_count_percent)
        
        if np.random.default_rng().uniform(0, 1) <= current_epsilon and not cfg.performance_mode:
            actions = np.random.default_rng().uniform(-1, 1, size=cfg.num_actions)
            means = [0, 0, 0, 0, 0]
            value = (10000000,)
        else:
            # Get mean and stdev values from model. Ignores value function.
            prediction = self.model(stack, training=False)
            means = prediction[0].numpy()[0]
            stdevs = prediction[1].numpy()[0]
            value = prediction[2].numpy()[0]
            actions = self.actions_from_gaussians(means, stdevs)

        if plane_number == 1:
            self.last_actions1 = actions
            # print(value)
        elif plane_number == 2:
            self.last_actions2 = actions


        if math.isnan(actions[0]):
            foo = 1
            print("prediction is NaN")
            pass

        return actions, means, value

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
            self.update_memory(self.stack_phi(phi_last), self.last_actions1, reward, self.stack_phi(phi_current))
        elif plane_number == 2:
            phi_last = list(self.phi2)
            self.phi2.append(transition)
            self.phi2.popleft()
            phi_current = self.phi2
            self.update_memory(self.stack_phi(phi_last), self.last_actions2, reward, self.stack_phi(phi_current))

    def clip_gradients(self, gradients):
        grad_list = []
        for g in gradients:
            if not g == None:
                grad_list.append(tf.clip_by_value(g, -1, 1))
            else:
                grad_list.append(None)
        return grad_list

    def get_batch_indices(self):
        '''Gets a list of experiences from the replay memory. In reality,
           returns a list of indexes that are used to access the parts of
           each experience in each array.'''

        indices = np.random.randint(0, self.train_data["filled_memory"], size=int(cfg.batch_size * (1 - cfg.priority_ratio)))

        kill_indices = np.where(self.reward_memory == cfg.reward_kill)[0]
        if kill_indices.shape[0] == 0:
            return indices
        else:
            kill_indices = np.random.default_rng().choice(kill_indices, size=int(cfg.batch_size * cfg.priority_ratio))
            return np.concatenate((indices, kill_indices))

    def learn(self):
        '''This function is also called by the game mainloop.
           As the name suggests, it performs a gradient descent step, and also resets the
           target Q-network if needed (single DQN only).'''

        # Create the mini-batch of experiences.
        indices = self.get_batch_indices()
        batch_size = indices.shape[0]

        state_imgs = self.state_img_memory[indices]
        state_insts = self.state_inst_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        transition_imgs = self.transition_img_memory[indices]
        transition_insts = self.transition_inst_memory[indices]

        # Set the target Q-network value to zero if the state is terminal.
        dones = np.ndarray((batch_size))
        for t in range(batch_size):
            if rewards[t] == -1:
                dones[t] = 1.0
            else:
                dones[t] = 0.0
        dones = tf.cast(tf.convert_to_tensor(dones), tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Predictions for S
            prediction_s = self.model((state_imgs / 255, state_insts), training=False)
            # Predictions for S'
            prediction_t = self.model((transition_imgs / 255, transition_insts), training=False)

            # Array of values S and S'. Masks by the "dones" array for terminal states;
            # terminal states have dones[] = 0, so the TD error is just the reward.
            value_s = tf.squeeze(prediction_s[2])
            value_t = tf.squeeze(prediction_t[2])
            # Wraps tensors in a tf.Variable so they are treated as constants by the tape.
            value_s_const = tf.Variable(value_s)    
            value_t_const = tf.Variable(value_t)

            expected_value = rewards + cfg.discount * value_t_const * (1 - dones)
            # this is the ACTUAL critic loss. The model aims to minimize THIS.
            td_error = expected_value - value_s_const

            # This little bit here is just so the proper gradients can be calculated. Negative sign
            #   for gradient ascent instead of descent (with Adam optimizer)
            # critic_loss = tf.keras.losses.Huber()(expected_value, value_s)
            critic_update = -td_error * value_s

            td_error_expanded = tf.reshape(tf.repeat(td_error, repeats=cfg.num_actions), [batch_size, cfg.num_actions])
            means_s = prediction_s[0]
            stdevs_s = prediction_s[1]
            means_s_untracked = tf.Variable(means_s)
            stdevs_s_untracked = tf.Variable(stdevs_s)

            # means_eligibility_vector = (actions - means_s_untracked) / tf.square(stdevs_s_untracked)
            # stdevs_eligibility_vector = tf.square((actions - means_s_untracked) / stdevs_s_untracked) - 1
            
            probability_density = tf.math.exp(-tf.math.square((actions - means_s) / stdevs_s) / 2) / (stdevs_s * tf.math.sqrt(2 * np.pi))
            
            
            # n = tf.math.is_nan(probability_density)
            # if not tf.math.reduce_sum(tf.cast(n, tf.float32)) == 0:
            #     foo = 1
            
            # threshold = tf.math.pow(10., -3)
            # temp1 = probability_density.numpy()
            # probability_density = tf.where(tf.math.less(probability_density, threshold), tf.ones_like(probability_density) * threshold, probability_density)
            # temp2 = probability_density.numpy()

            # means_eligibility_vector = tf.math.log(means_s)
            # stdevs_eligibility_vector = tf.math.log(stdevs_s)
            # actor_means_loss = -tf.math.reduce_sum(td_error_expanded * means_eligibility_vector)
            # actor_stdevs_loss = -tf.math.reduce_sum(td_error_expanded * stdevs_eligibility_vector)

            # ========================= #

            # eligibility_vector = tf.math.log(probability_density + 0.0001)
            # actor_update = -td_error_expanded * eligibility_vector

            means_elegibility_vector = (means_s * actions - tf.square(means_s) / 2) / tf.square(stdevs_s_untracked)
            stdevs_elegibility_vector = -(tf.square(actions - means_s_untracked) / stdevs_s) - stdevs_s

            actor_update = -td_error_expanded * (means_elegibility_vector + stdevs_elegibility_vector)

            update = tf.reduce_sum(critic_update) + tf.reduce_sum(actor_update)

            # ========================= #

            self.stats["performance_measure"].append(tf.math.reduce_mean(actor_update).numpy())
            self.stats["td_error"].append(tf.math.reduce_mean(td_error).numpy())



        '''=========================================='''


        # with tf.GradientTape(persistent=True) as tape:
        # state      = tf.repeat(((-0.09865861, 0.1, 0.0, 0.0, 0.00, 0.0, 0.06739839, 1.0, 0.0,
        #                                0.09865861, 0.1, 0.0, 0.0, 0.18, 0.0, 0.06739839, 1.0, 0.0),), cfg.stack_size, axis=0)
        #     transition = tf.repeat(((-0.09730333, 0.1, 0.0, 0.0, 0.00, 0.0, 0.06816988, 1.0, 0.0,
        #                               0.09730333, 0.1, 0.0, 0.0, 0.18, 0.0, 0.06816988, 1.0, 0.0),), cfg.stack_size, axis=0)
        #     action = tf.Variable((1.0, 0.0, 0.0, 0.0))
        #     reward = tf.Variable((-1.0,))

        #     prediction_s = self.model(tf.expand_dims(state, axis=0), training=False)
        #     prediction_t = self.model(tf.expand_dims(transition, axis=0), training=False)
        #     value_s = tf.squeeze(prediction_s[2])
        #     value_t = tf.squeeze(prediction_t[2])
        #     value_s_const = tf.Variable(value_s)    
        #     value_t_const = tf.Variable(value_t)
        #     expected_value = reward + cfg.discount * value_t_const
        #     td_error = expected_value - value_s_const
        #     critic_update = -td_error * value_s

        #     td_error_expanded = tf.reshape(tf.repeat(td_error, repeats=cfg.num_actions), [1, cfg.num_actions])
        #     means_s = prediction_s[0]
        #     stdevs_s = prediction_s[1]
        #     means_s_untracked = tf.Variable(means_s)
        #     stdevs_s_untracked = tf.Variable(stdevs_s)

        #     x = means_s.numpy()
        #     y = stdevs_s.numpy()

        #     # probability_density = tf.math.exp(-tf.math.square((action - means_s) / stdevs_s) / 2) / (stdevs_s * tf.math.sqrt(2 * np.pi))
        #     # eligibility_vector = tf.math.log(probability_density + 0.0001)

        #     means_elegibility_vector = (means_s * action - tf.square(means_s) / 2) / tf.square(stdevs_s_untracked)
        #     stdevs_elegibility_vector = -(tf.square(action - means_s_untracked) / stdevs_s) - stdevs_s

        #     means_update = -td_error_expanded * means_elegibility_vector
        #     stdevs_update = -td_error_expanded * stdevs_elegibility_vector
        #     actor_update = -td_error_expanded * (means_elegibility_vector + stdevs_elegibility_vector)

        #     # a = probability_density.numpy()
        #     # b = eligibility_vector.numpy()
        #     d = critic_update.numpy()
        #     c = actor_update.numpy()


        '''=========================================='''

        # tempx = gradients[0].numpy()


        tempD = actions
        asiodgfoq = value_s.numpy()
        asiodgfoq1 = value_t.numpy()
        gadfahp = expected_value.numpy()

        temp5 = means_s.numpy()
        tempC = stdevs_s.numpy()
        tempB = means_elegibility_vector.numpy()
        tempZ = stdevs_elegibility_vector.numpy()
        temp7 = actor_update.numpy()

        gradients = tape.gradient(update, self.model.trainable_variables)
        gradients = self.clip_gradients(gradients)

        # critic_grads = tape.gradient(critic_update, self.model.trainable_variables)
        # actor_grads = tape.gradient(actor_update, self.model.trainable_variables)
        # means_grads = tape.gradient(means_update, self.model.trainable_variables)
        # stdevs_grads = tape.gradient(stdevs_update, self.model.trainable_variables)

        # for x in actor_grads:
        #     if not x == None:
        #         n = tf.math.is_nan(x)
        #         if not tf.math.reduce_sum(tf.cast(n, tf.float32)) == 0:
        #             foo = 1
        # for x in critic_grads:
        #     if not x == None:
        #         n = tf.math.is_nan(x)
        #         if not tf.math.reduce_sum(tf.cast(n, tf.float32)) == 0:
        #             foo = 1

        # a = actor_grads[0].numpy()
        # critic_grads = self.clip_gradients(critic_grads)
        # actor_grads = self.clip_gradients(actor_grads)
        # b = actor_grads[0].numpy()

        # means_grads = tf.clip_by_global_norm(means_grads, 1)[0]
        # stdevs_grads = tf.clip_by_global_norm(stdevs_grads, 1)[0]
        # tempy = gradients[0].numpy()


        # aa = self.model(tf.expand_dims(state, axis=0), training=False)
        # print(f"Means: {aa[0][0]}")
        # print(f"StDevs: {aa[1][0]}")
        # print(f"Value: {aa[2][0]}")
        # print()

        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        # self.model.optimizer.apply_gradients(
        #     zip(critic_grads, self.model.trainable_variables))
        # self.model.optimizer.apply_gradients(
        #     zip(actor_grads, self.model.trainable_variables))
        
        # self.model.optimizer.apply_gradients(
        #     zip(means_grads, self.model.trainable_variables))
        # self.model.optimizer.apply_gradients(
        #     zip(stdevs_grads, self.model.trainable_variables))
        
        # bb = self.model(tf.expand_dims(state, axis=0), training=False)
        # print(f"Means delta: {bb[0][0] - aa[0][0]}")
        # print(f"StDevs delta: {bb[1][0] - aa[1][0]}")
        # print(f"Value delta: {bb[2][0] - aa[2][0]}")

        # for x in bb:
        #     n = tf.math.is_nan(x)
        #     if not tf.math.reduce_sum(tf.cast(n, tf.float32)) == 0:
        #         foo = 1

        # self.stats["loss"].append(loss.numpy())

        # If the learning rate is too large and causes the model to diverge to infinity, let the user know.
        if math.isnan(prediction_s[0][0][0]):
            print("NaN")
        if math.isinf(prediction_s[0][0][0]):
            print("inf")
        # ================================================================================ #

    def add_debug(self):
        self.stats["score"].append(self.game.score)
