'''F1 Agent Memory v1: Double DQN'''

# Import packages.
import pickle
from pickletools import optimize

import tensorflow as tf     # Tensorflow.
from keras import backend as K
from tensorflow.keras import layers

# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
from . import config as cfg

# =================================================================== #

def offset_sigmoid(x):
    return K.sigmoid(x) * 0.99 + 0.01

def make_model(name):
    '''Creates a tf.keras.Sequential model with numerous hidden layers.'''

    # The input tensor to the model.
    input_size = (cfg.stack_size, cfg.state_size)

    input_layer = tf.keras.Input(shape=input_size, name="input")
    x = layers.Flatten(name="flatten")(input_layer)
    x = layers.Dense(units=32, activation="relu", name="dense_1")(x)
    x = layers.Dense(units=64, activation="relu", name="dense_2")(x)
    x = layers.Dense(units=32, activation="relu", name="dense_3")(x)
    means_output = layers.Dense(units=cfg.num_actions, activation="tanh", name="actor_means")(x)
    stdevs_output = layers.Dense(units=cfg.num_actions, activation=offset_sigmoid, name="actor_stdevs")(x)
    value_output = layers.Dense(units=1, name="critic_value")(x)

    model = tf.keras.Model(input_layer, [means_output, stdevs_output, value_output])

    # model = tf.keras.Sequential(name=name)

    # # The network. Hidden layers were determined with this source: 
    # # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # model.add(tf.keras.layers.InputLayer(input_shape=input_size))
    # model.add(tf.keras.layers.Flatten())
    
    # model.add(tf.keras.layers.Dense(units=64, activation="relu"))
    # model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    # model.add(tf.keras.layers.Dense(units=64, activation="relu"))

    # # 5 outputs for 5 different actions.
    # model.add(tf.keras.layers.Dense(units=(2 * cfg.num_actions + 1)))

    print("Total VRAM used after creating " + name + ": " + str(tf.config.experimental.get_memory_info('GPU:0')["current"]/(10**9)) + " GB")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
                loss=tf.keras.losses.Huber(), run_eagerly=True)
    model.summary()
    tf.keras.utils.plot_model(model, to_file="model graph.png")

    return model


def make_data():
    # Create the replay memory if non-existent.
    train_data = {
        "update_index": 0,     # Which index of the replay memory to update.
        "filled_memory": 0,    # The amount of experiences in the memory.
        "reset_time": 0,      # The amount of seconds that have passed since the last target model reset.
        "frames_played": 0,
    }

    # Stats.
    stats = {
        "score": [0],     # Score per episode.
        "td_error": [0],      # Loss per frame.
        "performance_measure": [0],
        "testing": [0],
        "return": [0],
        "pitch_mean": [0],
    }

    return train_data, stats


def make_memory():

    # Replay Memory.
    states_memory = np.ndarray((cfg.memory_size, cfg.stack_size, cfg.state_size), dtype=cfg.memtype)
    action_memory = np.ndarray((cfg.memory_size, cfg.num_actions), dtype=cfg.memtype)
    reward_memory = np.ndarray((cfg.memory_size), dtype=cfg.memtype)
    transitions_memory = np.ndarray((cfg.memory_size, cfg.stack_size, cfg.state_size), dtype=cfg.memtype)

    return states_memory, action_memory, reward_memory, transitions_memory

# =================================================================== #


def load_models():
    try:
        # Try to load a saved model.
        model = tf.keras.models.load_model(cfg.save_path + "/MODEL")
        tf.keras.backend.set_value(model.optimizer.learning_rate, cfg.learning_rate)
    except Exception as e:
        print(e)
        try:
            # Try to load a saved model.
            model = tf.keras.models.load_model(cfg.backup_path + "/MODEL")
            tf.keras.backend.set_value(model.optimizer.learning_rate, cfg.learning_rate)
        except Exception as e:
            print(e)
            model = make_model("MODEL")

    return model


def load_data():
    try:
        # Try to load replay memory and statistics.
        with open(cfg.save_path + "/train_data.dat", "rb") as openfile:
            train_data = pickle.load(openfile)
        with open(cfg.save_path + "/stats.dat", "rb") as openfile:
            stats = pickle.load(openfile)
    except Exception as e:
        print(e)
        try:
            # Try to load replay memory and statistics.
            with open(cfg.backup_path + "/train_data.dat", "rb") as openfile:
                train_data = pickle.load(openfile)
            with open(cfg.backup_path + "/stats.dat", "rb") as openfile:
                stats = pickle.load(openfile)
        except Exception as e:
            print(e)
            train_data, stats = make_data()

    return train_data, stats


def load_memory():
    try:
        # Replay memory isn't stored using Pickle because it uses too much RAM.
        states_memory = np.load(cfg.save_path + "/states_memory.npy")
        action_memory = np.load(cfg.save_path + "/action_memory.npy")
        reward_memory = np.load(cfg.save_path + "/reward_memory.npy")
        transitions_memory = np.load(cfg.save_path + "/transitions_memory.npy")
    except Exception as e:
        print(e)
        try:
            # Replay memory isn't stored using Pickle because it uses too much RAM.
            states_memory = np.load(cfg.backup_path + "/states_memory.npy")
            action_memory = np.load(cfg.backup_path + "/action_memory.npy")
            reward_memory = np.load(cfg.backup_path + "/reward_memory.npy")
            transitions_memory = np.load(cfg.backup_path + "/transitions_memory.npy")
        except Exception as e:
            print(e)
            states_memory, action_memory, reward_memory, transitions_memory = make_memory()

    return states_memory, action_memory, reward_memory, transitions_memory

# =================================================================== #


def save_models(path, model):
    model.save(path + "/MODEL", overwrite=True, include_optimizer=True)


def save_data(path, train_data, stats):
    with open(path + "/train_data.dat", "wb") as openfile:
        pickle.dump(train_data, openfile)
    with open(path + "/stats.dat", "wb") as openfile:
        pickle.dump(stats, openfile)


def save_memory(path, states_memory, action_memory, reward_memory, transitions_memory):
    np.save(path + "/states_memory", states_memory)
    np.save(path + "/action_memory", action_memory)
    np.save(path + "/reward_memory", reward_memory)
    np.save(path + "/transitions_memory", transitions_memory)
