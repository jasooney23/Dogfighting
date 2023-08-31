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
    img_input = tf.keras.Input(shape=(cfg.stack_size, cfg.height, cfg.width), name="img_input")
    data_input = tf.keras.Input(shape=(cfg.stack_size, cfg.state_size), name="data_input")

    x = layers.Conv2D(filters=16, kernel_size=7, strides=4, padding="same", activation="relu", data_format="channels_first", name="conv2d_1")(img_input)
    x = layers.Conv2D(filters=32, kernel_size=4, strides=2, padding="same", activation="relu", data_format="channels_first", name="conv2d_2")(x)

    x = layers.Flatten(name="conv_flatten")(x)
    y = layers.Flatten(name="data_flatten")(data_input)

    x = layers.Dense(units=32, activation="relu", name="dense_1")(layers.concatenate((x, y)))
    x = layers.Dense(units=32, activation="relu", name="dense_2")(x)
    means_output = layers.Dense(units=cfg.num_actions, activation="tanh", name="actor_means")(x)
    stdevs_output = layers.Dense(units=cfg.num_actions, activation=offset_sigmoid, name="actor_stdevs")(x)
    value_output = layers.Dense(units=1, name="critic_value")(x)

    model = tf.keras.Model(inputs=(img_input, data_input), outputs=[means_output, stdevs_output, value_output])

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
    state_img_memory = np.ndarray((cfg.memory_size, cfg.stack_size, cfg.height, cfg.width), dtype=cfg.cam_memtype)
    state_inst_memory = np.ndarray((cfg.memory_size, cfg.stack_size, cfg.state_size), dtype=cfg.memtype)
    action_memory = np.ndarray((cfg.memory_size, cfg.num_actions), dtype=cfg.memtype)
    reward_memory = np.ndarray((cfg.memory_size), dtype=cfg.memtype)
    transition_img_memory = np.ndarray((cfg.memory_size, cfg.stack_size, cfg.height, cfg.width), dtype=cfg.cam_memtype)
    transition_inst_memory = np.ndarray((cfg.memory_size, cfg.stack_size, cfg.state_size), dtype=cfg.memtype)

    return state_img_memory, state_inst_memory, action_memory, reward_memory, transition_img_memory, transition_inst_memory

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
        state_img_memory = np.load(cfg.save_path + "/state_img_memory.npy")
        state_inst_memory = np.load(cfg.save_path + "/state_inst_memory.npy")
        action_memory = np.load(cfg.save_path + "/action_memory.npy")
        reward_memory = np.load(cfg.save_path + "/reward_memory.npy")
        transition_img_memory = np.load(cfg.save_path + "/transition_img_memory.npy")
        transition_inst_memory = np.load(cfg.save_path + "/transition_inst_memory.npy")
    except Exception as e:
        print(e)
        try:
            # Replay memory isn't stored using Pickle because it uses too much RAM.
            state_img_memory = np.load(cfg.backup_path + "/state_img_memory.npy")
            state_inst_memory = np.load(cfg.backup_path + "/state_inst_memory.npy")
            action_memory = np.load(cfg.backup_path + "/action_memory.npy")
            reward_memory = np.load(cfg.backup_path + "/reward_memory.npy")
            transition_img_memory = np.load(cfg.backup_path + "/transition_img_memory.npy")
            transition_inst_memory = np.load(cfg.backup_path + "/transition_inst_memory.npy")
        except Exception as e:
            print(e)
            state_img_memory, state_inst_memory, action_memory, reward_memory, transition_img_memory, transition_inst_memory = make_memory()

    return state_img_memory, state_inst_memory, action_memory, reward_memory, transition_img_memory, transition_inst_memory

# =================================================================== #


def save_models(path, model):
    model.save(path + "/MODEL", overwrite=True, include_optimizer=True)


def save_data(path, train_data, stats):
    with open(path + "/train_data.dat", "wb") as openfile:
        pickle.dump(train_data, openfile)
    with open(path + "/stats.dat", "wb") as openfile:
        pickle.dump(stats, openfile)


def save_memory(path, state_img_memory, state_inst_memory, action_memory, reward_memory, transition_img_memory, transition_inst_memory):
    np.save(path + "/state_img_memory", state_img_memory)
    np.save(path + "/state_inst_memory", state_inst_memory)
    np.save(path + "/action_memory", action_memory)
    np.save(path + "/reward_memory", reward_memory)
    np.save(path + "/transition_img_memory", transition_img_memory)
    np.save(path + "/transition_inst_memory", transition_inst_memory)
