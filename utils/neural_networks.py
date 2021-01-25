"""
utilities for the neural network agent
"""
import contextlib
from functools import wraps
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotNormal

from utils import export


@export
def get_input_shape(env):
    if type(env.observation_space) is gym.spaces.discrete.Discrete:
        #input_shape = (env.observation_space.n,)
        input_shape = (1,)
    elif type(env.observation_space) is gym.spaces.tuple.Tuple:
        input_shape = (len(env.observation_space),)
    elif type(env.observation_space) is gym.spaces.box.Box:
        input_shape = env.observation_space.shape
    else:
        raise TypeError(f'environment {env} is not supported.')
    return input_shape


@export
def get_output_neurons(env):
    n_outputs = env.action_space.n
    return n_outputs


@export
def create_cartpole_network(hidden_layers=2, neurons=56):
    """
    Network that can solve gyms 'CartPole-v1' environment.
    """

    net = Sequential()

    net.add(Dense(
        neurons,
        input_shape=(4,),
        kernel_regularizer=l2(0.001),
        kernel_initializer=GlorotNormal(),
        activation='relu'),
    )

    net.add(Dropout(0.1))

    for n in range(hidden_layers):
        net.add(Dense(
            neurons,
            kernel_regularizer=l2(0.001),
            kernel_initializer=GlorotNormal(),
            activation='relu'),
        )

        net.add(Dropout(0.1))

    net.add(Dense(2, activation='relu'))

    return net


@export
def create_sequential_dense_net(
        hidden_layers=1,
        neurons_per_layer=128,
        p_dropout=0.1,
        input_shape=(9,),
        n_outputs=9,
        lambda_regularization=10**(-4),
        hidden_activation_function='relu',
        final_activation_function='relu',
    ):

    net = Sequential()

    net.add(Dense(
        neurons_per_layer,
        input_shape=input_shape,
        kernel_regularizer=l2(lambda_regularization),
        kernel_initializer=GlorotNormal(),
        activation='relu'),
    )

    net.add(Dropout(p_dropout))

    for n in range(hidden_layers):
        net.add(Dense(
            neurons_per_layer,
            kernel_regularizer=l2(lambda_regularization),
            kernel_initializer=GlorotNormal(),
            activation=hidden_activation_function),
        )

        net.add(Dropout(p_dropout))

    net.add(Dense(n_outputs, activation=final_activation_function))

    return net


@export
def save_model_on_KeyboardInterrupt(func):
    """
    Decorator intended to save the neural network on KeyboardInterrupt
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            try:
                print('\nSaving Agent: Caught KeyboardInterrupt!')
                agent = args[0]
                path = agent.save_model_path
                agent.save_model(path=path, network='Q', overwrite=True)
            except Exception as e:
                print('saving failed:', e)
                raise

    return wrapper

