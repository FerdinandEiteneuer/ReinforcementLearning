"""
utilities for the neural network agent
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotNormal

from . import export


@export
def create_Sequential_Dense_net1(
        layers=1,
        neurons=128,
        p_dropout=0.1,
        input_shape=(9,),
        n_outputs=9,
        lambda_regularization=10**(-4),
        ):

    net = Sequential()

    net.add(Dense(
        neurons,
        input_shape=input_shape,
        kernel_regularizer=l2(lambda_regularization),
        kernel_initializer=GlorotNormal(),
        activation='relu'),
    )

    net.add(Dropout(p_dropout))

    for n in range(layers-1):
        net.add(Dense(
            neurons,
            kernel_regularizer=l2(lambda_regularization),
            kernel_initializer=GlorotNormal(),
            activation='relu'),
        )

        net.add(Dropout(p_dropout))

    net.add(Dense(n_outputs, activation='tanh'))

    return net


@export
def save_model_on_KeyboardInterrupt(func):
    """
    Decorator intended to save the neural network on KeyboardInterrupt
    """
    def save_model(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            try:
                print('\nStopped Agent: Caught KeyboardInterrupt!')
                agent = args[0]
                path = agent.save_model_path
                agent.save_model(path=path, network='Q', overwrite=True)
            except Exception as e:
                print('saving failed:', e)
                raise
    return save_model

