"""
utilities for the neural network agent
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

def create_Dense_net1(layers=1, neurons=128, p_dropout=0.1, input_shape=(9,), n_outputs=9, lambda_regularization=10**(-4)):

    net = Sequential()

    net.add(Dense(neurons, input_shape=input_shape, kernel_regularizer=l2(lambda_regularization), activation='relu'))
    net.add(Dropout(p_dropout))

    for n in range(layers-1):
        net.add(Dense(neurons, kernel_regularizer=l2(lambda_regularization), activation = 'relu'))
        net.add(Dropout(p_dropout))

    net.add(Dense(n_outputs, activation='tanh'))

    return net
