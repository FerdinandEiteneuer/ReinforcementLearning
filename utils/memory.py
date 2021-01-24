"""
Memory
"""
import numpy as np
import os

from utils import export


@export
class NumpyArrayMemory:
    """
    Datastructure for all the experiences (states, actions, rewards, next_states)
    the agent saw.
    """
    def __init__(self, size, input_shape, nb_actions, data_dir):

        self.data_dir = data_dir
        self.memory_path = os.path.join(data_dir, 'memory.npy')

        self.size = size
        self.input_shape = input_shape
        self.nb_actions = nb_actions

        shape_mem = size, input_shape + nb_actions
        self.memory = np.zeros(shape_mem)

        self.add_index = 0

    def add(self, states, qvalues):

        idx = self.add_index % self.size
        data = list(states) + list(qvalues)

        self.memory[idx] = data
        self.add_index += 1

    def save(self, path=None):

        if path is None:
            path = self.memory_path

        np.save(file=path, arr=self.memory)

    def load(self, path=None):

        if path is None:
            path = self.memory_path

        try:
            self.memory = np.load(path)
        except FileNotFoundError as e:
            print(f'Memory could not be loaded: {e}')

    def ready(self):
        """
        Does the memory still need to be filled up? Can Training begin?
        Not very reliable implementation, but it will do.
        """
        assert self.memory is not None
        return np.any(self.memory[-1] != 0)

    def complete_training_data(self):
        """
        Prepares the data in a format used for keras.
        """
        # one Q_memory row consists of [state, Qvalues(dim=nb_actions)]
        states = self.memory[:, :-self.nb_actions]
        targets = self.memory[:, -self.nb_actions:]

        return states, targets

    def get_batch(self, batch_size):
        raise NotImplementedError
        """
        deprecated
        memory = self.Q_memory[:self.episodes]
        batch_size = max(1, min(self.batch_size, self.e))

        if self.episodes > self.size_Q_memory:  # memory is filled
            indices = np.random.choice(range(self.size_Q_memory), self.batch_size)
            x_train = self.Q_memory[:, :-1][indices]
            y_train = self.Q_memory[:, -1][indices]
            return x_train, y_train, True

        else:  # memory is too small
            return None, None, False
        """
