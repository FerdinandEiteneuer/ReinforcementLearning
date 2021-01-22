"""
epsilon (exploration rate) schedulers.
TODO: make base scheduler class for all decayables, like learning rates?
"""

# external libraries
import numpy as np

# this package
from utils import export


@export
def decaying_learning_rate_scheduler(decay_factor=0.99, minimum=10**(-7)):
    def lr_scheduler(epoch, lr):
        return max(minimum, lr * decay_factor)
    return lr_scheduler

@export
class EpsilonScheduler:

    def __init__(self, eps=0.1, episodes=0, minimum=0):

        assert 0 <= minimum <= 1, f'{minimum=} must be between 0 and 1'
        self.minimum = minimum

        self.eps = eps

        self.episodes = episodes

    def _get_eps(self):
        return self._eps

    def _set_eps(self, eps):
        if 0 <= eps <= 1:
            self._eps = max(self.minimum, eps)
        else:
            raise ValueError(f'eps must be between 0 and 1, but is {eps}')

    @property
    def eps(self):
        return self._get_eps()

    @eps.setter
    def eps(self, eps):
        self._set_eps(eps)

    def __call__(self, episodes):
        raise NotImplementedError


@export
class DecayingEpsilonScheduler(EpsilonScheduler):
    def __init__(self, eps, decay_scale=10000, episodes=0, minimum=0):
        """
        After decay_scale episodes, the initial value of eps has dropped to 1/e
        """
        super().__init__(eps=eps, episodes=episodes, minimum=minimum)

        self.decay_scale = decay_scale
        self.decay_factor = 1/np.e**(1/decay_scale)
        self.eps_0 = eps

    def __call__(self, episodes=None):
        #self.episodes = episodes
        self.eps *= self.decay_factor
        return self.eps


@export
class LinearlyDecreasingEpsilonScheduler(EpsilonScheduler):
    def __init__(self, eps, end_of_decrease=40000, episodes=0, minimum=0):
        super().__init__(eps=eps, episodes=episodes, minimum=minimum)

        self.eps0 = eps
        self.end_of_decrease = end_of_decrease
        self.slope = (eps - minimum) / end_of_decrease

    def __call__(self, episodes=None):
        self.eps = max(self.minimum, self.eps - self.slope)
        return self.eps


@export
class ConstEpsilonScheduler(EpsilonScheduler):

    def __init__(self, eps=0.1):

        super().__init__(eps)

    def __call__(self, episodes=None):
        return self.eps


if __name__ == '__main__':

    cs = ConstEpsilonScheduler(0.1)

    print('\nTESTING LINEAR DECREASE SCHEDULER')

    n = 10
    ls = LinearlyDecreasingEpsilonScheduler(1, end_of_decrease=n, episodes=0, minimum=+0.5)
    for i in range(n+3):
        print(i, ls())

    print('\nTESTING DECAYING EPSILON SCHEDULER')
    starting_eps = 1
    ds = DecayingEpsilonScheduler(starting_eps, decay_scale=10, minimum=0.1)

    print(starting_eps/np.e)
    for i in range(1, 40):
        #print(f'{i=}, {ds()}')
        pass