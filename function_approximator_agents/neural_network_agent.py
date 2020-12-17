import numpy as np
import tensorflow


class NeuralNetworkAgent:

    def __init__(self, env, epsilon_scheduler, policy, gamma):

        self.Q = None
        self.env = env
        self.epsilon_scheduler = epsilon_scheduler
        self.eps = None
        self.gamma = gamma

        self.default_policy = policy
        self.set_policy(policy)

    def set_policy(self, policy='eps_greedy'):
        if policy == 'eps_greedy':
            self.policy = self.get_epsilon_greedy_action
        elif policy == 'random':
            self.policy = self.get_random_action
        elif policy == 'greedy':
            self.policy = self.get_greedy_action
        else:
            raise ValueError(f'Policy was {policy}, but must be "eps_greedy", "greedy" or "random".')

    def get_random_action(self, *args):
        """
        Samples a random action from the environment.
        """
        return self.env.action_space.sample()

    def get_epsilon_greedy_action(self, state, eps=None):
        """
        Picks the epsilon greedy action. With probability episolon,
        a random action is chosen. Otherwise, the greedy actions gets chosen.
        """

        if eps is None:
            self.eps = self.epsilon_scheduler(self.episodes)
        else:
            assert 0 <= eps <= 1
            self.eps = eps

        explore = self.eps > np.random.random()

        if explore:
            return self.get_random_action()
        else:
            return self.get_greedy_action(state)

    def predict(self, state):

        n_actions = self.env.action_space.n

        shape = (n_actions, self.Q_memory.shape[1] - 1)

        values = np.zeros(shape)

        values[:, :-1] = state
        values[:, -1] = range(n_actions)

        # if state is [12,52,21] and we have actions 0, 1, 2
        # values = [[12, 52, 21, 0],
        #           [12, 52, 21, 1],
        #           [12, 52, 21, 2]]

        qs = self.Q(values).numpy()  # use __call__ instead of predict, as its faster for smaller batch sizes
        return qs


    def analyse_maxQ(self, state):

        q = self.predict(state)

        index_max = 0
        q_max = - np.inf
        for i, s in enumerate(state):
            if s != 0:  # WARNING this means only s != is still valid move. WARNING
                continue  # only select legal actions
            if q[i] > q_max:
                q_max = q[i]
                index_max = i

        return q_max, index_max

    def get_greedy_action(self, state):
        """
        Picks the greedy action, given a state.
        """
        _, index_max = self.analyse_maxQ(state)
        return index_max

    def get_maxQ(self, state):
        """
        Selects maximum Q value for a given state under the assumption that only s != 0 are valid moves.
        """
        q_max, _ = self.analyse_maxQ(state)
        return q_max

