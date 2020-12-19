import numpy as np
import tensorflow


class NeuralNetworkAgent:

    def __init__(self, env, epsilon_scheduler, policy, gamma):

        self.env = env
        self.epsilon_scheduler = epsilon_scheduler
        self.eps = None
        self.gamma = gamma

        self.default_policy = policy

        # neural network
        self.Q = None
        self._prediction_network = 'Q'

        self.set_policy(policy)

    def set_policy(self, policy):
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
        #print('nnetagent chooses random action')
        action = self.env.action_space.sample()
        #print('nnetagent done chosing sampling random action', action)
        return action

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
        size_obs_space = self.env.observation_space.n
        shape = (n_actions, n_actions + size_obs_space)

        actions_one_hot = np.eye(n_actions)

        values = np.zeros(shape)
        values[:, :size_obs_space] = state
        values[:, size_obs_space:] = actions_one_hot
        # if state is [12,52,21] and we have actions 0, 1, 2
        # values = [[12, 52, 21, 1, 0, 0],
        #           [12, 52, 21, 0, 1, 0],
        #           [12, 52, 21, 0, 0, 1]]

        #print('\nIN PREDICT..INPUT IS', state, '\n', values)

        if self._prediction_network == 'Q':
            model = self.Q
        elif self._prediction_network == 'Q_fixed_weights':
            model = self.Q_fixed_weights
        else:
            raise ValueError(f'could not find prediction network')

        qs = model(values).numpy()  # use __call__ instead of predict, as its faster for smaller batch sizes
        #print(qs)
        return qs

    def analyse_maxQ(self, state):

        #connectx
        qs = self.predict(state)

        index_max = 0
        q_max = - np.inf
        for i, q in enumerate(qs):
            if self.env.state[i] != 0:  # WARNING this means only s != 0 is still valid move. WARNING
                continue  # only select legal actions
            if q > q_max:
                q_max = q
                index_max = i

        """
        #tictactoe
        q = self.predict(state)

        index_max = 0
        q_max = - np.inf
        for i, s in enumerate(state):
            if s != 0:  # WARNING this means only s != 0 is still valid move. WARNING
                continue  # only select legal actions
            if q[i] > q_max:
                q_max = q[i]
                index_max = i
        """
        return q_max, index_max

    def get_greedy_action(self, state):
        """
        Picks the greedy action, given a state.
        """
        _, index_max = self.analyse_maxQ(state)
        #print('player choosing to get greedy action', state, index_max)
        return index_max

    def get_maxQ(self, state):
        """
        Selects maximum Q value for a given state under the assumption that only s != 0 are valid moves.
        """
        q_max, _ = self.analyse_maxQ(state)
        return q_max

