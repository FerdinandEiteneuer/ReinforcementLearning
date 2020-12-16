import numpy as np
import tensorflow


class NeuralNetworkAgent:

    def __init__(self, env, epsilon_scheduler, policy, gamma):

        self.Q = None
        self.env = env
        self.epsilon_scheduler = epsilon_scheduler
        self.gamma = gamma

        # Set policy
        if policy == 'eps_greedy':
            self.policy = self.get_epsilon_greedy_action
        elif policy == 'random':
            self.policy = self.get_random_action
        elif policy == 'greedy':
            self.policy = self.get_greedy_action


    def get_random_action(self, state=None):
        """
        Samples a random action from the environment.
        """
        return self.env.action_space.sample()

    def get_epsilon_greedy_action(self, state):
        """
        Picks the epsilon greedy action. With probability episolon,
        a random action is chosen. Otherwise, the greedy actions gets chosen.
        """

        self.eps = self.epsilon_scheduler(self.episodes)

        explore = self.eps > np.random.random()

        if explore:
            return self.get_random_action()
        else:
            return self.get_greedy_action(state)

    def analyse_maxQ(self, state):

        q = self.Q.predict(state.reshape(1, 9))[0]
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

