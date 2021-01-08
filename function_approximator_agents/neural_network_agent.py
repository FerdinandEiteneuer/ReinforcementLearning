import numpy as np
import tensorflow
import gym

class NeuralNetworkAgent:

    def __init__(self, env, epsilon_scheduler, policy, gamma, self_play):

        self.env = env
        self.env.dtype_state = np.ndarray

        self.epsilon_scheduler = epsilon_scheduler
        self.eps = None
        self.gamma = gamma

        self.default_policy = policy
        self.self_play = self_play

        # neural network
        self.Q = None
        self._prediction_network = 'Q'

        self.set_policy(policy)

        if self_play:
            self.set_opponent_policy('greedy')

    def train_and_play(self, train=8000, play=1000, repeat=1, funcs=[]):
        for i in range(repeat):
            print(f'\ntrain/play loop #{i+1}')
            self.train(n_episodes=train)
            self.play(n_episodes=play)

            for func in funcs:
                func(self)

    def set_opponent_policy(self, policy):
        if policy == 'random':
            self.opponent_policy = self.get_random_action
        elif policy == 'greedy':
            self.opponent_policy = self.get_ideal_opponent_action
        else:
            raise ValueError(f'Opponent policy was {policy}, but must be "greedy" or "random".')

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

        if isinstance(self.env.observation_space, gym.spaces.tuple.Tuple):
            size_obs_space = len(self.env.observation_space)
        elif isinstance(self.env.observation_space, gym.spaces.discrete.Discrete):
            size_obs_space = self.env.observation_space.n
        else:
            raise NotImplementedError('do not understand environment.')

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

    def get_ideal_opponent_action(self, s_intermediate):
        """
        A function taking a state, returning a function which returns the optimal opponent action.
        After the environment responded to the selected player action, it is
        in an intermediary state s_intermed

        1. get all available opponent actions from s_intermed
           make a list q_values
        2. for all these potential opponent actions:
            collect s_next if the opponent would take that action.
            from s_next, get Q_max = max a' Q(s_next, a')
            insert Q_max in q_values
        3. find minimum of q_values
        4. the action responding to this minimum (min-maxxed) will be the one the opponent actually takes

        """
        #return self.get_random_action()

        # make a backup of environment state
        #env_original = self.env.env.state
        env_original = self.env.env.clone()

        s_intermed = np.copy(self.env.state)
        original_env_state = self.env.env.state

        possible_opponent_actions = self.get_allowed_actions(s_intermed)

        min_max_Q = np.inf
        min_max_action = None

        for a in possible_opponent_actions:

            # NOTE: if the whole thing here works, replace by env.state = s_entermed, that should work too and is probably 100 times faster

            #self.env.env = env_original.clone()
            self.env.env.state = original_env_state

            s_next, _, _, _ = self.env.execute_opponent_action(a)

            # next thing to improve: just put all s_next in one array and run predict only once
            q_max, index_max = self.analyse_maxQ(s_next)

            if q_max < min_max_Q:
                min_max_Q = q_max
                min_max_action = a

        # return environment back to its original state
        self.env.state = s_intermed
        self.env.env = env_original
        return min_max_action

    def get_allowed_actions(self, state=None):
        """
        Returns all allowed actions.
        If no state is given, the current state of the environment is used.
        """
        try:
            allowed_actions = self.env.get_allowed_actions(state=state)
        except AttributeError as e:
            allowed_actions = range(self.env.action_space.n)

        return allowed_actions

    def analyse_maxQ(self, state):

        #connectx
        qs = self.predict(state)
        allowed_actions = self.get_allowed_actions(state)

        index_max = 0
        q_max = - np.inf
        for i, q in enumerate(qs):

            if i not in allowed_actions:
                continue  # only select legal actions
            if q > q_max:
                q_max = q
                index_max = i

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



