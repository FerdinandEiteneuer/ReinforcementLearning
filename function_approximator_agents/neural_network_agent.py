import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gym

import contextlib
from collections import deque
from functools import wraps

import utils


def return_numpy_array_on_step(env):
    _old_step = env.step

    def new_step(action):
        next_state, *args = _old_step(action)
        next_state = np.array([next_state])
        return next_state, *args
    return new_step


def return_numpy_array_on_reset(env):
    _old_reset = env.reset

    @wraps(_old_reset)
    def new_reset():
        next_state = np.array([_old_reset()])
        return next_state
    return new_reset


@utils.export
class NeuralNetworkAgent:

    def __init__(self, env,
                 epsilon_scheduler,
                 learning_rate_scheduler,
                 policy,
                 gamma,
                 self_play,
                 experience_replay,
                 size_memory,
                 save_model_dir=None,
                 success_condition=None):

        self.env = env

        self.Q_input_shape = utils.get_input_shape(env)
        self.nb_actions = utils.get_output_neurons(env)

        if type(env.observation_space) is gym.spaces.discrete.Discrete:
            env.step = return_numpy_array_on_step(env)
            env.reset = return_numpy_array_on_reset(env)

        self.env.dtype_state = np.ndarray

        self.epsilon_scheduler = epsilon_scheduler
        self.learning_rate_scheduler = learning_rate_scheduler

        self.eps = None
        self.gamma = gamma
        self.episodes = 0

        self.default_policy = policy
        self.self_play = self_play

        self.save_model_dir = save_model_dir
        if save_model_dir is not None:
            self.autosave = True
        else:
            self.autosave = False

        # neural networks
        self.Q = None
        self.Q_fixed_weights = None

        # memory
        self.experience_replay = experience_replay
        if experience_replay:

            self.memory = utils.NumpyArrayMemory(
                size=size_memory,
                input_shape=self.Q_input_shape[0],
                nb_actions=self.nb_actions,
                data_dir=self.save_model_dir
            )

        self.policy = policy

        if self_play:
            self.opponent_policy = 'greedy'
        else:
            self.opponent_policy = 'random'

        if callable(success_condition):
            self.success_condition = success_condition

    """
    POLICY
    """
    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, policy):
        if isinstance(policy, str):  # predefined policies
            if policy == 'random':
                self._policy = self.get_random_action
            elif policy == 'greedy':
                self._policy = self.get_greedy_action
            elif policy == 'eps_greedy':
                self._policy = self.get_epsilon_greedy_action
            else:
                raise ValueError(f'{policy=}, but only "greedy", "eps_greedy" or "random" are valid.')
        elif utils.is_valid_policy_function(policy):  # custom policies
            self._policy = policy
        else:
            raise ValueError(f'Object {policy=} is not a proper policy function.')

    @property
    def opponent_policy(self):
        return self._opponent_policy

    @opponent_policy.setter
    def opponent_policy(self, policy):
        if policy == 'random':
            self._opponent_policy = self.get_random_action
        elif policy == 'greedy':
            self._opponent_policy = self.get_ideal_opponent_action2
        else:
            raise ValueError(f'Opponent policy was {policy}, but must be "greedy" or "random".')


    def get_random_action(self, *args):
        """
        Samples a random action from the environment.
        """
        action = self.env.action_space.sample()
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

    """
    USING MODEL
    """
    def predict(self, state, network):
        """
        Evaluates the state for the given network.
        """
        if network == 'Q':
            model = self.Q
        elif network == 'Q_fixed_weights':
            model = self.Q_fixed_weights
        else:
            raise ValueError(f'Tried to access {network=}, but must be either \'Q\' or \'Q_fixed_weights\'')

        # we use model's __call__ instead of using model.predict(...)
        # It is faster for smaller batch sizes
        qs = model(np.expand_dims(state, axis=0)).numpy()[0]
        assert qs.shape == (self.nb_actions, ), f'{qs.shape=}, {self.nb_actions=}'
        return qs

    def predict_Q(self, state):
        return self.predict(state, 'Q')

    def predict_Q_fixed_weights(self, state):
        return self.predict(state, 'Q_fixed_weights')

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

        if np.random.uniform() < 0.1:
            return self.get_random_action()

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
            q_max, index_max = self.analyse_maxQ(s_next, 'Q')

            if q_max < min_max_Q:
                min_max_Q = q_max
                min_max_action = a

        # return environment back to its original state
        self.env.state = s_intermed
        self.env.env = env_original
        return min_max_action

    def get_ideal_opponent_action2(self, s_intermediate):
        q_min, action_min = self.analyse_min_Q(s_intermediate)
        return action_min

    def analyse_min_Q(self, s_intermediate, network='Q'):

        allowed_actions = self.get_allowed_actions(s_intermediate)
        qs = self.predict(s_intermediate, 'Q')

        action_min = 0
        q_min = + np.inf

        for action in allowed_actions:

            if qval := qs[action] < q_min:
                q_min = qval
                action_min = action

        return q_min, action_min


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

    def analyse_maxQ(self, state, network):

        qs = self.predict(state, network)
        allowed_actions = self.get_allowed_actions(state)

        index_max = 0
        q_max = - np.inf

        for i in allowed_actions:

            if qs[i] > q_max:
                q_max = qs[i]
                index_max = i

        return q_max, index_max

    def get_greedy_action(self, state):
        """
        Picks the greedy action, given a state.
        """
        _, index_max = self.analyse_maxQ(state, 'Q')
        return index_max

    def get_maxQ(self, state, network):
        """
        Selects maximum Q value for a given state under the assumption that only s != 0 are valid moves.
        """
        q_max, _ = self.analyse_maxQ(state, network)
        return q_max

    """
    SAVING MODEL
    """
    def save_model(self, network, path=None, overwrite=True, save_memory=True):
        if path is None:
            path = self.save_model_dir
        if path is None:
            raise ValueError('no path given')


        try:
            model = getattr(self, network)
        except AttributeError:
            print(f'warning: could not find {network=}, using \'Q\' instead')
            model = self.Q

        self.save_model_dir = path

        print('saving model to:', path)
        with contextlib.redirect_stderr(None):  # do not clutter output
            with contextlib.redirect_stdout(None):
                model.save(path, overwrite=overwrite)

        if save_memory:
            self.memory.save()

    def load_model(self, path, load_memory=True):

        self.Q = tf.keras.models.load_model(path)
        self.Q_fixed_weights = self.Q

        if load_memory:
            self.memory.load()

    """
    TRAINING
    """
    #@utils.save_model_on_KeyboardInterrupt
    def train_and_play(self, train=8000, play=1000, repeat=1, funcs=[]):

        for i in range(repeat):
            print(f'\ntrain/play loop #{i+1}')
            self.train(n_episodes=train)
            self.play(n_episodes=play)

            if self.autosave:
                self.save_model('Q', path=self.save_model_dir)

            for func in funcs:
                func(self)

    def train(self, n_episodes):
        """
        Train the agent n_episodes
        """
        self.policy = 'eps_greedy'

        if self.self_play:
            self.opponent_policy = 'greedy'
        else:
            self.opponent_policy = 'random'

        self._loop(n_episodes, train=True)

    def play(self, n_episodes, opponent_policy='random'):

        self.policy = 'greedy'
        self.opponent_policy = opponent_policy

        return self._loop(n_episodes, train=False)

    def _loop(self, n_episodes, train=True):
        """
        Main training or evaluation loop.
        """
        rewards_ = deque(maxlen=1000)  # last reward of episode
        total_rewards_ = deque(maxlen=1000)  # total reward of episode
        wins = 0

        ema_reward = 0
        bias_adjusted_ema_reward = 0

        mean_loss = 0

        beta = 0.996
        total_reward = 0
        total_avg_reward = 0
        successes = 0

        with tqdm(total=n_episodes, postfix='T=') as t:

            for k in range(n_episodes):

                if train:
                    info = self.train_one_episode()
                else:
                    info = self.play_one_episode()

                if self.success_condition(info):
                    successes += 1

                postfix = ''

                if reward := info.get('last_reward'):
                    rewards_.append(reward)
                    total_reward += reward

                    if reward == self.env.reward_range[-1]:  # e.g reward_range = (-1, 1), where 1 is the win
                        wins += 1

                    ema_reward = beta * ema_reward + (1 - beta) * reward
                    bias_adjusted_ema_reward = ema_reward/(1-beta**(k+1))

                postfix += f'success={100 * successes / (k + 1):.2f}%, '
                postfix += f'r={bias_adjusted_ema_reward:.2f}, '

                if 'total_reward' in info:
                    total_reward = info.get('total_reward')
                    total_rewards_.append(total_reward)
                    postfix += f'tot r={np.mean(total_rewards_):.2f}, '

                mean_loss = info.get('mean_loss') or mean_loss
                postfix += f'mean loss={mean_loss:.3f}, '

                if self.eps is not None:
                    postfix += f'eps={self.eps:.2e} '

                t.postfix = postfix
                t.update()

        try:
            success_percentage = successes / n_episodes * 100
        except TypeError:
            success_percentage = 'n/a'

        if self.self_play:
            print(f'did {n_episodes=}, {train=}, {total_reward=} (including negative rewards), win %: {success_percentage:.2f}\n')
        else:
            total_avg_reward = np.mean(total_rewards_) if len(total_rewards_) > 0 else 0
            print(f'did {n_episodes=}, {train=}, {total_avg_reward=:.2f}, success rate={success_percentage:.2f}%\n')


        return total_reward
