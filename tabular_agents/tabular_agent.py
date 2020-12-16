import numpy as np
import gym
from collections import deque
import itertools
from inspect import signature
from tqdm import tqdm

class TabularAgent:

    def __init__(self, env, gamma, q_value_initialization,
                 epsilon_scheduler, alpha_scheduler=None, policy=None):

        self.env = env
        self.gamma = gamma
        self.alpha = 0.1
        self.print_every = 0

        dims_obs_space = None

        if type(env.observation_space) is gym.spaces.discrete.Discrete:

            dims_obs_space = [env.observation_space.n]
            self.S = range(dims_obs_space[0])

        elif type(env.observation_space) is gym.spaces.tuple.Tuple:

            dims_obs_space = [obs_space.n for obs_space in env.observation_space]
            ranges = [list(range(dim)) for dim in dims_obs_space]
            self.S = [state for state in itertools.product(*ranges)]

            if hasattr(env, 'state_offset') and env.state_offset != 0:
                for i, state in enumerate(self.S):
                    corrected_state = tuple(idx + env.state_offset for idx in state)
                    self.S[i] = corrected_state

        else:

            raise NotImplementedError('Not Implemented')

        dims_action_space = [env.action_space.n]
        self.dims_obs_action_space = dims_obs_space + dims_action_space

        self.A = range(dims_action_space[0])

        self.observation_action_space = list(itertools.product(self.S, self.A))

        self.Q = {}
        self.N = {}

        # terminal state
        # for s in self.S:
        #     self.Q[s, 'terminal'] = 0
        self.Q['terminal', 'terminal'] = 0

        self.train_statistic = {'wins': 0, 'plays': 0}
        self.episode_info = {'T': None, 'total reward': None, 'wins': 0}  # to be updated for every episode

        # Initialize Q-Values
        if q_value_initialization == 'random':
            self.q_value_init = lambda s,a : np.random.uniform(0, 1)  # random numbers in [0,1]
        elif len(signature(q_value_initialization).parameters) != 2:
            raise TypeError('Q value initialization function must have 2 parameters: state + action.')
        else:
            self.q_value_init = q_value_initialization

        # Set epsilon scheduler
        if len(signature(epsilon_scheduler).parameters) != 1:
            raise TypeError('epsilon scheduler initialization function must have 1 parameter: episode.')
        else:
            self.epsilon_scheduler = epsilon_scheduler

        # Set policy
        if policy == 'eps_greedy':
            self.policy = self.get_epsilon_greedy_action
        elif policy == 'random':
            self.policy = self.get_random_action
        elif policy == 'greedy':
            self.policy = self.get_greedy_action
        else:
            if len(signature(policy).parameters) != 1:
                raise TypeError('Action selection function must have 1 parameter: state.')
            self.policy = policy

        # Set alpha (step size)
        if alpha_scheduler is None:
            pass
        elif isinstance(alpha_scheduler, (int, float)):
            self.fixed_alpha = True
            self.alpha_scheduler = lambda episodes: alpha_scheduler
            self.alpha = alpha_scheduler
        else:
            self.fixed_alpha = False
            if len(signature(alpha_scheduler).parameters) != 1:
                raise TypeError('alpha_scheduler function must have 1 parameter: episode.')
            self.alpha_scheduler = alpha_scheduler


        self.reset()  # resets the agent, not the environment

    def reset(self):

        obs_action_space = itertools.product(self.S, self.A)

        #self.Q = dict.fromkeys(obs_action_space, 0)
        #self.N = dict.fromkeys(obs_action_space, 0)

        for s, a in obs_action_space:
            self.Q[s, a] = self.q_value_init(s, a)
            self.N[s, a] = 0

        self.episodes = 0
        self.eps = 1

        self.episode_lengths = deque(maxlen=100000)
        self.rewards = deque(maxlen=100000)


    def discounted_reward(self, episode, t_start, simple=False):

        T = episode['T']

        if simple:
            return episode[T]['r']  # implemented for decrease in computation time
        else:

            disc_reward = 0

            for t in range(t_start, T + 1):
                disc_reward += self.gamma ** t * episode[t]['r']

            return disc_reward

    def run_episode(self):

        # print('starting new episode')
        self.episodes += 1

        maxT = 500

        while True:

            # print('trying episode ...')

            t = 0

            old_state = self.env.reset()
            old_action = self.policy(old_state)

            episode = {0: {'s': old_state, 'a': old_action}}
            total_reward = 0

            while True:

                # if t % 1000 == 0:
                #     print('IN EPSIODE', self.episodes, 'STEP', t)
                t += 1
                state, reward, done, info = self.env.step(old_action)

                total_reward += reward

                if done:

                    episode[t] = {'s': 'terminal', 'a': 'terminal', 'r': reward}
                    episode['T'] = t

                    # episode[t] = {'s': state, 'a': 'terminal', 'r': reward}
                    # episode[t+1] = {'s': 'terminal', 'a': 'terminal', 'r': 0}

                    break

                else:

                    action = self.policy(state)
                    episode[t] = {'s': state, 'a': action, 'r': reward}

                    old_action = action

            # statistic

            if episode['T'] < maxT:
                break

        self.episode_info['T'] = episode['T']
        self.episode_info['total reward'] = total_reward

        self.episode_lengths.append(episode['T'])
        self.rewards.append(total_reward)

        self.train_statistic['plays'] += 1
        if reward == 1:
            self.train_statistic['wins'] += 1  # only valid if environment terminates with reward of 1 if successfull.

        return episode

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

    def get_greedy_action(self, state):
        """
        Picks the greedy action, given a state.
        """

        Q = self.Q.predict([state])[0]

    def print_actionvalue_function(self):
        """
        Prints the action value function Q.
        """
        qs = []
        for s in self.S:
            qs.append([self.Q[s, a] for a in self.A])

        print(f'\nactionvalue function Q: (size statespace = {len(self.S)}, size actionspace = {len(self.A)})\n')
        print(np.array(qs))

    def train(self, n_episodes):
        """
        Trains the agent a number of episodes.
        """

        ema_T = 0  # exponentially moving average for episode duration
        ema_avg_reward = 0
        beta = 0.995

        with tqdm(total=n_episodes, postfix='T=') as t:

            for k in range(n_episodes):

                self.train_one_episode()

                postfix = ''

                if self.episode_info['T'] != None:
                    T = self.episode_info['T']
                    ema_T = beta * ema_T + (1 - beta) * T
                    postfix += f'T={ema_T:.2f}, '
                else:
                    postfix += 'T=n/a, '

                if self.episode_info['total reward'] != None:
                    reward = self.episode_info['total reward']
                    ema_avg_reward = beta * ema_avg_reward + (1 - beta)  * reward
                    postfix += f'avg_ret={ema_avg_reward:.2f}, '
                else:
                    postfix += f'avg_ret=n/a, '

                if self.eps != None:
                    postfix += f'eps={self.eps:.2e}, '

                if self.episode_info['wins'] != None:
                    wins = self.episode_info['wins']
                    postfix += f'total wins={wins}'
                else:
                    postfix += f'total wins n/a'

                t.postfix = postfix
                t.update()

                if self.print_every != 0 and (k + 1) % self.print_every == 0:

                    episode_lengths = np.array(self.episode_lengths)
                    rewards = np.array(self.rewards)

                    wins = int(np.sum(rewards[-1000:] == 1))
                    episode_lengths = np.mean(episode_lengths[-1000:])
                    print(f'episodes: {self.train_statistic["plays"]}, {100 * k / n_episodes:.2f}%, eps={self.eps:.2f}, wins={wins}, mean length episode: {episode_lengths}')

    def train_one_episode(self):
        raise NotImplementedError('Need to implement a training method')

    def play(self, episodes, print_statistic=True, random=False, max_episode_length=1000):
        """
        Tests the agent abilities in random or greedy play.
        """
        max_consecutive_wins = 0
        consecutive_wins = 0
        wins = 0
        episode_lengths = []

        for i in range(episodes):

            state = self.env.reset()

            for k in range(max_episode_length):

                if random:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_greedy_action(state)

                state, reward, done, info = self.env.step(action)


                if done:
                    break

            if reward == 1:

                wins += 1
                consecutive_wins += 1

                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)

            else:

                consecutive_wins = 0

            episode_lengths.append(k+1)

        win_percentage = 100 * wins / episodes

        print(f'played {episodes} games:\n\twins:{wins}\n\tlosses:{episodes-wins}\n\twin ratio:{win_percentage}%\n\tlargest winstreak: {max_consecutive_wins}\n\tavg episode lengths: {sum(episode_lengths)/len(episode_lengths)}')
        return wins

    def learn_and_test(self, n_train, n_test=2000, random=False, print_valuefunction=True):

        self.train(n_train)

        stat = self.train_statistic

        #print(stat)
        #print('win percentage:', 100 * stat['wins'] / stat['plays'])

        wins = self.play(n_test, random=random)

        if print_valuefunction:
            self.print_actionvalue_function()

        return wins

