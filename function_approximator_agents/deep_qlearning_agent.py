# external libraries
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm

# standard libraries
from collections import deque

# this package
from . import export
from .neural_network_agent import NeuralNetworkAgent
from utils import create_Sequential_Dense_net1, save_model_on_KeyboardInterrupt


@export
class DeepQLearningAgent(NeuralNetworkAgent):

    def __init__(self,
                 env,
                 save_model_path=None,
                 batch_size=16,
                 epsilon_scheduler=lambda episodes: 0.1,
                 policy='eps_greedy',
                 experience_replay=True,
                 size_Q_memory=1024,
                 fixed_target_weights=True,
                 update_period_fixed_target_weights=400,
                 gamma=1,
                 self_play=False):

        super().__init__(env=env,
                         policy=policy,
                         epsilon_scheduler=epsilon_scheduler,
                         gamma=gamma,
                         self_play=self_play,
                         save_model_path=save_model_path)

        if type(env.observation_space) is gym.spaces.discrete.Discrete:
            #self.Q_input_shape = (env.observation_space.n + env.action_space.n, )
            self.Q_input_shape = (env.observation_space.n, )
        elif type(env.observation_space) is gym.spaces.tuple.Tuple:
            self.Q_input_shape = (len(env.observation_space), )
            #self.Q_input_shape = (len(env.observation_space) + env.action_space.n, )

        self.Q = create_Sequential_Dense_net1(
            input_shape=self.Q_input_shape,
            n_outputs=env.action_space.n,
            layers=7,
            neurons=128,
            p_dropout=0.1,
            lambda_regularization=10**(-4),
        )

        self.starting_learning_rate = 5*10**(-5)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.starting_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )

        #self.loss = 'mean_absolute_error'
        #self.loss = 'mean_squared_error'
        self.loss = tf.keras.losses.Huber()

        self.Q.compile(
            optimizer=optimizer,
            loss=self.loss,
        )

        self.episodes = 0


        self.experience_replay = experience_replay
        self.fixed_target_weights = fixed_target_weights

        # reserve space for the experienced episodes, if desired
        if experience_replay:
            self.size_Q_memory = size_Q_memory
            #memorypoint = [state, action, target]
            self.Q_memory = np.zeros((size_Q_memory, self.Q_input_shape[0] + self.nb_actions))

        # here, we build a second network that will generate the predictions
        # to greate the Q learning target. This is done to create more stability.
        if fixed_target_weights:
            # the attribute 'Q_fixed_weights' shall be reserved for the second network
            self.Q_fixed_weights = tf.keras.models.clone_model(self.Q)
            self.update_period_fixed_target_weights = update_period_fixed_target_weights

    def get_batch(self):
        """
        deprecated
        """
        memory = self.Q_memory[:self.episodes]
        batch_size = max(1, min(self.batch_size, self.e))

        if self.episodes > self.size_Q_memory:  # memory is filled

            indices = np.random.choice(range(self.size_Q_memory), self.batch_size)
            x_train = self.Q_memory[:, :-1][indices]
            y_train = self.Q_memory[:, -1][indices]
            return x_train, y_train, True

        else:  # memory is too small

            return None, None, False

    def training_data(self):
        #state_action_inputs = self.Q_memory[:, :-1]
        #rewards = self.Q_memory[:, -1]
        #return state_action_inputs, rewards

        # one Q_memory row consists of [state, Qvalues(dim=nb_actions)]
        states = self.Q_memory[:, :-self.nb_actions]
        targets = self.Q_memory[:, -self.nb_actions:]

        return states, targets

    def update_Q(self, batch_size=128, episodes=2):

        if self.memory_ready():
        #if self.episodes > self.size_Q_memory:

            #state_action_inputs, rewards = self.training_data()
            states, targets = self.training_data()

            fit_result = self.Q.fit(
                #x=state_action_inputs,
                #y=rewards,
                x=states,
                y=targets,
                batch_size=batch_size,
                epochs=episodes,
                verbose=False)

            return fit_result
            # self.Q.train_on_batch(states, rewards)
        else:
            return False  # dont train here

    def update_fixed_target_weights(self):
        """
        Updates the fixed weights of the prediction network Q_fixed_targets.
        """
        current_weights = self.Q.get_weights()

        # fresh optimizer, since we will be updating towards new goals. (Old opt. weights are irrelevant.)
        opt = tf.keras.optimizers.Adam(
            learning_rate=self.starting_learning_rate
        )

        self.Q_fixed_weights = tf.keras.models.clone_model(self.Q)
        self.Q_fixed_weights.set_weights(current_weights)

        self.Q_fixed_weights.compile(
            optimizer=opt,
            loss=self.loss
        )

    def train_one_episode(self):
        # initialize
        self.episodes += 1

        state = self.env.reset()
        losses = []

        fit_info = None
        terminal = False

        if self.fixed_target_weights:
            if self.episodes % self.update_period_fixed_target_weights == 0:
                self.update_fixed_target_weights()

        # train the neural network
        if self.episodes % 20 == 0:
            fit_info = self.update_Q(episodes=10)
            if fit_info:
                history = fit_info.history
                losses.extend(history['loss'])
                if losses[-1] == 0:
                    print('loss was zero. ', fit_info)

        while not terminal:

            action = self.policy(state)

            if self.self_play and self.episodes > self.size_Q_memory:
                opponent_action = self.get_ideal_opponent_action
            else:
                opponent_action = self.get_random_action

            next_state, reward, terminal, info = self.env.step(action, opponent_action)

            if not terminal:
                maxQ = self.get_maxQ(next_state, network='Q_fixed_weights')
                target = reward + self.gamma * maxQ
            else:
                target = reward

            if self.experience_replay:

                qvalues = self.predict(state, network='Q')
                qvalues[action] = target

                self.Q_memory[self.episodes % self.size_Q_memory] = list(state) + list(qvalues)

            state = next_state

        mean_loss = np.mean(losses) if len(losses) > 0 else None
        train_info = {'mean_loss': mean_loss, 'last_reward': reward}

        return train_info

    def play_one_episode(self):
        """
        The agent plays one episode.
        """

        # setup
        terminal = False
        reward = None
        state = self.env.reset()
        episode_memory = []

        while not terminal:
            action = self.policy(state)

            # episode_memory.append((state, action, reward, terminal))

            next_state, reward, terminal, info = self.env.step(action, self.opponent_policy)

            state = next_state

        info['last_reward'] = reward
        info['episode_memory'] = episode_memory
        return info

    def _loop(self, n_episodes, train=True):
        """
        Main training or evaluation loop.
        """
        rewards_ = deque(maxlen=1000)
        wins = 0

        ema_reward = 0

        mean_loss = 0
        ema_mean_loss = 0
        beta_mean_loss = 0.99

        beta = 0.994
        total_reward = 0

        with tqdm(total=n_episodes, postfix='T=') as t:

            for k in range(n_episodes):

                if train:
                    info = self.train_one_episode()
                else:
                    info = self.play_one_episode()

                postfix = ''

                reward = info['last_reward']
                rewards_.append(reward)
                total_reward += reward

                try:
                    if reward == self.env.reward_range[-1]:  # e.g reward_range = (-1, 1), where 1 is the win
                        wins += 1
                    postfix += f'wins={100*wins/(k+1):.2f}%, '
                except AttributeError:
                    wins = 'n/a'

                ema_reward = beta * ema_reward + (1 - beta) * reward
                postfix += f'reward:{ema_reward/(1-beta**(k+1)):.2f}, '

                if 'mean_loss' in info and info['mean_loss']:
                    mean_loss = info['mean_loss']
                    ema_mean_loss = beta_mean_loss * ema_mean_loss + (1 - beta_mean_loss) * mean_loss
                    #postfix += f'mean loss={ema_mean_loss/(1-beta)**(k+1):.2f}, '
                postfix += f'mean loss={mean_loss:.3f} '

                if self.eps != None:
                    postfix += f'eps={self.eps:.2e} '

                #postfix += f'meanR={np.mean(rewards_):.3f}'

                t.postfix = postfix
                t.update()

                #if k % 500 == 0:
                #    print(self.predict(9*[0]).reshape((3,3)), f'{k=}')

        try:
            win_percentage = wins / n_episodes * 100
        except TypeError:
            win_percentage = 'n/a'

        print(f'did {n_episodes=}, {train=}, {total_reward=} (including negative rewards), win %: {win_percentage:.2f}')
        return total_reward

    def train(self, n_episodes):
        """
        Train the agent n_episodes
        """
        self.policy = 'eps_greedy'

        if self.self_play:
            self.opponent_policy = 'greedy'
        else:
            self.opponent_policy = 'random'
            
        return self._loop(n_episodes, train=True)

    def play(self, n_episodes, opponent_policy='random'):

        self.policy = 'greedy'
        self.opponent_policy = opponent_policy
        
        return self._loop(n_episodes, train=False)


if __name__ == '__main__':
    print('\n\n')
    agent = DeepQLearningAgent(env=None)
