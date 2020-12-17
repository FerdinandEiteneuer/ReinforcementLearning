# external libraries
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys


from function_approximator_agents.neural_network_agent import NeuralNetworkAgent
import function_approximator_agents.utils

class DeepQLearningAgent(NeuralNetworkAgent):

    def __init__(self,
                 env,
                 size_Q_memory=(1024, 10),
                 Q=None,
                 batch_size=16,
                 epsilon_scheduler=lambda episodes: 0.1,
                 policy='eps_greedy',
                 experience_replay=True,
                 fixed_target_weights=False,
                 gamma=1):

        super().__init__(env=env, policy=policy, epsilon_scheduler=epsilon_scheduler, gamma=gamma)

        if Q is not None:
            self.Q = Q
        else:
            self.Q = function_approximator_agents.utils.create_Dense_net1(
                layers=1,
                neurons=256,
                p_dropout=0.3,
            )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False
        )

        self.Q.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
        )


        self.episodes = 0

        # reserve space for the experienced episodes, if desired
        if experience_replay:
            self.Q_memory = np.zeros(size_Q_memory)
            self.size_Q_memory = self.Q_memory.shape[0]

        if not fixed_target_weights and not experience_replay:
            raise NotImplementedError

        elif not fixed_target_weights and experience_replay:
            self.update_Q = self.update_Q_yes_replay_no_fixed

        elif fixed_target_weights and not experience_replay:
            raise NotImplementedError

        elif fixed_target_weights and experience_replay:
            raise NotImplementedError

        self.experience_replay = experience_replay
        self.fixed_target_weights = fixed_target_weights

    def get_batch(self):

        memory = self.Q_memory[:self.episodes]
        batch_size = max(1, min(self.batch_size, self.e))

        if self.episodes > self.size_Q_memory:  # memory is filled

            indices = np.random.choice(range(self.size_Q_memory), self.batch_size)
            x_train = self.Q_memory[:, :-1][indices]
            y_train = self.Q_memory[:, -1][indices]
            return x_train, y_train, True

        else:  # memory is too small

            return None, None, False


    def get_training_data(self):
        states = self.Q_memory[:, :-1]
        rewards = self.Q_memory[:, -1]
        return states, rewards

    def update_Q_yes_replay_no_fixed(self, episodes=4):

        if self.episodes > self.size_Q_memory:

            states, rewards = self.training_data()

            fit_result =self.Q.fit(x=states, y=rewards, batch_size=128, epochs=episodes, verbose=False)
            return fit_result
            # self.Q.train_on_batch(states, rewards)
        else:
            return False # dont train here

    def train_one_episode(self):

        # initialize
        self.episodes += 1
        terminal = False

        state = self.env.reset()

        losses = []
        fit_info = None

        while not terminal:

            action = self.policy(state)
            #action = self.get_random_action()
            #print(f'{self.episodes=}, {action=}, {self.env.board}')

            next_state, reward, terminal, info = self.env.step(action)

            if terminal:
                target = reward
            else:
                target = reward + self.gamma * self.get_maxQ(next_state)

            if self.experience_replay:
                self.Q_memory[self.episodes % self.size_Q_memory] = list(state) + [target]  # add to memory

            if self.episodes % 50 == 0:
                fit_info = self.update_Q(episodes=20)

            if fit_info:
                history = fit_info.history
                losses.extend(history['loss'])
                fit_info = None

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

            #episode_memory.append((state, action, reward, terminal))

            next_state, reward, terminal, info = self.env.step(action)

            state = next_state

        info['last_reward'] = reward
        info['episode_memory'] = episode_memory
        return info

    def _loop(self, n_episodes, train=True):
        """
        Main training or evaluation loop.
        """
        ema_reward = 0

        ema_mean_loss = 0
        beta_mean_loss = 0.9

        beta = 0.99
        total_reward = 0

        with tqdm(total=n_episodes, postfix='T=') as t:

            for k in range(n_episodes):

                if train:
                    info = self.train_one_episode()
                else:
                    info = self.play_one_episode()


                postfix = ''

                reward = info['last_reward']
                total_reward += reward

                ema_reward = beta * ema_reward + (1 - beta) * reward
                postfix += f'avg reward:{ema_reward:.2f}, '


                if 'mean_loss' in info and info['mean_loss']:
                    mean_loss = info['mean_loss']
                    ema_mean_loss = beta_mean_loss * ema_mean_loss + (1 - beta_mean_loss) * mean_loss
                postfix += f'mean loss={ema_mean_loss:.2f}, '

                if self.eps != None:
                    postfix += f'eps={self.eps:.2e}'

                t.postfix = postfix
                t.update()

        print(f'did {n_episodes=}, {total_reward=}, win %: {total_reward/n_episodes*100}')
        return total_reward

    def train(self, n_episodes, policy='eps_greedy'):
        """
        Train the agent n_episodes
        """
        self.set_policy(policy)
        return self._loop(n_episodes, train=True)

    def play(self, n_episodes, policy='greedy'):

        self.set_policy(policy)
        return self._loop(n_episodes, train=False)


if __name__ == '__main__':
    print('MAIN')
    agent = DeepQLearningAgent(env=None)
