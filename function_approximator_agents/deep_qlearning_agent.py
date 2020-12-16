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
            self.Q = function_approximator_agents.utils.create_Dense_net1()

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
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



    def update_Q_yes_replay_no_fixed(self):

        if self.episodes > self.size_Q_memory:
            states = self.Q_memory[:, :-1]
            rewards = self.Q_memory[:, -1]

            fit_result =self.Q.fit(x=states, y=rewards, batch_size=32, epochs=32)
            print('FIT RESULT', fit_result)
            sys.exit()
            return fit_result
            # self.Q.train_on_batch(states, rewards)
        else:
            return None # dont train here

    def train_one_episode(self):

        # initialize
        self.episodes += 1
        terminal = False

        state = self.env.reset()

        while not terminal:

            action = self.policy(state)
            next_state, reward, terminal, info = self.env.step(action)

            if terminal:
                target = reward
            else:
                target = reward + self.gamma * self.get_maxQ(next_state)

            if self.experience_replay:
                self.Q_memory[self.episodes % self.size_Q_memory] = list(state) + [target]  # add to memory

            train_info = self.update_Q()
            state = next_state

        return reward


    def get_random_action(self):
        valid_actions = np.where(self.env.board == 0)[0]
        #print(valid_actions)#, np.random.choice(valid_actions))
        return np.random.choice(valid_actions)

    def train(self, n_episodes):

        ema_reward = 0
        beta = 0.995

        with tqdm(total=n_episodes, postfix='T=') as t:

            for k in range(n_episodes):
                print(k)
                self.train_one_episode()

                postfix = ''
                """
                T = self.episode_info['T']
                    ema_T = beta * ema_T + (1 - beta) * T
                    postfix += f'T={ema_T:.2f}, '
                else:
                    postfix += 'T=n/a, '

                if self.episode_info['total reward'] != None:
                    reward = self.episode_info['total reward']
                    ema_avg_reward = beta * ema_avg_reward + (1 - beta) * reward
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
                """



if __name__ == '__main__':
    print('MAIN')
    agent = DeepQLearningAgent(env=None)
