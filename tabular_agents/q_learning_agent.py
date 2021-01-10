import numpy as np

import sys
from inspect import signature

from tabular_agents import TabularAgent

__all__ = ['QLearningAgent']

class QLearningAgent(TabularAgent):
    """
    Implements a Q Learning Agent, thereby solving the Bellman optimality equation iteratively.
    """
    def __init__(self, env, q_value_initialization, epsilon_scheduler, alpha_scheduler=lambda episodes: 0.01, gamma=1, behaviour_policy='random'):
        super().__init__(env, gamma, q_value_initialization, epsilon_scheduler, alpha_scheduler, behaviour_policy)
        
        self.q_memory = []
        self.last_memorized = 0

    def train_one_episode(self):
        """
        Updates Action Value function for one episode
        """

        # setup
        terminal = False
        self.episodes += 1
        t = 0
        total_reward = 0
        alpha = self.alpha_scheduler(self.episodes) #  learning rate

        # initialize environment
        state = self.env.reset()

        while not terminal:

            action = self.policy(state)
            next_state, reward, terminal, info = self.env.step(action)

            if terminal:
                target = reward
            else:
                max_q = max(self.Q[next_state, a] for a in self.A)
                target = reward + self.gamma * max_q

            self.Q[state, action] = (1 - alpha) * self.Q[state, action] + alpha * target

            state = next_state

            total_reward += reward  # just for statistics
            t += 1  # just for statistics

        # statistic

        self.episode_info['T'] = t
        self.episode_info['total reward'] = total_reward

        if self.env.reward_range[-1] == reward:
            self.episode_info['wins'] += 1



        self.T = t
        self.episode_lengths.append(t)
        self.rewards.append(reward)

        self.train_statistic['plays'] += 1
        if reward == 1:
            self.train_statistic['wins'] += 1

