#external libraries
import numpy as np

# standard libraries
import sys

# this package
from tabular_agents import TabularAgent


__all__ = ["SarsaLambdaAgent"]

class SarsaLambdaAgent(TabularAgent):

    def __init__(self, env, q_value_initialization, epsilon_scheduler, alpha_scheduler, policy, lambd=0.9, gamma=1, forward_view=False):

        super().__init__(env, gamma, q_value_initialization, epsilon_scheduler, alpha_scheduler, policy)

        self.lambd = lambd
        self.Q = np.zeros(self.dims_obs_action_space)

        if forward_view:
            self.train_one_episode = self.train_one_episode_forward_view
        else:
            #self.train_one_episode = self.train_one_episode_backward_view_slow_but_easy_implementation
            self.train_one_episode = self.train_one_episode_backward_view

    def train_one_episode_backward_view(self):
        """
        Computationally effecient version of the backward view
        """
        self.episodes += 1

        E = np.zeros(self.dims_obs_action_space)

        terminal = False

        # initialize environment and first action
        state = self.env.reset()
        action = self.policy(state)  # eps greedy by default. This is configured in "super().__init__"

        # for bookkeeping, no functionality
        episode_length = 0
        total_reward = 0

        while not terminal:

            alpha = self.alpha_scheduler(self.episodes)

            next_state, reward, terminal, info = self.env.step(action)

            if not terminal:
                next_action = self.policy(next_state)
                TD_Target = reward + self.gamma * self.Q[next_state][next_action]
            else:
                next_action = None
                TD_Target = reward + 0 # Q[next_state=terminal][next_action] = 0

            delta = TD_Target - self.Q[state][action]

            E[state][action] += 1

            #for s, a in self.observation_action_space:
            #    self.Q[s, a] += alpha * delta * E[s][a]
            #    E[s][a] *= self.gamma * self.lambd

            self.Q += alpha * delta * E
            E *= self.gamma*self.lambd

            # prepare for next step
            state = next_state
            action = next_action

            # bookkeeping
            episode_length += 1
            total_reward += reward

        self.episode_info['T'] = episode_length
        self.episode_info['total reward'] = total_reward

        if self.env.reward_range[-1] == reward:
            self.episode_info['wins'] += 1


    def train_one_episode_backward_view_slow_but_easy_implementation(self):
        """
        Computationally inefficient, but was easier to implement.
        """

        self.episodes += 1
        E = dict.fromkeys(self.Q, 0)
        terminal = False

        # initialize environment and first action
        state = self.env.reset()
        action = self.policy(state)  # eps greedy by default. This is configured in "super().__init__"

        episode_length = 0
        total_reward = 0

        while not terminal:

            alpha = self.alpha_scheduler(self.episodes)

            next_state, reward, terminal, info = self.env.step(action)

            if not terminal:
                next_action = self.policy(next_state)
                TD_Target = reward + self.gamma * self.Q[next_state, next_action]
            else:
                next_action = None
                TD_Target = reward + 0 # Q[next_state=terminal, next_action] = 0

            delta = TD_Target - self.Q[state, action]

            E[state, action] += 1

            for s, a in self.observation_action_space:
                self.Q[s, a] += alpha * delta * E[s, a]
                E[s, a] *= self.gamma * self.lambd

            # prepare for next step
            state = next_state
            action = next_action

            # bookkeeping
            episode_length += 1
            total_reward += reward

        self.episode_info['T'] = episode_length
        self.episode_info['total reward'] = total_reward

        if self.env.reward_range[-1] == reward:
            self.episode_info['wins'] += 1


    def train_one_episode_forward_view(self):

        pass

