
from tabular_agents import TabularAgent


class MonteCarloEveryVisitAgent(TabularAgent):

    def __init__(self, env, q_value_initialization, epsilon_scheduler, gamma=1):

        super().__init__(
            env,
            gamma,
            q_value_initialization,
            epsilon_scheduler,
            alpha_scheduler=None,
            policy='eps_greedy'
        )

    def train_one_episode(self):
        """
        Updates Action Value function for one episode
        """

        episode = self.run_episode()
        T = episode['T']

        for t in range(T):

            step = episode[t]

            state = step['s']
            action = step['a']

            Gt = self.discounted_reward(episode, t + 1, simple=False)

            self.N[state, action] += 1
            self.Q[state, action] += 1/self.N[state, action]*(Gt - self.Q[state, action])

        #print(self.episodes, T, self.eps)