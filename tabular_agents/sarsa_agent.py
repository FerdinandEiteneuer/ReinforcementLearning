from tabular_agents import TabularAgent


class SarsaAgent(TabularAgent):

    def __init__(self, env, q_value_initialization, epsilon_scheduler, alpha_scheduler, gamma=1):

        super().__init__(env, gamma, q_value_initialization, epsilon_scheduler, alpha_scheduler, 'eps_greedy')

    def train_one_episode(self):

        # setup
        terminal = False
        self.episodes += 1

        # initialize environment and first action
        state = self.env.reset()
        action = self.policy(state)  # eps greedy by default. This is set in "__init__"

        episode_length = 0
        total_reward = 0

        while not terminal:

            next_state, reward, terminal, info = self.env.step(action)

            if not terminal:
                next_action = self.policy(next_state)
                TD_Target = reward + self.gamma * self.Q[next_state, next_action]
            else:
                next_action = None
                TD_Target = reward

            alpha = self.alpha_scheduler(self.episodes)

            self.Q[state, action] += alpha * (TD_Target - self.Q[state, action])

            # prepare for next step
            state = next_state
            action = next_action

            # bookkeeping
            episode_length += 1
            total_reward += reward


        self.episode_info['T'] = episode_length
        self.episode_info['total reward'] = total_reward
        self.episode_lengths.append(episode_length)
        self.rewards.append(total_reward)
        self.train_statistic['plays'] += 1
        if reward == 1:
            self.train_statistic['wins'] += 1
        return info

