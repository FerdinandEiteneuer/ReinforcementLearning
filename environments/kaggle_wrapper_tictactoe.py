import numpy as np
import contextlib
import gym


with contextlib.redirect_stdout(None):
    # don't print annoying gfootball warning
    from kaggle_environments import make


class KaggleTicTacToe():
    """
    Wrapper around the environment 'tictactoe' to make it open ai gym compliant.
    This means, reset(), step() return the appropriate values.
    """
    def __init__(self):

        self.env = make("tictactoe", debug=True)
        self.state = np.zeros(9)

        self.player1_icon = 'X'
        self.player2_icon = 'o'
        self.empty_state_human_readable = '-'

        # make environment more gym compliant
        self.action_space = gym.spaces.discrete.Discrete(9)
        self.action_space.sample = self.get_random_action

        self.observation_size = gym.spaces.discrete.Discrete(9)

    def __str__(self):
        """
        Pretty print the tic tac toe board.
        """
        board_human_readable = []
        for s in self.state:
            if s == 1:
                board_human_readable.append(self.player1_icon)
            elif s == 2:
                board_human_readable.append(self.player2_icon)
            else:
                board_human_readable.append(self.empty_state_human_readable)

        representation = '{0} {1} {2}\n{3} {4} {5}\n{6} {7} {8}'.format(*board_human_readable)
        return representation

    def parse_observation(self, observation):
        terminal = observation[0]['status'] == 'DONE'
        reward = observation[0]['reward']
        state = np.array(observation[0]['observation']['board'])

        status1 = observation[0]['status']
        status2 = observation[1]['status']
        valid = status1 != 'INVALID' and status2 != 'INVALID'

        player_active = observation[0]['status'] == 'ACTIVE'
        info = {'valid': valid, 'player_active': player_active}

        return state, reward, terminal, info

    def step(self, action, mode='random_opponent'):
        if mode == 'random_opponent':

            self.state, reward, terminal, info = self.parse_observation(self.env.step([action, None]))

            if not info['valid']:
                print(action, terminal, reward, self.state, info)
                raise RuntimeError('player chose non valid action')
            if info['player_active']:
                print(action, terminal, reward, self.state, info)
                raise RuntimeError('after player moved, he is not in state inactive')

            if terminal:
                return self.state, reward, terminal, info  # the plyaer won

            action_opponent = self.get_random_action()
            observation = self.env.step([None, action_opponent])  # put -1 just to make sure this is the opponent.

            self.state, reward, terminal, info = self.parse_observation(observation)
            if not info['valid']:
                print(action, action_opponent, reward, self.state, info)
                raise RuntimeError('invalid opponent action')

            return self.state, reward, terminal, info

        else:
            raise NotImplementedError('wrong mode')

    def reset(self):
        self.state = np.array(self.env.reset()[0]['observation']['board'])
        #self.state = self.env.reset()[0]['observation']['board']
        return self.state

    def get_random_action(self):
        """
        Returns a randomly sample action.
        """

        possibilities = []
        for i, s in enumerate(self.state):
            if s == 0:
                possibilities.append(i)

        return int(np.random.choice(possibilities))

        choices = np.where(self.state == 0)[0]
        return int(np.random.choice(choices))

    def get_state(self, observation):
        board = observation[0]['observation']['board']
        return board


if __name__ == '__main__':

    def step(action):
        state, reward, terminal, info = env.step(action)
        print(env)
        print(f'{reward=}, {terminal=}')

    env = KaggleTicTacToe()
    state = env.reset()
