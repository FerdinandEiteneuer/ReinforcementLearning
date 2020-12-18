import numpy as np
import contextlib
import gym
from gym.spaces.discrete import Discrete


with contextlib.redirect_stdout(None):
    # don't print annoying gfootball warning
    from kaggle_environments import make


class KaggleEnvWrapper():
    """
    Wrapper around the environments 'connectx' and 'tictactoe'
    to make it open ai gym compliant.
    This means, reset(), step() return the appropriate values.
    In addition, action_space and observation_space are defined.
    """
    def __init__(self):
        pass

    def __str__(self):
        return env.render(mode='ansi')

    def get_random_action(self):
        raise NotImplementedError('The derived class must implement this.a')

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

    def step(self, action, action_opponent='random'):

        #print('EXECUTING PLAYER STEP')
        self.state, reward, terminal, info = self.parse_observation(self.env.step([action, None]))

        if not info['valid']:
            print(action, terminal, reward, self.state, info)
            raise RuntimeError('player chose non valid action')
        if info['player_active']:
            print(action, terminal, reward, self.state, info)
            raise RuntimeError('after player moved, he is not in state inactive')

        if terminal:
            return self.state, reward, terminal, info  # the plyaer won



        if isinstance(action_opponent, str) and action_opponent == 'random':
            action_opponent = self.get_random_action()

        #print('EXECUTING OPPONENT STEP')
        observation = self.env.step([None, action_opponent])

        self.state, reward, terminal, info = self.parse_observation(observation)
        if not info['valid']:
            print(action, action_opponent, reward, self.state, info)
            raise RuntimeError('invalid opponent action')

        return self.state, reward, terminal, info

    def reset(self):
        """
        Resets the environment and returns the state
        """
        observation = self.env.reset()
        self.state = np.array(self.get_state(observation))
        return self.state

    def get_state(self, observation):
        board = observation[0]['observation']['board']
        return board


class KaggleConnectX(KaggleEnvWrapper):

    def __init__(self, rows=6, columns=7):

        super().__init__()

        config = {'rows': rows, 'columns': columns}

        self.env = make("connectx",
                        configuration=config,
                        debug=True)

        self.state = np.zeros((rows, columns))

        # make environment more gym compliant
        self.action_space = Discrete(columns)
        self.observation_space = Discrete(rows * columns)
        self.action_space.sample = self.get_random_action

        self.rows = rows
        self.columns = columns

    def get_random_action(self):
        """
        Returns a randomly sampled action.
        """
        indices_first_row = [self.rows * i  for i in range(self.columns)]
        first_row = state[indices_first_row]
        return np.random.choice(np.where(first_row == 0)[0])


class KaggleTicTacToe(KaggleEnvWrapper):

    def __init__(self):

        self.env = make("tictactoe", debug=True)

        self.state = np.zeros(9)

        # make environment more gym compliant
        self.action_space = Discrete(9)
        self.observation_space = Discrete(9)

        self.action_space.sample = self.get_random_action


    def get_random_action(self):
        """
        Returns a randomly sampled action.
        """
        #print('tactactoe env sampling random action', self.state, np.random.choice(np.where(self.state == 0)[0]))
        return int(np.random.choice(np.where(self.state == 0)[0]))


if __name__ == '__main__':

    def step(action):
        state, reward, terminal, info = env.step(action)
        print(env)
        print(f'{reward=}, {terminal=}')

    env = KaggleConnectX()
    state = env.reset()
