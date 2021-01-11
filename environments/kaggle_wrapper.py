# external libraries
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple

# standard libraries
import contextlib
with contextlib.redirect_stdout(None):
    # don't print annoying gfootball warning
    from kaggle_environments import make

from . import export

class Env:

    def __init__(self):
        pass

    def player_action(self, action):

        pass

    def opponent_action(self, action):
        pass

class KaggleEnvWrapper():
    """
    Wrapper around the environments 'connectx' and 'tictactoe'
    to make it open ai gym compliant.
    This means, reset(), step() return the appropriate values.
    In addition, action_space and observation_space are defined.
    """
    def __init__(self, dtype_state=np.ndarray, opponent_policy='random'):

        self.opponent_policy = opponent_policy

        supported_types = [np.ndarray, tuple]
        if dtype_state not in supported_types:
            raise NotImplementedError(f'datatype {dtype_state} not supported. Can only use {supported_types}.')
        else:
            self.dtype_state = dtype_state

    def __str__(self):
        return self.env.render(mode='ansi')

    def _transform(self, state):
        """
        Different agents require different datatypes to work with.
        Whenever we return the state to an agent, transform the internal numpy
        representation accordingly.
        """
        if self.dtype_state == np.ndarray:
            return np.array(state)
        elif self.dtype_state == tuple:
            return tuple(state)


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


    def player_action(self, action):

        action = int(action)
        # print('EXECUTING PLAYER STEP')

        observation = self.env.step([action, None])
        self.state, reward, terminal, info = self.parse_observation(observation)

        if not info['valid']:
            print(action, terminal, reward, self.state, info)
            raise RuntimeError('player chose non valid action')
        if info['player_active']:
            print(action, terminal, reward, self.state, info)
            raise RuntimeError('after player moved, he is not in state inactive')

        return self.state, reward, terminal, info

    def execute_opponent_action(self, action_opponent=None):
        # print('EXECUTING OPPONENT STEP')
        observation = self.env.step([None, int(action_opponent)])

        self.state, reward, terminal, info = self.parse_observation(observation)
        if not info['valid']:
            print(action_opponent, action_opponent, reward, self.state, info)
            raise RuntimeError('invalid opponent action')

        state = self._transform(self.state)
        return state, reward, terminal, info

    def step(self, action, action_opponent='random'):
        """
        Executes one complete turn. Note: action_opponent is a function receiving the intermediate
        state after the environment has executed the players step.
        """
        state, reward, terminal, info = self.player_action(action)

        if terminal:
            state = self._transform(self.state)
            return state, reward, terminal, info  # the player won

        if action_opponent == 'random':
            action_opp = self.get_random_action()
        else:
            action_opp = action_opponent(state)  # calculate the opponents action

        state, reward, terminal, info = self.execute_opponent_action(action_opp)
        state = self._transform(self.state)
        return state, reward, terminal, info

    def step_DEPRECATED(self, action, action_opponent=None):
        """
        Deprecated
        """
        action = int(action)
        #print('EXECUTING PLAYER STEP')
        self.state, reward, terminal, info = self.parse_observation(self.env.step([action, None]))

        if not info['valid']:
            print(action, terminal, reward, self.state, info)
            raise RuntimeError('player chose non valid action')
        if info['player_active']:
            print(action, terminal, reward, self.state, info)
            raise RuntimeError('after player moved, he is not in state inactive')

        if terminal:
            state = self._transform(self.state)
            return state, reward, terminal, info  # the player won

        if action_opponent is None:
            if self.opponent_policy == 'random':
                action_opponent = self.get_random_action()
            elif self.opponent_policy == 'first_allowed':
                action_opponent = int(self.get_allowed_actions()[0])
            else:
                raise ValueError('no opponent action or strategy provided')
        elif action_opponent:
            action_opp = action_opponent(self.state)

        #print('EXECUTING OPPONENT STEP')
        observation = self.env.step([None, action_opp])

        self.state, reward, terminal, info = self.parse_observation(observation)
        if not info['valid']:
            print(action, action_opponent, reward, self.state, info)
            raise RuntimeError('invalid opponent action')

        state = self._transform(self.state)
        return state, reward, terminal, info

    def reset(self):
        """
        Resets the environment and returns the state
        """
        observation = self.env.reset()
        self.state = np.array(self.get_state(observation))
        return self._transform(self.state)

    def get_state(self, observation):
        board = observation[0]['observation']['board']
        return board

    def get_allowed_actions(self, state=None):
        """
        Gets all allowed actions, given a state.
        If the allowed actions depend on the state, the environment must override this method
        """
        return range(self.env.action_space.n)

    def get_random_action(self):
        """
        Returns a randomly sampled action.
        """
        allowed = self.get_allowed_actions()
        return int(np.random.choice(allowed))

@export
class KaggleConnectX(KaggleEnvWrapper):
    """
    Info for rows=3, colums=3, inarow=3 random vs random policy leads to 54% winrate.
             4/4/3 -> random policy -> 62% winrate
    """
    def __init__(self, rows=6, columns=7, inarow=4, opponent_policy='random', dtype_state=np.ndarray):

        super().__init__(dtype_state=dtype_state, opponent_policy=opponent_policy)

        config = {'rows': rows, 'columns': columns, 'inarow': inarow}

        self.env = make("connectx",
                        configuration=config,
                        debug=True)

        self.state = np.zeros(rows * columns)

        # make environment more gym compliant
        self.action_space = Discrete(columns)
        self.action_space.sample = self.get_random_action

        #self.observation_space = Discrete(rows * columns)
        self.observation_space = Tuple(rows * columns * [Discrete(3)])

        self.rows = rows
        self.columns = columns
        self.reward_range = (-1, 1)

    def __str__(self):
        baserepr = super().__str__()

        replace = {
            '|': ' ',
            '+': ' ',
            '-': ' ',
            '0': '-',
            '1': 'X',
            '2': 'o',
        }

        for search, replacement in replace.items():
            baserepr = baserepr.replace(search, replacement)

        baserepr += '  ' + '   '.join(map(str, range(self.columns)))
        return baserepr

    def get_allowed_actions(self, state=None):

        if state is None:
            state = self.state

        first_row = state[:self.columns]
        allowed = np.where(first_row == 0)[0]
        return allowed

@export
class KaggleTicTacToe(KaggleEnvWrapper):
    """
    This class is a wrapper class around kaggles TicTacToe environment.

    The average winrate for a random player vs a random opponent is about 58%.
    """
    def __init__(self, dtype_state=np.ndarray):

        super().__init__(dtype_state=dtype_state)

        self.env = make("tictactoe", debug=True)

        self.state = np.zeros(9)

        # make environment more gym compliant
        self.action_space = Discrete(9)
        self.action_space.sample = self.get_random_action

        #self.observation_space = Discrete(9)
        #self.observation_space = Tuple([Discrete(3), Discrete(3)])
        self.observation_space = Tuple(9*[Discrete(3)])

        self.reward_range = (-1, 1)

    def get_allowed_actions(self, state=None):
        """
        Returns the allowed actions.
        """
        if state is None:
            state = self.state
        return np.where(state == 0)[0]


if __name__ == '__main__':

    def step(action):
        state, reward, terminal, info = env.step(action)
        print(env)
        print(f'{reward=}, {terminal=}')


    env = KaggleConnectX(rows=3, columns=3, inarow=3)
    #state = env.reset()
