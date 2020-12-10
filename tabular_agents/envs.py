import numpy as np
import gym

grid1 = {

    'world': np.array([
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    ]),
    'start': (0, 3),  # coordinates for a system with y pointing north and x pointing east
    'goal': (7, 3)

}

grid2 = {

    'world': np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    'start': (0, 3),  # coordinates for a system with y pointing north and x pointing east
    'goal': (7, 3)

}

WORLDS = {
    'windy_gridworld': grid1,
    'no_wind': grid2,
}


class WindyGridWorld:

    def __init__(self, world, moves):


        self.reward_range = (-1, 0)

        if world in WORLDS:

            self.world = WORLDS[world]['world']
            self.start = WORLDS[world]['start']
            self.goal = WORLDS[world]['goal']
            self.height, self.width = self.world.shape

            self.max_x = self.width - 1
            self.max_y = self.height - 1

        else:
            raise ValueError(f'Select valid grid from {list(WORLDS.keys())}')

        self.action_to_move = {
            0: (0, 1),  # north
            1: (1, 0),  # east
            2: (0, -1),  # south
            3: (-1, 0),  # west
            4: (1, 1),  # north-east
            5: (1, -1),  # south-east
            6: (-1, -1),  # south-west
            7: (-1, 1),  # north-west
            8: (0, 0),  # no movement
        }

        self.moves_humanreadable = {
           0: '↑',
           1: '→',
           2: '↓',
           3: '←',
           4: '↗',
           5: '↘',
           6: '↙',
           7: '↖',
           8: '-'
        }

        self.moves = moves

        if self.moves == 'standard':
            self.action_space = gym.spaces.discrete.Discrete(4)
        elif self.moves == 'king':
            self.action_space = gym.spaces.discrete.Discrete(8)
        elif self.moves == 'king+':
            self.action_space = gym.spaces.discrete.Discrete(9)
        else:
            raise ValueError('bad moves')

        self.observation_space = gym.spaces.discrete.Discrete(self.width * self.height)



    def pos_to_state(self, pos):

        state = pos[0] + pos[1] * self.width
        return state

    def state_to_pos(self, state):

        pos = (state % self.width, state // self.width)
        return pos

    def reset(self):

        self.pos = self.start
        return self.pos_to_state(self.pos)

    def step(self, action, verbose=False):
       
        pos_x, pos_y = self.pos

        # windy
        wind_y = self.world[self.max_y - pos_y, pos_x]

        move_x, move_y = self.action_to_move[action]

        # move
        pos_x_new = pos_x + move_x
        pos_y_new = pos_y + move_y + wind_y

        # border constraints
        pos_x_new = max(0, min(pos_x_new, self.max_x))  # constrain 0 <= pos_x_new <= width
        pos_y_new = max(0, min(pos_y_new, self.max_y))  # constrain 0 <= pos_y_new <= width

        # pos_y_new = max(0, min(pos_y_new + wind_y, self.max_y))

        self.pos = pos_x_new, pos_y_new
        new_state = self.pos_to_state(self.pos)

        # print(self.goal, new_state, type(new_state), type(self.goal))
        terminal = (self.pos == self.goal)
        reward = -1 if not terminal else 0

        # print(terminal, reward, self.pos, new_state)

        if verbose:
            pass

        return new_state, reward, terminal, {'pos': self.pos, 'wind': wind_y}

    def print_action_valuefunction(self, Q):

        grid = ''

        for y in range(self.height)[::-1]:
            for x in range(self.width):
                state = self.pos_to_state((x, y))

                q_values = [Q[state, a] for a in range(self.action_space.n)]
                action = q_values.index(max(q_values))
                printout = self.moves_humanreadable[action]

                padded = f'{printout:3s}'

                grid += padded
            grid += '\n'

        grid += self.width*'---' + '\n'
        colums = '  '.join(map(str, list(range(self.width))))
        grid += colums

        print(grid)

    def print_episode(self, episode):

        path = ''
        
        pos = self.state_to_pos(episode[0]['s'])

        path += 'starting: ' + str(pos) + '\n'

        T = episode['T']
        for t in range(1, T):

            step = episode[t]
            pos = self.state_to_pos(step['s'])
            action = self.moves_humanreadable[step['a']]

            path += f'--> {pos}, going {action}\n'

        print(path)


class TicTacToe:

    def __init__(self):

        self.action_space = gym.spaces.discrete.Discrete(9)

        
    def calc_size_observation_space(self):

        pass


class Easy21:
    
    def __init__(self):
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.color = [1, 1, -1]  # for modifying the sign

        self.action_space = gym.spaces.Discrete(2)  # hit or stick

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(10),
            gym.spaces.Discrete(21)
        ))
        self.state_offset = 1  # This means we have states 1..21 instead of 0..20

    def draw_card(self, only_positive=False):

        card = np.random.choice(self.cards)

        if only_positive:
            return card
        else:
            return card * np.random.choice(self.color)

    def reset(self):

        dealer = self.draw_card(only_positive=True)
        player = self.draw_card(only_positive=True)

        self.state = (dealer, player)

        return dealer, player

    def step(self, action):

        reward = None
        terminal = False

        dealer, player = self.state

        if action == 0:  # stick
            self.state, reward, terminal = self.dealer(self.state)
            return self.state, reward, terminal, {}

        elif action == 1:  # hit
            new_player = player + self.draw_card()

            next_state = dealer, new_player

            if self.busted(new_player):
                reward = -1
                terminal = True
            else:
                reward = 0
                terminal = False

            self.state = next_state
            return next_state, reward, terminal, {}

    def busted(self, value):
        return value < 1 or value > 21

    def dealer(self, state):

        dealer, player = state

        while not self.busted(dealer):

            if dealer >= 17:
                break

            dealer += self.draw_card()

        terminal = True
        reward = None

        if self.busted(dealer):
            reward = 1
        else:
            if dealer == player:
                reward = 0
            elif dealer > player:
                reward = -1
            elif dealer < player:
                reward = +1

        return state, reward, terminal

    def print_action_valuefunction(self, Q):
        print(Q)

    @classmethod
    def get_valuefunction_numpy(self, Q):

        xs, ys = [], []

        for key, value in Q.items():
            state, action = key

            if state == 'terminal':
                continue
            xs.append(state[0])
            ys.append(state[1])

        minx = min(xs)
        miny = min(ys)
        maxx = max(xs)
        maxy = max(ys)

        arr = np.zeros((maxx, maxy))
        #print(Q)
        for x in range(maxx):
            for y in range(maxy):
                try:
                    s = (x + 1, y + 1)
                    qmax = max(Q[s, 0], Q[s, 1])
                    arr[x, y] = qmax
                    #arr[x, y] = V[(x + 1, y + 1)]
                except KeyError:
                    arr[x, y] = 0  # state not seen

        return arr



if __name__ == '__main__':

    # env = WindyGridWorld('grid1', moves='standard')

    env = Easy21()
    
    s = env.reset()

    s = env.step