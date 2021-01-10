from environments import KaggleTicTacToe, KaggleConnectX
import numpy as np
import random
from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent
from tests import connectx_3rows_3cols_3inarow_testcases

import sys
import tensorflow as tf

np.set_printoptions(precision=3)

print('\n------------\n')
const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes
bound_below_scheduler = lambda eps: lambda episodes: max(eps, 7000/(7000+episodes))


def predict_starting_position(agent, mode='connectx'):
    if mode == 'tictactoe':
        print(agent.predict(9*[0]).reshape((3, 3)))
    elif mode == 'connectx':
        print(agent.predict(9*[0]).reshape(agent.env.action_space.n))


def pred(model, mode='connectx'):
    if mode == 'tictactoe':
        vals = np.zeros((9, 18))
        vals[:, -1] = np.eye(9)
        print('using', vals)
        print(model(vals).numpy().reshape((3, 3)))
    elif mode == 'connectx':
        vals = np.zeros((3, 12))
        vals[:, -3:] = np.eye(3)
        print('using', vals)
        print(model(vals).numpy())

def time_(env, N):
    import time

    t_start=time.time()
    for i in range(N):
        e1 = env.env.clone()

    duration = time.time() - t_start
    print(f'cloning env via clone function took {duration} s. {duration/N}')

    from copy import copy, deepcopy
    t_start = time.time()
    for i in range(N):
        e2 = deepcopy(env)

    duration = time.time()- t_start
    print(f'cloning env via clone function took {duration} s. {duration/N}')
    sys.exit()

def time2_(agent):
    state = np.zeros(25)
    agent.predict(state)

def make_pr(environment):
    pr=lambda environment: print(environment.render(mode='ansi'))
    return pr

def set_random_seeds(n):
    random.seed(n)
    np.random.seed(n)
    tf.random.set_seed(n)

if __name__ == '__main__':

    #set_random_seeds(123)

    env = KaggleTicTacToe()
    #env = KaggleConnectX(rows=6, columns=7, inarow=4)

    #time_(env, 10000)

    starting_position = np.zeros((1, 9))

    # one memory point would be [state, action(one_hot), target]
    # the last one is the target r + max_a' Q(s',a', fixed_weights)

    #size_memory = 2*1024, len(env.observation_space) + env.action_space.n + 1
    size_memory = 4*512
    update_period = 2*512

    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        epsilon_scheduler=bound_below_scheduler(0.15),
        #epsilon_scheduler=const_scheduler(0.15),
        experience_replay=True,
        size_Q_memory=size_memory,
        fixed_target_weights=True,
        update_period_fixed_target_weights=update_period,
        self_play=True,
    )

    #for i in range(10): agent.play(5000, opponent_policy='random')
    #sys.exit()

    #predict_starting_position(agent)

    #evaluate_testcases = connectx_3rows_3cols_3inarow_testcases.check

    agent.train_and_play(train=update_period,
                         play=600,
                         repeat=160,
                         funcs=[
                            #predict_starting_position,
                            #evaluate_testcases
                            ]
                         )


    print(agent.Q.weights[0][0][:35])

