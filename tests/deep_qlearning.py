# external libraries
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)

# standard library
import random
import os

# this package
from utils import ConstEpsilonScheduler, DecayingEpsilonScheduler
from environments import KaggleTicTacToe, KaggleConnectX
from function_approximator_agents import DeepQLearningAgent
from tests import connectx_3rows_3cols_3inarow_testcases

print('\n------------\n')

def predict_starting_position(agent, mode='tictactoe'):
    if mode == 'tictactoe':
        print(agent.predict_Q(np.zeros(9,)).reshape((3, 3)))
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

    tictactoe = KaggleTicTacToe()
    connect4 = KaggleConnectX(rows=6, columns=7, inarow=4)
    connect3 = KaggleConnectX(rows=4, columns=4, inarow=3)


    env = connect3

    starting_position = np.zeros((1, 9))

    size_memory = 4*512
    update_period = 2*512

    data_path = os.path.join(os.environ['HOME'], 'rl', 'reinforcement_learning', 'data')
    model_path = data_path + os.path.join(data_path, 'latest_network2')
    eps_scheduler = DecayingEpsilonScheduler(eps=1, decay_scale=20000, minimum=0.1)

    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        epsilon_scheduler=eps_scheduler,
        experience_replay=True,
        size_Q_memory=size_memory,
        fixed_target_weights=True,
        update_period_fixed_target_weights=update_period,
        self_play=True,
        save_model_path=model_path,
    )

    try:
        agent.load_model(agent.save_model_path, load_memory=True)
    except Exception as e:
        print('-COULD NOT LOAD AGENT-')
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


    agent.save_model('Q', agent.save_model_path, overwrite=True, save_memory=True)

