from environments import KaggleTicTacToe, KaggleConnectX
import numpy as np
from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent

import sys
import tensorflow as tf

np.set_printoptions(precision=3)

print('\n------------\n')
const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes


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



if __name__ == '__main__':

    env = KaggleTicTacToe()
    #env = KaggleConnectX(rows=3, columns=3, inarow=3)

    starting_position = np.zeros((1, 9))

    # one memory point would be [state, action(one_hot), target]
    # the last one is the target r + max_a' Q(s',a', fixed_weights)
    size_memory = 2*1024, len(env.observation_space) + env.action_space.n + 1
    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        epsilon_scheduler=const_scheduler(0.10),
        experience_replay=True,
        size_Q_memory=size_memory,
        fixed_target_weights=True,
        update_period_fixed_target_weights=2000,
    )

    predict_starting_position(agent)

    agent.train_and_play(train=5000, play=1000, repeat=10, func=predict_starting_position)

    """
    #total_reward = agent.play(10000, policy='random') # random play: about ~29% winrate
    
    for i in range(5):
        print('\n\nTRAINING / PLAY LOOP', i)
        agent.train(8000, policy='eps_greedy')
        agent.play(1000, policy='greedy')

        print('starting position after train')
        predict_starting_position(agent)
    """
    print(f'TESTING AGENT IN RANDOM PLAY\n')
    total_reward = agent.play(1000, policy='random')
    print(total_reward)
