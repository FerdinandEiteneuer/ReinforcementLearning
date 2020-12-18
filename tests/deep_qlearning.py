from environments import KaggleTicTacToe, KaggleConnectX
import numpy as np
from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent

import sys
import tensorflow as tf

np.set_printoptions(precision=3)

print('\n------------\n')
const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes

def predict_starting_position(agent):
    print(agent.predict(9*[0]).reshape((3,3)))

def pred(model):
    vals = np.zeros((9, 18))
    vals[:,-1] = np.eye(9)
    print('using', vals)
    print(model(vals).numpy().reshape((3,3)))


if __name__ == '__main__':
    """
    Train the agent on tictactoe.
    """
    env = KaggleTicTacToe()

    starting_position = np.zeros((1, 9))

    # one memory point would be [state, action(one_hot), target] -> 9+9+1 dimensions.
    # the last one is the target r + max_a' Q(s',a')
    size_memory = 1*1024, env.observation_space.n + env.action_space.n + 1
    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        epsilon_scheduler=const_scheduler(0.25),
        experience_replay=True,
        size_Q_memory=size_memory,
        fixed_target_weights=True,
        update_period_fixed_target_weights=500,
    )

    """
    pred(agent.Q)

    model2 = tf.keras.models.clone_model(agent.Q)
    model2.set_weights(agent.Q.get_weights())
    model2.compile()

    pred(model2)

    sys.exit()

    #total_reward = agent.play(1000, policy='random') # random play: about ~29% winrate

    """

    for i in range(10):
        agent.train(3000, policy='eps_greedy')
        agent.play(1000, policy='greedy')

    print('starting position after train')
    predict_starting_position(agent)

    print(f'TESTING AGENT\n')
    total_reward = agent.play(10000, policy='random')
