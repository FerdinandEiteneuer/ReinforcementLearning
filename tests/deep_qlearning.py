from environments import KaggleTicTacToe
import numpy as np
from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent

np.set_printoptions(precision=3)

print('\n------------\n')
const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes

def predict_starting_position(agent):
    print(agent.predict(9*[0]).reshape((3,3)))

if __name__ == '__main__':
    """
    Train the agent on tictactoe.
    """
    env = KaggleTicTacToe()

    starting_position = np.zeros((1, 9))

    # one memory point would be [state, action, target] -> 11 dimensions.
    # first ten are [*s, a] the last one is the target r + max_a' Q(s',a')
    size_memory = 1024, env.observation_space.n + 1 + 1

    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        size_Q_memory=size_memory,
        epsilon_scheduler=const_scheduler(0.15),
        experience_replay=True,
        fixed_target_weights=False
    )

    #total_reward = agent.play(1000, policy='random') # random play: about ~29% winrate


    agent.train(15000)

    print('starting position after train')
    predict_starting_position(agent)

    print(f'TESTING AGENT\n')
    total_reward = agent.play(10000, policy='random')
