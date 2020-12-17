from environments import KaggleTicTacToe
import numpy as np
from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent

print('\n------------\n')
const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes


if __name__ == '__main__':

    env = KaggleTicTacToe()

    starting_position = np.zeros((1, 9))

    size_memory = 1024, env.action_space.n + 1

    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        size_Q_memory=size_memory,
        epsilon_scheduler=const_scheduler(0.15),
        experience_replay=True,
        fixed_target_weights=False
    )

    #total_reward = agent.play(10000, policy='random')

    print('starting position:\n', agent.Q(starting_position).numpy().reshape((3,3)))

    agent.train(15000)

    print('starting position after train:', agent.Q(starting_position).numpy())

    print(f'TESTING AGENT\n')
    total_reward = agent.play(10000, policy='random')
