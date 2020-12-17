from environments import KaggleTicTacToe
import numpy as np
from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent


const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes


if __name__ == '__main__':

    env = KaggleTicTacToe()


    size_memory = 256, env.action_space.n + 1

    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        size_Q_memory=size_memory,
        epsilon_scheduler=const_scheduler(0.1),
        experience_replay=True,
        fixed_target_weights=False
    )

    agent.train(5000)

    total_reward = agent.play(5000, policy='random')
    print(f'{total_reward=}')