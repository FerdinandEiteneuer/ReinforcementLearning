from environments import TicTacToe

from function_approximator_agents.deep_qlearning_agent import DeepQLearningAgent


const_scheduler = lambda eps: lambda episodes: eps
decay_scheduler1 = lambda episodes: 1/episodes

if __name__ == '__main__':

    env = TicTacToe()
    size_memory = (1024, env.action_space.n + 1)

    agent = DeepQLearningAgent(
        env=env,
        gamma=1,
        size_Q_memory=size_memory,
        epsilon_scheduler=const_scheduler(1),
        experience_replay=True,
        fixed_target_weights=False
    )

    agent.train(1050)