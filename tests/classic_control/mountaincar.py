import gym

from function_approximator_agents import DeepQLearningAgent
import utils

def success_condition(info):
    return info['episode_length'] < 199


if __name__ == '__main__':

    # ENVIRONMENT
    env = gym.make('MountainCar-v0')

    # AGENT CONFIGURATION
    size_memory = 200
    update_period = 10

    eps_scheduler = utils.LinearlyDecreasingEpsilonScheduler(
        eps=1,
        end_of_decrease=7_000,
        minimum=0.2,
    )

    input_shape = utils.get_input_shape(env)  # 2
    n_outputs = utils.get_output_neurons(env)  # 3 actions (left, nothing, right)

    network = utils.create_sequential_dense_net(
        input_shape=input_shape,
        n_outputs=n_outputs,
        hidden_layers=1,
        neurons_per_layer=20,
        final_activation_function='linear',
        p_dropout=0
    )

    agent = DeepQLearningAgent(
        Q=network,
        env=env,
        gamma=0.98,
        epsilon_scheduler=eps_scheduler,
        experience_replay=True,
        size_memory=size_memory,
        fixed_target_weights=True,
        update_period_fixed_target_weights=update_period,
        starting_learning_rate=10*10**(-4),
        self_play=False,  # self_play must be False in this env
        success_condition=success_condition,
    )


    agent.train_and_play(
        train=200,  # using eps greedy
        play=10,  # using greedy
        repeat=20,
    )
