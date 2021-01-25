import gym

from function_approximator_agents import DeepQLearningAgent
import utils

def success_condition(info):
    return info['total_reward'] > 190

if __name__ == '__main__':

    # ENVIRONMENT
    env = gym.make('CartPole-v1')

    # AGENT CONFIGURATION
    size_memory = 2000
    update_period = 400

    eps_scheduler = utils.LinearlyDecreasingEpsilonScheduler(
        eps=1,
        end_of_decrease=20_000,
        minimum=0.1
    )

    input_shape = utils.get_input_shape(env)  # 4
    n_outputs = utils.get_output_neurons(env)  # 2

    network = utils.create_sequential_dense_net(
        input_shape=input_shape,
        n_outputs=n_outputs,
        hidden_layers=2,
        neurons_per_layer=100,
        final_activation_function='relu',
        p_dropout=0
    )

    agent = DeepQLearningAgent(
        Q=network,
        env=env,
        gamma=0.98,
        epsilon_scheduler=eps_scheduler,
        experience_replay=True,
        size_memory=size_memory,
        fixed_target_weights=False,
        update_period_fixed_target_weights=update_period,
        starting_learning_rate=10*10**(-5),
        self_play=False,  # self_play must be False in this env
        success_condition=success_condition,
    )

    agent.train_and_play(
        train=400,  # using eps greedy
        play=20,  # using greedy
        repeat=20
    )
