# external libraries
import numpy as np
from tensorflow.keras.models import load_model

# this package
from environments import KaggleTicTacToe, KaggleConnectX
from function_approximator_agents import DeepQLearningAgent
from utils import LATEST_NETWORK_PATH

REWARD_MSG = {
    0: 'Its a draw!',
    -1: 'You won!',
    1: 'You lost!'
}

def get_player_action(valid_actions):

    while True:

        try:
            action = int(input(f'Choose a wise action from ({list(valid_actions)}): '))
            if action in valid_actions:
                return action
        except Exception as e:
            continue


def play_one_round(agent):

    env = agent.env

    state = env.reset()
    terminal = False

    while not terminal:

        agent_action = agent.policy(state)

        state, reward, terminal, info = env.player_action(agent_action)

        print(f'after agent step')
        print(env)

        if terminal:
            print(REWARD_MSG[reward])
            return reward

        valid_actions = env.get_allowed_actions(state)
        player_action = get_player_action(valid_actions)


        state, reward, terminal, info = env.execute_opponent_action(player_action)

        print(f'after player step')
        print(env)

        if terminal:
            print(REWARD_MSG[reward])
            return reward


def main():


    game = 'connect4'

    if game == 'tictaotoe':
        env = KaggleTicTacToe()
        model = load_model('/home/ferdinand/rl/reinforcement_learning/data/tictactoe_network')
    elif game == 'connect4':
        print('trying to load', LATEST_NETWORK_PATH)
        env = KaggleConnectX(rows=6, columns=7, inarow=4)
        model = load_model(LATEST_NETWORK_PATH)


    agent = DeepQLearningAgent(
        env=env,
        policy='greedy'
    )

    agent.Q = model

    while True:
        print('\nstarting round ...')
        play_one_round(agent)


if __name__ == '__main__':
    main()

