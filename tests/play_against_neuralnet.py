import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from environments import KaggleTicTacToe, KaggleConnectX
from function_approximator_agents import DeepQLearningAgent

REWARD_MSG = {
    0: 'Its a draw!',
    -1: 'You won!',
    1: 'You lost!'
}


def get_player_action(valid_actions):

    while True:

        try:
            action = input(f'Choose a wise action from ({list(valid_actions)}): ')
        except EOFError:
            exit()

        if action == 'q' or action == 'x':
            exit()

        try:
            action = int(action)
            if action in valid_actions:
                return action
        except ValueError:
            pass  # just try again...


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

    datapath = os.path.join(os.environ['HOME'], 'rl', 'reinforcement_learning', 'data')

    if game == 'tictaotoe':
        env = KaggleTicTacToe()
        model_path = os.path.join(datapath, 'tictactoe_network')
    elif game == 'connect4':
        env = KaggleConnectX(rows=6, columns=7, inarow=4)
        model_path = os.path.join(datapath, 'connect4net')
    elif game == 'connect3':
        env = KaggleConnectX(rows=4, columns=4, inarow=3)
        model_path = os.path.join(datapath, 'connect3net')

    agent = DeepQLearningAgent(
        env=env,
        policy='greedy',
        save_model_dir=model_path,
    )

    agent.load_model(model_path, load_memory=False)

    while True:
        print('\nstarting round ...')
        play_one_round(agent)


if __name__ == '__main__':
    main()
