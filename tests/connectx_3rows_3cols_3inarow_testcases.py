import numpy as np

TESTCASES = {

    1: {
        'board': [0, 0, 0,
                  1, 0, 0,
                  1, 2, 2],
        'correct_action': 0
    },

    2: {
        'board': [0, 0, 0,
                  2, 1, 2,
                  1, 2, 1],
        'correct_action': 0
    },

    3: {
        'board': [0, 0, 0,
                  2, 1, 0,
                  2, 1, 0],
        'correct_action': 1
    },

    4: {
        'board': [0, 0, 2,
                  0, 0, 2,
                  0, 1, 1],
        'correct_action': 0
    },

    5: {
        'board': [0, 0, 0,
                  1, 1, 2,
                  2, 1, 2],
        'correct_action': 1
    },

    6: {
        'board': [0, 0, 0,
                  2, 0, 0,
                  1, 1, 2],
        'correct_action': 1
    }
}

def check(agent):

    if agent.env.dtype_state == tuple:
        transform = lambda state: tuple(state)
    elif agent.env.dtype_state == np.ndarray:
        transform = lambda state: np.array(state)

    failed = 0

    for n in TESTCASES:
        testcase = TESTCASES[n]

        board = testcase['board']
        correct_action = testcase['correct_action']

        board = transform(board)

        action = agent.policy(board)

        if action != correct_action:
            failed += 1
            print(f'\nagent failed test {n=}. {correct_action=}, but chose {action} in state:')
            print(f'{np.array(board).reshape((3,3))}')

    if failed > 0:
        print(f'\nagent failed {failed}/{n} tests.')
    else:
        print(f'\nagent passed all tests.')