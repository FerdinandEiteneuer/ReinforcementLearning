import gym
import plotly
import plotly.graph_objects as go
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from environments import KaggleTicTacToe, KaggleConnectX
import tabular_agents

N0 = 5000

def q_init(state, action):
    return 0

def eps_scheduler(episodes):
    return max(0.1, N0/(N0 + episodes))
    return 1
    return N0 / (N0 + episodes)
    return 1/episodes

def alpha_scheduler(episodes):
    return 0.1
    a0 = 500
    return min(0.1, a0 / (a0 + episodes))


def training_cycle(agent, cycles, learn, test):

    agent.wins_ = []
    for j in range(cycles):
        n_wins = agent.learn_and_test(learn, test, random=False, print_valuefunction=False)

        agent.wins_.append(100 * n_wins / test)

    print('AGENT DONE')


def print_Q(Q, state, agent):
    values = [Q[state, a] for a in agent.A]
    printout = ''
    for i, val in enumerate(values):
        printout += f'Q[{state}, {i}] = val'
    print(printout)


if __name__ == '__main__':


    #env = gym.make('Blackjack-v0')

    #env = gym.make('FrozenLake-v0')
    #env = gym.make('FrozenLake8x8-v0')


    #env = tabular_agents.envs.WindyGridWorld('windy_gridworld', 'standard')
    #env = tabular_agents.envs.WindyGridWorld('no_wind', 'standard')
    #env = tabular_agents.envs.Easy21()

    #env = KaggleConnectX(rows=3, columns=3, inarow=3)
    env = KaggleTicTacToe()

    r = env.reset
    s = env.step


    sarsa_agent = tabular_agents.SarsaAgent(
        env=env,
        gamma=1,
        q_value_initialization=q_init,
        epsilon_scheduler=eps_scheduler,
        alpha_scheduler=alpha_scheduler
    )


    mceveryvisit_agent = tabular_agents.MonteCarloEveryVisitAgent(
        env=env,
        gamma=1,
        q_value_initialization=q_init,
        epsilon_scheduler=eps_scheduler,
    )

    qlearning_agent = tabular_agents.QLearningAgent(
        env=env,
        gamma=1,
        q_value_initialization=q_init,
        epsilon_scheduler=eps_scheduler,
        alpha_scheduler=alpha_scheduler,
        behaviour_policy='eps_greedy'
    )

    sarsalambda_agent = tabular_agents.SarsaLambdaAgent(
        env=env,
        gamma=1,
        q_value_initialization=q_init,
        alpha_scheduler=alpha_scheduler,
        epsilon_scheduler=eps_scheduler,
        lambd=0.3,
        forward_view=False,
        policy='eps_greedy',
    )

    #agent = mceveryvisit_agent
    #agent = sarsa_agent
    #agent = sarsalambda_agent
    agent = qlearning_agent

    agent.learn_and_test(n_train=2*10**4, n_test=3*10**3, random=False, print_valuefunction=False)
    agent.learn_and_test(n_train=2*10**4, n_test=3*10**3, random=False, print_valuefunction=False)
    agent.learn_and_test(n_train=2*10**4, n_test=3*10**4, random=False, print_valuefunction=False)
    #agent.play(episodes=1000, random=True)

    try:
        env.print_action_valuefunction(agent.Q)
    except:
        pass
        #print('could not print actionvalue function from env function')

    try:
        pass
        #agent.print_actionvalue_function()
    except:
        pass
        #print('could not print actionvalue function from agent')

    #training_cycle(agent, cycles=20, learn=2000, test=20)

    """
    Q = tabular_agents.envs.Easy21.get_valuefunction_numpy(agent.Q)
    fig = go.Figure(data=go.Surface(z=Q))
    fig.update_layout(title='test', width=900, height=800)
    fig.update_xaxes(title_text = 'player')
    fig.show()
    """
    

    # traces = [go.Scatter(y=agents[j].wins_, name=f'agent {j}, α={agents[j].alpha}') for j in range(N_agents)]

    # fig = go.Figure(data=traces)

    # fig.show()

    # fig = go.Figure(data=[go.Surface(z=arr, x=x, y=y)])

    # arr0 = -np.ones((32, 11))
    # arr1 = -np.ones((32,11))

    # for x in range(32):
    #     for y in range(11):
    #         q0 = max(agent.Q[(x, y, 0),a] for a in agent.A)
    #         q1 = max(agent.Q[(x, y, 1),a] for a in agent.A)
            
    #         arr0[x, y] = q0
    #         arr1[x, y] = q1

    # fig = go.Figure(data=[go.Surface(z=arr1, colorscale=[[0, 'red'], [0.5, 'orange'], [1, 'green']])])
    # fig.update_layout(title=f'Blackjack Valuefunction', width=900, height=800)

    # fig.show()



    #qmem = agent.q_memory
    #Q = agent.Q

    # x = agent.q_memory
    # y = x[1:] + [x[-1]]

    # x = np.array(x)
    # y = np.array(y)

    # diff = x - y

    # print(np.std(x))
    # print(np.mean(np.abs(diff)))