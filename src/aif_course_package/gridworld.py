import matplotlib.pylab as plt
from gym.spaces import Tuple, Discrete
from itertools import product
import numpy as np

        
class GridWorld:
    def __init__(self):
        self.observation_space = Tuple((Discrete(2, start = 1), Discrete(2, start = 1)))
        self.action_space = Discrete(4)

        self.rewards = dict()
        self.rewards[1, 1] = {0: -1, 1: -1, 2: -1, 3: -1}
        self.rewards[1, 2] = {0: -1, 1: -1, 2: -1, 3:  5}
        self.rewards[2, 1] = {0: -1, 1: -1, 2: -1, 3: -1}

        self.action2states = dict()
        self.action2states[1, 1] = {0: (1, 1), 1: (1, 1), 2: (1, 2), 3: (2, 1)}
        self.action2states[1, 2] = {0: (1, 1), 1: (1, 2), 2: (1, 2), 3: (2, 2)}
        self.action2states[2, 1] = {0: (2, 1), 1: (1, 1), 2: (2, 2), 3: (2, 1)}

        self.action2meaning = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}

        self.plot = GridWorldPlot()


    def reset(self):
        self.current_state = (2, 1)
        return self.current_state
    

    def step(self, action):
        s = self.current_state
        s_ = self.action2states[s][action]
        
        if s_ == (2, 2):
            done = True
        else:
            done = False
        
        r = self.rewards[s][action]
        info = {'old_state': s, 'action': a, 'reward': r, 'new_state': s_, 'episode_finished': done}
        self.current_state = s_

        return r, s_, done, info
    

    def plot_current_position(self):
        self.plot.plot_agent(self.current_state)


class GridWorldPlot:

    def __init__(self):
        self.state2coordinate = {}
        self.state2coordinate[1, 1] = (-0.05, 0.05)
        self.state2coordinate[1, 2] = (0.95, 0.05)
        self.state2coordinate[2, 1] = (-0.05, 1.05)

        self.state2agent = {}
        self.state2agent[1, 1] = (0, 0)
        self.state2agent[1, 2] = (1, 0)
        self.state2agent[2, 1] = (0, 1)
        self.state2agent[2, 2] = (1, 1)

        self.stateaction2coordinate = dict()
        self.stateaction2coordinate[1, 1] = {0: (-0.45, 0.025), 1: (-0.05, -0.425), 2: (0.375, 0.025), 3: (-0.05, 0.45)}
        self.stateaction2coordinate[1, 2] = {0: (0.55, 0.025), 1: (0.95, -0.425), 2: (0.95, 0.425), 3: (1.375, 0.025)}
        self.stateaction2coordinate[2, 1] = {0: (-0.45, 1.025), 1: (0.375, 1.025), 2: (-0.05, 0.575), 3: (-0.05, 1.45)}

        self.rewards = dict()
        self.rewards[1, 1] = {0: -1, 1: -1, 2: -1, 3: -1}
        self.rewards[1, 2] = {0: -1, 1: -1, 2:  5, 3: -1}
        self.rewards[2, 1] = {0: -1, 1: -1, 2: -1, 3: -1}


    @staticmethod
    def drawline(x, y):
        plt.plot([x[0], y[0]], [x[1], y[1]], 'k--', alpha = 0.8)


    def plot_box(self):
        cliff = [[1, 1], [2, 3]]
        plt.figure(figsize = (6, 6))
        plt.imshow(cliff, cmap = 'Greys', alpha = 0.5)
        self.drawline([-0.5, 0.5], [1.5, 0.5])
        self.drawline([0.5, 1.5], [0.5, -0.5])
        plt.xticks([0, 1], ['1', '2'])
        plt.yticks([0, 1], ['1', '2'])
        plt.text(-0.075, 1.25, 'start')
        plt.text(0.95, 1.25, 'end')


    def plot_center_position(self, state, value):
        x1, x2 = self.state2coordinate[state]
        plt.text(x1, x2, value)


    def plot_outer_position(self, state, action, value):
        x1, x2 = self.stateaction2coordinate[state][action]
        plt.text(x1, x2, value)


    def plot_state_values(self, V):
        self.plot_box()
        for state in V.keys():
            self.plot_center_position(state, V[state])

        
    def plot_action_values(self, Q):
        self.plot_box()
        for state in Q.keys():
            for action in Q[state].keys():
                self.plot_outer_position(state, action, Q[state][action])


    def plot_agent(self, state):
        self.plot_box()
        x1, x2 = self.state2agent[state]
        plt.scatter(x1, x2, marker = 'o', color = 'purple')


    def plot_game(self):
        self.plot_action_values(self.rewards)
        self.plot_agent((2, 1))


class ValueAgent:
    def __init__(self, environment):
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self.initialize_state_and_action_values()

    def initialize_state_and_action_values(self):
        state_indices = []
        for space in self.observation_space.spaces:
            state_idx = []
            for i in range(space.start, space.start+space.n, 1):
                state_idx.append(i)
            state_indices.append(state_idx)

        self.V = dict()
        for state in product(state_idx1, state_idx2):
            self.V[state] = 0.0
        self.Q = dict()
        for state in product(state_idx1, state_idx2):
            self.Q[state] = dict()
            for action in range(self.action_space.start, self.action_space.start+self.action_space.n, 1):
                self.Q[state][action] = 0.0

    def action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            a = self.action_space.sample()
        else:
            a = list(self.Q[state].keys())[np.argmax(list(self.Q[state].values()))]
        
        return a