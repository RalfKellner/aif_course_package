import matplotlib.pylab as plt
from gym.spaces import MultiDiscrete, Discrete
from itertools import product
import numpy as np
import gym


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
        self.stateaction2coordinate[1, 2] = {0: (0.55, 0.025), 1: (0.95, -0.425), 2: (1.375, 0.025), 3: (0.95, 0.425)}
        self.stateaction2coordinate[2, 1] = {0: (-0.45, 1.025), 1: (-0.05, 0.575), 2: (0.375, 1.025), 3: (-0.05, 1.45)}

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


    def plot_agent(self, state):
        x1, x2 = self.state2agent[state]
        plt.scatter(x1, x2, marker = 'o', color = 'purple')


    def plot_state_values(self, V, title = None, savepath = None):
        self.plot_box()
        if title:
            plt.title(title)
        for state in V.keys():
            self.plot_center_position(state, np.round(V[state], 2))
        if savepath:
            plt.savefig(savepath)

        
    def plot_action_values(self, Q, title = None, savepath = None):
        self.plot_box()
        if title:
            plt.title(title)
        for state in Q.keys():
            for action in Q[state].keys():
                self.plot_outer_position(state, action, np.round(Q[state][action], 2))
        if savepath:
            plt.savefig(savepath)


    def plot_action_probabilities(self, pi_, title = None, savepath = None):
        self.plot_box()
        if title:
            plt.title(title)
        for state in pi_.keys():
            for action in pi_[state].keys():
                self.plot_outer_position(state, action, np.round(pi_[state][action], 2))
        if savepath:
            plt.savefig(savepath)


    def plot_game(self, title = None, savepath = None):
        self.plot_action_values(self.rewards)
        self.plot_agent((2, 1))
        if title:
            plt.title(title)
        if savepath:
            plt.savefig(savepath)


class GridWorld(gym.Env):

    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        self.observation_space = MultiDiscrete([2, 2])
        self.action_space = Discrete(4)

        self.rewards = dict()
        self.rewards[0, 0] = {0: -1, 1: -1, 2: -1, 3: -1}
        self.rewards[0, 1] = {0: -1, 1: -1, 2: -1, 3:  5}
        self.rewards[1, 0] = {0: -1, 1: -1, 2: -1, 3: -1}

        self.action2states = dict()
        self.action2states[0, 0] = {0: [0, 0], 1: [0, 0], 2: [0, 1], 3: [1, 0]}
        self.action2states[0, 1] = {0: [0, 0], 1: [0, 1], 2: [0, 1], 3: [1, 1]}
        self.action2states[1, 0] = {0: [1, 0], 1: [0, 0], 2: [1, 1], 3: [1, 0]}

        self.action2meaning = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
        self.plot = GridWorldPlot()


    def reset(self):
        self.current_state = np.array([1, 0])
        return self.current_state
    

    def step(self, action):
        s = self.current_state
        s_ = self.action2states[tuple(s)][action]
        
        if s_ == [1, 1]:
            done = True
        else:
            done = False
        
        r = self.rewards[tuple(s)][action]
        info = {'old_state': s, 'action': action, 'reward': r, 'new_state': s_, 'episode_finished': done}
        self.current_state = s_

        return np.array(s_), r, done, info

