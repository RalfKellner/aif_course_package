import matplotlib.pylab as plt
import numpy as np
import random


########################################
# Gridworld environment for simulation
########################################

# Let us define an environment: all we need to know which actions are possible
# which state follows an action and which reward is collected, given a state-action pair 
class GridworldEnvironment:
    def __init__(self):
        # the environment keeps the information of possible actions and following states in an
        # action dictionary
        self.actions = {'A1': {'down': 'A2', 'right': 'B1'},
                        'A2': {'up': 'A1', 'right': 'B2'},
                        'B1': {'left': 'A1', 'down': 'B2'},
                        'B2': {'final': 'B2'}}

        # the rewards dictionary includes the information of rewards which follow a state-action pair
        self.rewards = {'A1': {'down': -1, 'right': -1},
                        'A2': {'up': -1, 'right': -1},
                        'B1': {'left': -1, 'down': +5},
                        'B2': {'final': 0}}


        self.reward_dict = {}.fromkeys(self.rewards)
        for s in self.reward_dict.keys():
            self.reward_dict[s] = {}.fromkeys(self.actions[s])

        for s in self.reward_dict.keys():
            for a in self.reward_dict[s].keys():
                self.reward_dict[s][a] = -1
        self.reward_dict['B1']['down'] = 5

        self._init_state_values()
        self._init_action_values()
        self.reset()

    def reset(self):
      self.current_position = 'A2'

    # the environment has one environment specific function which takes a current state, an action
    # it returns the reward which has been collected
    # the next state 
    # and if the next state is the final state    
    def step(self, action):
        r  = self.rewards[self.current_position][action]
        s_ = self.actions[self.current_position][action]
        if s_ == 'B2':
            done = True
        else:
            done = False
        
        self.current_position = s_
                
        return (r, s_, done)
    

    def plot_reward_map(self, savepath = None):
        for s in self.Q_vals.keys():
            for a in self.Q_vals[s].keys():
                self.Q_vals[s][a] = self.reward_dict[s][a]
        if savepath:
            self.plot_action_values('Reward Map', savepath=savepath, round = 0)
        else:
            self.plot_action_values('Reward Map', round = 0)


    def plot_state_values(self, title, savepath = None, round = 2):
        cliff = [[1, 1], [2, 3]]

        plt.figure(figsize = (6, 6))
        plt.imshow(cliff, cmap = 'Greys', alpha = 0.5)
        self._drawline([-0.5, 0.5], [1.5, 0.5])
        self._drawline([0.5, 1.5], [0.5, -0.5])
        plt.xticks([0, 1], ['A', 'B'])
        plt.yticks([0, 1], ['1', '2'])

        plt.text(-0.05, 0.05, np.round(self.V_s['A1'], round))
        plt.text(-0.05, 1.05, np.round(self.V_s['A2'], round))
        plt.text(0.95, 0.05, np.round(self.V_s['B1'], round))
        plt.text(0.95, 1.05, 'End')

        plt.title(title)
        
        if not(savepath == None):
            plt.savefig(savepath)

        plt.show()
        

    def plot_action_values(self, title, savepath = None, round = 2):
        cliff = [[1, 1], [2, 3]]

        plt.figure(figsize = (6, 6))
        plt.imshow(cliff, cmap = 'Greys', alpha = 0.5)
        self._drawline([-0.5, 0.5], [1.5, 0.5])
        self._drawline([0.5, 1.5], [0.5, -0.5])
        plt.xticks([0, 1], ['A', 'B'])
        plt.yticks([0, 1], ['1', '2'])

        plt.text(-0.05, 0.45, np.round(self.Q_vals['A1']['down'], round))
        plt.text(0.35, 0.05, np.round(self.Q_vals['A1']['right'], round))
        plt.text(-0.05, 0.60, np.round(self.Q_vals['A2']['up'], round))
        plt.text(0.35, 1.05, np.round(self.Q_vals['A2']['right'], round))
        plt.text(0.55, 0.05, np.round(self.Q_vals['B1']['left'], round))
        plt.text(0.95, 0.45, np.round(self.Q_vals['B1']['down'], round))
        #plt.text(-0.05, 1.05, 'Start')
        plt.text(0.95, 1.05, 'End')

        plt.title(title)
        
        if not(savepath == None):
            plt.savefig(savepath)

        plt.show()

    @staticmethod
    def _drawline(x, y):
        plt.plot([x[0], y[0]], [x[1], y[1]], 'k--', alpha = 0.8)


    def _init_state_values(self):
        self.V_s = {}.fromkeys(self.rewards, 0.0)


    def _init_action_values(self):
        self.Q_vals = {}.fromkeys(self.actions)
        for s in self.Q_vals.keys():
            self.Q_vals[s] = {}.fromkeys(self.actions[s], 0.0)

 
# an epsilon_greedy agent
class EpsilonGreedyAgent:
    def action(self, state, environment, epsilon):
        if random.uniform(0, 1) <= epsilon:
            return random.choice(list(environment.actions[state].keys()))
        else: 
            return list(environment.Q_vals[state].keys())[np.argmax(list(environment.Q_vals[state].values()))]      


# a new policy agent class
class PolicyAgent():
    def __init__(self, default_preference):
        # default preference value can be set an initialization
        self.policy = {
            'A1': {'down': default_preference, 'right': default_preference},
            'A2': {'up': default_preference, 'right': default_preference},
            'B1': {'left': default_preference, 'down': default_preference},
            'B2': {'final': default_preference}
        }

    # general softmax function
    def softmax(self, state):
      z = np.array(list(self.policy[state].values()))
      return np.exp(z) / np.sum(np.exp(z))

    # get probability for specific state and action
    def get_grad(self, state, action):
      return 1 - self.softmax(state)[list(self.policy[state].keys()).index(action)]

    # get the probability in
    # a method for choosing an action   
    def action(self, state):
        # given a state, determine action probability distribution
        probs = self.softmax(state)
        # randomly draw an action from the probability distribution, gives an one-hot encoded array, e.g., [0, 1, 0] for action number two
        prob_move = np.random.multinomial(n = 1, pvals = probs)
        # get the index for the action to choose
        move_idx = np.where(prob_move == 1)[0][0]
        # choose the actual action for the action index
        move = list(self.policy[state].keys())[move_idx]
        return move
