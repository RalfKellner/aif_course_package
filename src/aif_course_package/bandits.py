import random
import operator
import numpy as np
import matplotlib.pyplot as plt


#general bandit defined by win probability and a pull method giving a reward of -1 or 2
class Bandit():
    def __init__(self, pi):
        self.pi = pi
        
    def pull(self):
        if np.random.uniform(size = 1)[0] <= self.pi:
            return 2.0
        else:
            return -1.0
        
# a class for a multi-armed bandit with n bandits
class MultiArmedBandit():

    # a function which initializes an instance of Mulitbandit
    def __init__(self, winning_probabilities):
        # set the user specified winning probabilities and empty dictionaries for bandits, rewards and q-values
        self.winning_probabilities = winning_probabilities
        self.bandits = {}
        # define each bandit instance 
        for i, prob in enumerate(self.winning_probabilities): 
            self.bandits[f'bandit_{i + 1}'] = Bandit(prob)

    def step(self, action):
        # pull the bandit with the specified action
        reward = self.bandits[action].pull()
        # update the reward and q-value memory
        return reward


class BanditValueAgent:
    def __init__(self, environment):
        self.num_bandits = len(environment.winning_probabilities)
        self.rewards = {}
        self.q_values = {}
        # define a memory for each bandit 
        for i, prob in enumerate(range(self.num_bandits)): 
            self.rewards[f'bandit_{i + 1}'] = []
            self.q_values[f'bandit_{i + 1}'] = 0.0


      # an epsilon greedy function for choosing the bandit
    def epsilon_greedy(self, epsilon):
        # the bandit is either chosen randomly
        if np.random.uniform(size = 1)[0] <= epsilon:
            return random.choice(list(self.rewards.keys()))
        # or by the highest q-value
        else:
            return max(self.q_values.items(), key=operator.itemgetter(1))[0]
        

    def update(self, action, reward, stationary, **kwargs):
        # update the reward memory
        self.rewards[action].append(reward)
        # update the q-value memory
        if stationary:
            self.q_values[action] +=  kwargs['eta'] * (r - self.q_values[a])
        else:
            self.q_values[action] =  np.mean(self.rewards[action])

    
    def print_qvalues(self):
        for bandit in self.q_values.keys():
            print(f'Bandit {bandit} has a q-value of {self.q_values[bandit]:.4f}, it has been played {len(self.rewards[bandit])} times')

    
    def get_current_estimates(self):
        return  list(self.q_values.values())  


class BanditPolicyAgent():
    def __init__(self, environment):
        # an array representing action preferences
        self.policy = {}
        # define each bandit instance 
        for i, _ in enumerate(environment.winning_probabilities): 
            self.policy[f'bandit_{i + 1}'] = 1.0
        self.action2idx = {}
        for i, key in enumerate(self.policy.keys()):
            self.action2idx[key] = i

    # the softmax function which turns the policy into action probabilities
    def softmax(self):
      preferences = np.array(list(self.policy.values()))
      return np.exp(preferences) /np.sum(np.exp(preferences))

    # randomly select an action, given action probabilities, basically, this is a random draw from a Multinoulli distribution  
    def act(self, deterministic = False):
        if deterministic == True:
          idx = np.argmax(self.softmax())
        else:
          idx =  list(np.random.multinomial(n = 1, pvals = self.softmax())).index(1)
        return list(self.policy.keys())[idx]

    # as explained above, the derivative for the log of the action probability with respect to its preference
    def action_grad(self, action):
      return 1 - self.softmax()[self.action2idx[action]]
    
    # updating rule given a reward and an acion
    def update(self, reward, action, eta):
        self.policy[action] += eta * reward * self.action_grad(action)

    def get_current_probabilities(self):
        return  list(self.softmax())
