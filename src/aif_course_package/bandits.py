import numpy as np
import random
import operator
import matplotlib.pylab as plt

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
class MultiBandit():

  # a function which initializes an instance of Mulitbandit
  def __init__(self, winning_probs):
    # set the user specified winning probabilities and empty dictionaries for bandits, rewards and q-values
    self.winning_probs = winning_probs
    self.bandits = {}
    self.rewards = {}
    self.q_values = {}

    # define each bandit instance and a rewards and q-value memory for it
    for i, prob in enumerate(self.winning_probs): 
      self.bandits[f'bandit_{i + 1}'] = Bandit(prob)
      self.rewards[f'bandit_{i + 1}'] = []
      self.q_values[f'bandit_{i + 1}'] = 0.0

  # an epsilon greedy function for choosing the bandit
  def epsilon_greedy(self, epsilon):
    # the bandit is either chosen randomly
    if np.random.uniform(size = 1)[0] <= epsilon:
        return random.choice(list(self.bandits.keys()))
    # or by the highest q-value
    else:
        return max(self.q_values.items(), key=operator.itemgetter(1))[0]

  # a function for playing n times at all slot machines and examining the progress of q-values over time
  def simulate(self, n, eps, eps_decay, savefig = None):

    # for each simulation we memorize the q-values estimates over time to visualize them after we are done
    q_estimates_over_t = {}
    for key in self.bandits.keys():
      q_estimates_over_t[key] = []
    xticks = []

    # now in a loop of n, select the bandit according to epsilon-greedy, memorize the reward for this bandit and 
    # calculate the current q-value estimate for this bandit
    print('-'*100)
    print(f'Epsilon at start: {eps:.4f}')
    for t in range(n):
      bandit_t = self.epsilon_greedy(eps)
      self.rewards[bandit_t].append(self.bandits[bandit_t].pull())
      self.q_values[bandit_t] = np.mean(self.rewards[bandit_t])

      # save some information over time for plotting
      if (t % 10) == 0:
        xticks.append(t)
        for key in self.bandits.keys():
          q_estimates_over_t[key].append(self.q_values[key])

        eps *= eps_decay

    print(f'Epsilon at the end: {eps:.4f}')
    print('-'*100)
    print('')
    # visualize, set figure size a little larger than standard
    plt.figure(figsize =(8, 4))

    #print the evolution of estimated q values
    for i, key in enumerate(self.bandits.keys()):
      plt.plot(xticks, q_estimates_over_t[key], label = key)
    
    plt.ylim(-1,2)
    plt.legend(ncol = len(self.bandits.keys()))
    plt.xlabel('games played')
    plt.ylabel(r'$\hat{Q}(a_0|s_0)$')
    if savefig:
       plt.savefig(savefig)
    plt.show()

    print('')
    print('-'*100)
    for i, key in enumerate(self.bandits.keys()):
      print(f"Bandit {i+1} has been played {len(self.rewards[key])} times")
    print('-'*100)