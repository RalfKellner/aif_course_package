import matplotlib.pylab as plt
from gym.spaces import Tuple, Discrete
from itertools import product
import numpy as np
import tensorflow as tf
from collections import deque
   

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
        self.rewards[1, 2] = {0: -1, 1: -1, 2:  -1, 3: 5}
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


class GridWorld:

    def __init__(self):
        self.observation_space = Tuple((Discrete(2, start = 1), Discrete(2, start = 1)))
        self.action_space = Discrete(4)

        default_reward = -1
        self.rewards = dict()
        self.rewards[1, 1] = {0: default_reward, 1: default_reward, 2: default_reward, 3: default_reward}
        self.rewards[1, 2] = {0: default_reward, 1: default_reward, 2: default_reward, 3:  5}
        self.rewards[2, 1] = {0: default_reward, 1: default_reward, 2: default_reward, 3: default_reward}

        self.action2states = dict()
        self.action2states[1, 1] = {0: (1, 1), 1: (1, 1), 2: (1, 2), 3: (2, 1)}
        self.action2states[1, 2] = {0: (1, 1), 1: (1, 2), 2: (1, 2), 3: (2, 2)}
        self.action2states[2, 1] = {0: (2, 1), 1: (1, 1), 2: (2, 2), 3: (2, 1)}

        self.action2meaning = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
        self.plot = GridWorldPlot()


    def reset(self):
        self.current_state = (2, 1)
        return np.array(self.current_state)
    

    def step(self, action):
        s = self.current_state
        s_ = self.action2states[s][action]
        
        if s_ == (2, 2):
            done = True
        else:
            done = False
        
        r = self.rewards[s][action]
        info = {'old_state': s, 'action': action, 'reward': r, 'new_state': s_, 'episode_finished': done}
        self.current_state = s_

        return np.array(s_), r, done, info
    

    def plot_current_position(self):
        self.plot.plot_agent(self.current_state)


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

        state_idx1, state_idx2 = state_indices
        self.V = dict()
        for state in product(state_idx1, state_idx2):
            self.V[state] = 0.0
        self.Q = dict()
        for state in product(state_idx1, state_idx2):
            self.Q[state] = dict()
            for action in range(self.action_space.start, self.action_space.start+self.action_space.n, 1):
                self.Q[state][action] = 0.0

        self.V.pop((2, 2), None)
        self.Q.pop((2, 2), None)

    def action(self, state, epsilon):
        s = tuple(state)
        if np.random.rand() <= epsilon:
            a = self.action_space.sample()
        else:
            a = list(self.Q[s].keys())[np.argmax(list(self.Q[s].values()))]
        return a
    

class PolicyAgent:
    def __init__(self, environment, preference):
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self.initialize_state_and_action_values(preference)


    def initialize_state_and_action_values(self, preference):
        state_indices = []
        for space in self.observation_space.spaces:
            state_idx = []
            for i in range(space.start, space.start+space.n, 1):
                state_idx.append(i)
            state_indices.append(state_idx)

        state_idx1, state_idx2 = state_indices
        self.policy = dict()
        for state in product(state_idx1, state_idx2):
            self.policy[state] = dict()
            for action in range(self.action_space.start, self.action_space.start+self.action_space.n, 1):
                self.policy[state][action] = preference

        self.policy.pop((2, 2), None)


    def get_probabilities(self):
        pi_ = self.policy.copy()
        for state in pi_.keys():
            state_probabilities = self.state_softmax(state)
            for i, action in enumerate(pi_[state].keys()):
                pi_[state][action] = state_probabilities[i]
        return pi_


    # softmax function
    def state_softmax(self, state):
      s = tuple(state)
      z = np.array(list(self.policy[s].values()))
      return np.exp(z) / np.sum(np.exp(z))


    # get probability for specific state and action
    def grad(self, state, action):
      return 1 - self.state_softmax(state)[list(self.policy[state].keys()).index(action)]
    

    # a method for choosing an action   
    def action(self, state):
        # given a state, determine action probability distribution
        probs = self.state_softmax(state)
        # randomly draw an action from the probability distribution, gives an one-hot encoded array, e.g., [0, 1, 0] for action number two
        prob_move = np.random.multinomial(n = 1, pvals = probs)
        # get the index for the action to choose
        move_idx = np.where(prob_move == 1)[0][0]
        # choose the actual action for the action index
        move = list(self.policy[tuple(state)].keys())[move_idx]
        return move



class ValueFunctionAgent:
    def __init__(self):
        self.V = np.random.normal(size = 5)
        self.set_scaler()


    def s2f(self, state):
        s1, s2 = state
        x = np.array([s1, s2, s1**2, s2**2, s1*s2])   
        return self.scaler(x)


    def s2V(self, state):
        x = self.s2f(state)
        V_s = x.dot(self.V)
        return V_s
    
    @staticmethod
    def action(state):
        return np.random.choice(4, p = [0.25]*4)


    def return_state_values(self):
        V_hat = dict()
        for s in [np.array([1, 1]), np.array([2, 1]), np.array([1, 2])]:
            V_hat[tuple(s)] = self.s2V(s)
        return V_hat
    

    def set_scaler(self):
        states = [np.array([1, 1]), np.array([1, 2]), np.array([2, 1]), np.array([2, 2])]
        features = []
        for s in states:
            s1, s2 = s[0], s[1]
            features.append(np.array([s1, s2, s1**2, s2**2, s1*s2]))

        features = np.array(features)
        self.mean_ = np.mean(features, axis = 0)
        self.sd_ = np.std(features, axis = 0)

    
    def scaler(self, x):
        return (x - self.mean_) / self.sd_


class QNetworkAgent:
    def __init__(self, environment, hidden_neurons, activation):
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self.Q = self.make_qnn(hidden_neurons, activation)
        self.Q_target = self.make_qnn(hidden_neurons, activation)
        self.update_target_network()


    def make_qnn(self, hidden_neurons = 10, activation = 'elu'):
        fnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (len(self.observation_space),)),
            tf.keras.layers.Dense(hidden_neurons, activation = activation),
            tf.keras.layers.Dense(self.action_space.n)
        ])
        return fnn 
    

    def action(self, state, epsilon):
        s_tensor = tf.convert_to_tensor(state)
        s_tensor = tf.expand_dims(s_tensor, 0)
        if np.random.rand() <= epsilon:
            a = self.action_space.sample()
        else:
            action_values = self.Q(s_tensor, training = False)
            a = tf.argmax(action_values[0]).numpy()
        return a
    

    def update_target_network(self):
        self.Q_target.set_weights(self.Q.get_weights())


    def return_qvalue_dictionary(self):
        states = [np.array([2, 1]), np.array([1, 1]), np.array([1, 2])]
        Q_values = dict()
        for s in states:
            Q_values[tuple(s)] = dict()
            s_tensor = tf.convert_to_tensor(s)
            s_tensor = tf.expand_dims(s, 0)
            state_q_values = self.Q(s_tensor).numpy()
            for a, val in enumerate(state_q_values[0]):
                Q_values[tuple(s)][a] = np.round(val, 2)
        
        return Q_values
    


class DQNAgent:
    def __init__(self, environment, hidden_neurons, activation, memory_size):
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self.Q = self.make_qnn(hidden_neurons = hidden_neurons, activation = activation)
        self.Q_target = self.make_qnn(hidden_neurons = hidden_neurons, activation = activation)
        self.update_target_network()
        self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.loss_fun = tf.keras.losses.MeanSquaredError()
        self.memory_size = memory_size

        self.states = deque(maxlen = memory_size)
        self.actions = deque(maxlen = memory_size)
        self.rewards = deque(maxlen = memory_size)
        self.next_states = deque(maxlen = memory_size)
        self.dones = deque(maxlen = memory_size)


    def make_qnn(self, hidden_neurons = 10, activation = 'elu'):
        fnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (len(self.observation_space),)),
            tf.keras.layers.Dense(hidden_neurons, activation = activation),
            tf.keras.layers.Dense(self.action_space.n)
        ])
        return fnn    


    def action(self, state, epsilon):
        s_tensor = tf.convert_to_tensor(state)
        s_tensor = tf.expand_dims(s_tensor, 0)
        if np.random.rand() <= epsilon:
            a = self.action_space.sample()
        else:
            action_values = self.Q(s_tensor, training = False)
            a = tf.argmax(action_values[0]).numpy()
        return a
    

    def memorize(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)


    def update_network(self, batch_size):
        if len(self.states) >= batch_size:
            idxs = np.random.choice(min(len(self.states), self.memory_size), size = batch_size)
            states_batch = [self.states[idx] for idx in idxs]
            actions_batch = [self.actions[idx] for idx in idxs]
            rewards_batch = [self.rewards[idx] for idx in idxs]
            next_states_batch = [self.next_states[idx] for idx in idxs]
            dones_batch = [self.dones[idx] for idx in idxs]

            with tf.GradientTape() as tape:
                tape.watch(self.Q.trainable_variables)
                predictions = tf.reduce_sum(tf.multiply(self.Q(tf.convert_to_tensor(states_batch)), tf.one_hot(actions_batch, self.action_space.n)), axis = 1)
                targets = tf.convert_to_tensor(rewards_batch, dtype = tf.float32) + tf.reduce_max(self.Q_target(tf.convert_to_tensor(next_states_batch)), axis = 1) * (1 - tf.cast(dones_batch, tf.float32))
                loss = self.loss_fun(targets, predictions)
            grads = tape.gradient(loss, self.Q.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.Q.trainable_variables))
        else:
            pass


    def update_target_network(self):
        self.Q_target.set_weights(self.Q.get_weights())


    def return_qvalue_dictionary(self):
        states = [np.array([2, 1]), np.array([1, 1]), np.array([1, 2])]
        Q_values = dict()
        for s in states:
            Q_values[tuple(s)] = dict()
            s_tensor = tf.convert_to_tensor(s)
            s_tensor = tf.expand_dims(s, 0)
            state_q_values = self.Q(s_tensor).numpy()
            for a, val in enumerate(state_q_values[0]):
                Q_values[tuple(s)][a] = np.round(val, 2)
        
        return Q_values
    
class ReinforceAgent:
    
    def __init__(self, environment, hidden_neurons, activation):
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self.policy = self.make_policy(hidden_neurons, activation)
        self.state_action_probs = dict.fromkeys(environment.action2states, 0)
        for key in self.state_action_probs.keys():
            self.state_action_probs[key] = dict.fromkeys(environment.action2states[key], 0)


    def make_policy(self, hidden_neurons, activation):
        policy = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (len(self.observation_space), )),
            tf.keras.layers.Dense(hidden_neurons, activation = activation),
            tf.keras.layers.Dense(self.action_space.n, activation='softmax')
        ])
        return policy
    
    
    def action(self, state):
        s_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
        probs = self.policy(s_tensor)
        action = np.random.choice(self.action_space.n, p = np.squeeze(probs))
        return action
    
    def get_current_policy(self):
        for state in self.state_action_probs.keys():
            for i, prob in enumerate(self.policy(tf.expand_dims(tf.convert_to_tensor(np.array(state)), 0)).numpy().flatten()):
                self.state_action_probs[state][i] = prob
        return self.state_action_probs