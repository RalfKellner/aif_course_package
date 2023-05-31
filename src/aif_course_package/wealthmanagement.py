import pandas as pd
import numpy as np
from scipy.stats import mode
from .utils import load_course_data
#from aif_course_package.wealthmanagement import feature_analysis
import yfinance as yf
import tensorflow as tf
import matplotlib.pylab as plt
import pandas as pd


class WealthManager:
    def __init__(self, ticker, start_train, end_train, start_test, end_test, lags = 5, transaction_costs = 0.001, reward_scaler = 10):
        self.ticker = ticker
        self.start_train = start_train
        self.end_train = end_train
        self.start_test = start_test
        self.end_test = end_test
        self.lags = lags
        self.transaction_costs = transaction_costs
        self.reward_scaler = reward_scaler

        self.full_data = load_course_data('yfinance')
        self.available_companies = self.get_available_companies()
        self.init_data()
        self.init_agents()


    def get_available_companies(self):
        return self.full_data.ticker.unique().tolist()
    

    def prepare_data(self, log_return_series):
        m, s = self.scaler_values
        X = log_return_series.to_frame('logr_t')
        for lag in range(1, self.lags+1):  
            X.loc[:, f'logr_t_{lag}'] = X['logr_t'].shift(lag)
        colnames = X.columns.tolist()
        colnames.reverse()
        X = X.loc[:, colnames]
        X.dropna(inplace = True)
        y = X['logr_t'].values
        X.drop(['logr_t'], axis = 1, inplace = True)
        X = (X - m) / s
        return X, y
    

    def init_data(self):
        if not(self.ticker in self.available_companies):
            print('The desired company is not in the internal data set...trying to get data from yahoo finance...')
            self.single_stock_data = yf.download(self.ticker, start = self.start_train)
            self.single_stock_data.index = [dt.strftime('%Y-%m-%d') for dt in self.single_stock_data.index]
            self.single_stock_data.loc[:, 'ticker'] = self.ticker
        else:
            self.single_stock_data = self.full_data[self.full_data.ticker == self.ticker]

        self.df_stock_train = self.single_stock_data.loc[self.start_train:self.end_train]
        self.df_stock_test = self.single_stock_data.loc[self.start_test:self.end_test]
        self.log_returns_train = self.df_stock_train.Close.apply(np.log).diff().dropna()
        self.log_returns_test = self.df_stock_test.Close.apply(np.log).diff().dropna()
        self.scaler_values = self.log_returns_train.mean(), self.log_returns_train.std()
        self.X_train, self.y_train = self.prepare_data(self.log_returns_train)
        self.X_test, self.y_test = self.prepare_data(self.log_returns_test)
        self.X_train_rnn = self.X_train.values.reshape(1, self.X_train.shape[0], self.X_train.shape[1])
        self.X_test_rnn = self.X_test.values.reshape(1, self.X_test.shape[0], self.X_test.shape[1])


    def init_agents(self):
        self.agent_train = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.X_train.shape[0], self.X_train.shape[1])),
            tf.keras.layers.SimpleRNN(1, activation='sigmoid', return_sequences=True),
            tf.keras.layers.Flatten()
        ])

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.025)

        self.agent_test = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.X_test.shape[0], self.X_test.shape[1])),
            tf.keras.layers.SimpleRNN(1, activation='sigmoid', return_sequences=True),
            tf.keras.layers.Flatten()
        ])


    def _update_test_agent(self):
        self.agent_test.set_weights(self.agent_train.get_weights())


    def train_agent(self, epochs = 200, performance_evaluation = 'pseudo_sharpe_ratio', plot_training = False, compete_to_buy_and_hold = True, lpm_threshold = 0):
        losses = []
        for e in range(epochs):
            balance = 1
            with tf.GradientTape() as tape:
                tape.watch(self.agent_train.trainable_variables)
                holdings = self.agent_train(self.X_train_rnn)
                holdings = tf.reshape(holdings, self.X_train.shape[0])
                holdings_np = holdings.numpy().copy()
                holdings_np_shift = np.insert(holdings_np, 0, 0, 0)
                pos_change = holdings_np - holdings_np_shift[:-1] 

                rewards = []
                for i, (holding, log_return) in enumerate(zip(holdings, self.y_train)):
                    new_balance = holding * (1 - tf.math.abs(pos_change[i]) * self.transaction_costs) * balance * np.exp(log_return) + (1 - holding) * balance
                    trading_return = tf.math.log(tf.divide(new_balance, balance))
                    if compete_to_buy_and_hold:
                        rewards.append(trading_return - log_return)  
                    else:
                        rewards.append(trading_return)
                    balance = new_balance

                tf_rewards = tf.convert_to_tensor(rewards) * self.reward_scaler
                if performance_evaluation == 'pseudo_sharpe_ratio':
                    neg_performance_ratio =  -tf.divide(tf.math.reduce_mean(tf_rewards), tf.math.reduce_std(tf_rewards))  
                elif performance_evaluation == 'lpm0':
                    neg_performance_ratio = -tf.divide(tf.math.reduce_mean(tf_rewards), tf.math.reduce_mean(tf.maximum(lpm_threshold-tf_rewards, 0))) 

                losses.append(neg_performance_ratio)

            grads = tape.gradient(neg_performance_ratio, self.agent_train.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.agent_train.trainable_variables))

            if e % 20 == 0:
                print(f'The performance metric after {e+1} epochs is:')
                print(-neg_performance_ratio.numpy())
                print('-'*75)
                if plot_training:
                    fig, axs = self.plot_model_performance(mode = 'training')
                    plt.show()

        print('Loss evolution during training...')
        plt.plot(losses)

        # update the test agent with the same parameters
        self._update_test_agent()


    def evaluate_model(self, mode = 'training'):
        if mode == 'training':
            X = self.X_train_rnn
            y = self.log_returns_train.iloc[self.lags:].values 
            model = self.agent_train
        elif mode == 'test':
            X = self.X_test_rnn
            y = self.log_returns_test.iloc[self.lags:].values
            model = self.agent_test

        holdings = model(X)
        holdings = tf.reshape(holdings, X.shape[1])
        holdings_np = holdings.numpy().copy()
        holdings_np_shift = np.insert(holdings_np, 0, 0, 0)
        pos_change = holdings_np - holdings_np_shift[:-1]
                
        trading_returns = []
        balance = 1
        for i, (holding, log_return) in enumerate(zip(holdings_np, y)):
            new_balance = holding * (1 - np.abs(pos_change[i]) * self.transaction_costs) * balance * np.exp(log_return) + (1 - holding) * balance
            trading_return = np.log(new_balance) - np.log(balance)
            trading_returns.append(trading_return)
            balance = new_balance

        return holdings_np, trading_returns


    def plot_model_performance(self, mode = 'training'):
        
        if mode == 'training':
            date_index = pd.to_datetime(self.X_train.index)
            holdings, trading_returns = self.evaluate_model(mode = 'training')
            buy_and_hold_returns = self.y_train
        elif mode == 'test':               
            date_index = pd.to_datetime(self.X_test.index)
            holdings, trading_returns = self.evaluate_model(mode = 'test')
            buy_and_hold_returns = self.y_test

        fig, axs = plt.subplots(1, 3, figsize = (14, 5))
        axs[0].scatter(buy_and_hold_returns, trading_returns)
        axs[0].set_xlabel('log-returns')
        axs[0].set_ylabel('trading log-returns')
        axs[1].plot(date_index, np.cumsum(trading_returns), label = 'trading strategy')
        axs[1].plot(date_index, np.cumsum(buy_and_hold_returns), label = 'buy and hold')
        axs[1].set_title('Cumulative log-returns')
        axs[1].set_xticks(axs[1].get_xticks(), axs[1].get_xticklabels(), rotation=45, ha='right')
        axs[1].legend()
        axs[2].plot(date_index, holdings)
        axs[2].set_title('holdings')
        axs[2].set_xticks(axs[2].get_xticks(), axs[2].get_xticklabels(), rotation=45, ha='right')

        return fig, axs


    def xai_analysis(self, mode = 'training'):
        fnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.lags + 1, )),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        agent_weights = self.agent_train.get_weights()
        fnn.set_weights([np.concatenate((agent_weights[0], agent_weights[1])), agent_weights[2]])

        if mode == 'training':
            holdings = self.agent_train(self.X_train_rnn)
            holdings = tf.reshape(holdings, self.X_train_rnn.shape[1])
            holdings_np = holdings.numpy().copy()
            holdings_np_shift = np.insert(holdings_np, 0, 0, 0)
            holdings_tf_shift = tf.reshape(tf.convert_to_tensor(holdings_np_shift[:-1]), (len(holdings), 1))
            X_train_fnn = tf.concat([tf.convert_to_tensor(self.X_train, dtype = 'float'), holdings_tf_shift], axis = 1)
            sum, sum_c = feature_analysis(X_train_fnn, fnn, self.X_train.columns.tolist() + ['pos_t_t1'])
        elif mode == 'test':
            holdings = self.agent_test(self.X_test_rnn)
            holdings = tf.reshape(holdings, self.X_test_rnn.shape[1])
            holdings_np = holdings.numpy().copy()
            holdings_np_shift = np.insert(holdings_np, 0, 0, 0)
            holdings_tf_shift = tf.reshape(tf.convert_to_tensor(holdings_np_shift[:-1]), (len(holdings), 1))
            X_test_fnn = tf.concat([tf.convert_to_tensor(self.X_test, dtype = 'float'), holdings_tf_shift], axis = 1)
            sum, sum_c = feature_analysis(X_test_fnn, fnn, self.X_test.columns.tolist() + ['pos_t_t1'])

        return sum, sum_c




# a function for feature analysis
def feature_analysis(X, nn, feature_names):
    # determine first and second partial derivatives
    with tf.GradientTape() as snd:
        snd.watch(X)
        with tf.GradientTape() as fst:
            fst.watch(X)
            # prediction with the neural network, i.e., f(X)
            pred = nn(X)
        # gradient
        g = fst.gradient(pred, X)
    # jacobian which outputs Hessian matrix
    h = snd.batch_jacobian(g, X)

    # first partial derivatives
    g_np = g.numpy()
    # average squard partial derivatives
    g_mag_sq = (g_np**2).mean(axis = 0)
    # square root of average squard partial derivatives
    g_mag = np.sqrt(g_mag_sq)
    # sign of average partial derivatives
    g_dir = np.sign(g_np.mean(axis = 0))

    # normalizing constant
    C_ = np.sum(g_mag)
    # normalized feature importance with sign
    fi = (g_mag * g_dir) / C_

    # get signs of each sample
    fi_signs = np.sign(g_np)
    # the mode is the sign which can be observed most often among all samples, the counts is how often this sign is observed
    fi_modes, fi_counts = mode(fi_signs)
    # dividing the count of the sign which is observed most often by the overall sample size gives us a frequency measure
    # which is closer to one, the higher the conformity of the sign
    fi_conformity = fi_counts / g_np.shape[0] #fi_modes * 

    # in analogy to the calculation above, we do the same thing with the second partial derivatives
    h_np = h.numpy()
    # get the square root of average squared direction of curvature and interactions
    h_mag_sq = (h_np**2).mean(axis = 0)
    h_mag = np.sqrt(h_mag_sq)

    # the the sign of average curvature and interactions
    h_dir = np.sign(h_np.mean(axis = 0))

    # normalize the values on the diagonal line to compare the degree of non-linearity
    C_nonlin = np.sum(h_mag.diagonal())
    nonlinearity = (h_dir.diagonal() * h_mag.diagonal()) / C_nonlin

    # normlize the interactions
    lti = np.tril_indices(h_mag.shape[0], k = -1)
    C_ia = np.sum(h_mag[lti])
    interactions = (h_mag[lti] * h_dir[lti]) / C_ia

    # bring curvature and interaction effects back to matrix format
    snd_degree_summary = np.diag(nonlinearity)
    a, b = lti
    inter_iter = iter(interactions)
    for i, j in zip(a, b):
        snd_degree_summary[i, j] = next(inter_iter)
        snd_degree_summary[j, i] = snd_degree_summary[i, j]

    # get the conformity of second order effects
    snd_signs = np.sign(h_np)
    snd_degree_modes, snd_degreee_counts = mode(snd_signs)
    snd_degree_conformity = snd_degreee_counts / h_np.shape[0] #snd_degree_modes * 

    # finally summarize feature importances and second order effects
    summary = pd.DataFrame(data = snd_degree_summary, index = feature_names, columns = feature_names)
    summary.loc[:, 'feature_importance'] = fi
    # as well as their conformity
    summary_conformity = pd.DataFrame(data = snd_degree_conformity.reshape(h_np.shape[1], h_np.shape[2]), index = feature_names, columns = feature_names)
    summary_conformity.loc[:, 'feature_conformity'] = fi_conformity.flatten()

    return summary, summary_conformity
