from .utils import load_course_data, feature_analysis, plot_weights_over_time
import yfinance as yf
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from arch.univariate import ARX
from arch.univariate import GARCH
from arch.univariate import SkewStudent
import os



class CourseDataManager:
    def __init__(self, lags = 5):
        self.lags = lags
        self.run()

        '''
        This class is a data container which simplifies data access and preprocessing of course data
        
        Parameter:
        ------------

        lags: int setting the number of past days to be included as state variable features

        Returns:
        -----------

        '''

    def load_data(self):
       
        '''
        This method loads course data
        
        Parameter:
        ------------

        

        Returns:
        -----------

        Tuple: Two dataframes, the first with corporate stock price data, the second S&P 500 ETF (SPY) price data

        '''    

        self.df, self.spy_df = load_course_data('yfinance')


    def get_spy_returns_with_lags(self):
        
        '''
        This method preprocesses S&P 500 ETF (SPY) price data
        
        Parameter:
        ------------

        

        Returns:
        -----------


        '''    

        self.spy_returns = self.spy_df['Adj Close'].pct_change().dropna().to_frame('spy_return_t')
        for lag in range(1, self.lags+1):
            self.spy_returns.loc[:, f'spy_return_t_{lag}'] = self.spy_returns.spy_return_t.shift(lag)


    def calculate_company_returns(self):
        
        '''
        This method determines discrete returns for each company
        
        Parameter:
        ------------

        

        Returns:
        -----------

        '''    

        self.df_returns = pd.DataFrame(index = self.df.index.unique()[1:])
        self.tickers = self.df.ticker.unique().tolist()
        for ticker in self.tickers:
            df_tmp = self.df[self.df.ticker == ticker]
            df_ret_tmp = df_tmp['Adj Close'].pct_change().dropna().to_frame(f'{ticker}')
            self.df_returns = self.df_returns.merge(df_ret_tmp, left_index=True, right_index=True, how = 'outer')


    def add_lagged_features(self, with_spy = False):
        '''
        This method adds lagged returns and absolute returns to the data set
        
        Parameter:
        ------------


        Returns:
        -----------

        '''    

        if with_spy:
            column_names = [f'spy_return_t_{lag}' for lag in range(self.lags, 0, -1)] + [f'return_abs_t_{lag}' for lag in range(self.lags, 0, -1)] + [f'return_t_{lag}' for lag in range(self.lags, 0, -1)] + ['return_t', 'ticker']
        else:
            column_names = [f'return_abs_t_{lag}' for lag in range(self.lags, 0, -1)] + [f'return_t_{lag}' for lag in range(self.lags, 0, -1)] + ['return_t', 'ticker']
        
        self.df_returns_extended = pd.DataFrame(columns = column_names)

        for ticker in self.tickers:
            df_tmp = self.df_returns.loc[:, ticker]
            df_tmp = df_tmp.to_frame('return_t')

            # add lagged returns
            for lag in range(1, self.lags+1):
                    df_tmp.loc[:, f'return_t_{lag}'] = df_tmp.return_t.shift(lag)
            # add lagged absolute returns
            for lag in range(1, self.lags+1):
                    df_tmp.loc[:, f'return_abs_t_{lag}'] = df_tmp.return_t.abs().shift(lag)
                    
            if with_spy:
                # add lagged market returns represented by a S&P 500 ETF
                df_tmp = df_tmp.merge(self.spy_returns.drop(['spy_return_t'], axis = 1), left_index = True, right_index = True)
            df_tmp_columns = df_tmp.columns.tolist()
            df_tmp_columns.reverse()
            df_tmp = df_tmp.loc[:, df_tmp_columns]
            df_tmp.loc[:, 'ticker'] = ticker

            self.df_returns_extended = pd.concat((self.df_returns_extended, df_tmp))


    def get_available_tickers(self, train_start, test_end):
        '''
        
        This method returns all available tickers with full data availability for training and test data
        
        Parameter:
        ------------
        train_start: str in %Y-%m-%d format, e.g., 2020-01-01
        test_end: str in %Y-%m-%d format, e.g., 2020-01-01


        Returns:
        -----------

        ''' 
        tickers_with_full_data = self.df_returns.columns[self.df_returns.loc[train_start:test_end].isnull().sum() == 0].tolist()
        return tickers_with_full_data


    def run(self):
        '''
        This method runs the load_data, calculate_company_returns and add_lagged_features method. It should be executed after initalization of a class instance
        
        Parameter:
        ------------


        Returns:
        -----------

        ''' 

        self.load_data()
        # self.get_spy_returns_with_lags()
        self.calculate_company_returns()
        self.add_lagged_features()



class RecurrentWealthManager:
    def __init__(self, ticker, trainer_tickers, course_data_manager, output_activation, optimizer_lr, start_train, end_train, start_test, end_test, performance_benchmark = 'buy_and_hold', risk_benchmark = 'buy_and_hold', risk_measure = 'none', transaction_costs = 0.001, n_sims = 100, init_agent_weights = 0.1):
        
        '''
        This class initiates the wealth manager from the course. 
        
        Parameter:
        ------------

        ticker: a string identifying a company
        trainer_tickers: a list of strings identifying other companies which can be included in the training process
        course_data_manager: an instance of the CourseDataManager class
        output_activation: str, should be sigmoid for long investment only or tanh for allowing short selling
        optimizer_lr: float setting the tensorflow Adam learning rate
        start_train: str in %Y-%m-%d format, e.g., 2020-01-01
        end_train: str in %Y-%m-%d format, e.g., 2020-01-01
        start_test: str in %Y-%m-%d format, e.g., 2020-01-01
        end_est: str in %Y-%m-%d format, e.g., 2020-01-01
        performance_benchmark: str, should be buy_and_hold or none
        risk_benchmark: str, should be buy_and_hold or none
        risk_measure: str, should be none, std, lpm1 or lpm2
        transaction_costs: float, the transaction costs time the absolute change in the stock position are subtracted from the investment funds
        n_sims: int, the number of simulated time series paths which can be included in the training process
        init_agent_weights: float or any type, if you want to create reproducible results set this value to 0.1, for random initialization provide a string or any other type 

        Returns:
        -----------

        ''' 

        self.ticker = ticker
        self.trainer_tickers = trainer_tickers
        assert isinstance(course_data_manager, CourseDataManager), 'Please provide data using the CourseDataManager class'
        
        self.start_train = start_train
        self.end_train = end_train
        self.start_test = start_test
        self.end_test = end_test

        self.data = dict()
        self.data['original'] = course_data_manager.df_returns_extended[course_data_manager.df_returns_extended.ticker == self.ticker]
        self.external_training_data = dict()
        for ticker in self.trainer_tickers:
            self.data[ticker] = course_data_manager.df_returns_extended[course_data_manager.df_returns_extended.ticker == ticker].dropna()
            prepared_ticker_data = self.prepare_train_test_data(self.data[ticker])
            self.external_training_data[ticker] = dict()
            self.external_training_data[ticker]['X_train_tensor'] = prepared_ticker_data['X_train_tensor']
            self.external_training_data[ticker]['y_train_tensor'] = prepared_ticker_data['y_train_tensor']

        self.ts_returns = self.data['original'].return_t.loc[start_train:end_train]
        self.data['original'] = self.data['original'].dropna()

        self.output_activation = output_activation
        self.optimizer_lr = optimizer_lr

        assert performance_benchmark in ['buy_and_hold', 'none'], 'Please select buy_and_hold or None as a performance benchmark'
        self.performance_benchmark = performance_benchmark
        assert risk_benchmark in ['buy_and_hold', 'none'], 'Please select buy_and_hold or None as a performance benchmark'
        self.risk_benchmark = risk_benchmark
        assert risk_measure in ['none', 'std', 'lpm1', 'lpm2'], 'Please select None, std, lpm1 or lpm2 as a risk measure'
        self.risk_measure = risk_measure
        self.transaction_costs = transaction_costs

        original_data = self.prepare_train_test_data(self.data['original'])
        self.df_train, self.df_test = original_data['df_train'], original_data['df_test']
        self.y_train, self.y_test = original_data['y_train'], original_data['y_test']
        self.X_train_tensor, self.X_test_tensor = original_data['X_train_tensor'], original_data['X_test_tensor']
        self.y_train_tensor, self.y_test_tensor = original_data['y_train_tensor'], original_data['y_test_tensor']
        self.train_dates, self.test_dates = original_data['train_dates'], original_data['test_dates']

        self.init_agent_weights = init_agent_weights
        self.init_agent(self.init_agent_weights)   

        self.fit_ts_model()
        self.n_sims = n_sims
        self.lags = course_data_manager.lags
        self.generate_sim_data()


    def prepare_train_test_data(self, data):

        '''
        This function splits data according to training and test time periods. Furthermore, data is separated into features and the label (return at time t) and converted to tensors for deriving gradients
        using tensorflow
        
        Parameter:
        ------------


        Returns:
        -----------

        ''' 
        df_train, df_test = data.loc[self.start_train:self.end_train], data.loc[self.start_test:self.end_test]
        X_train, y_train = df_train.drop(['return_t', 'ticker'], axis = 1), df_train['return_t']
        X_test, y_test = df_test.drop(['return_t', 'ticker'], axis = 1), df_test['return_t']
        train_dates = [pd.to_datetime(dt) for dt in y_train.index]
        test_dates = [pd.to_datetime(dt) for dt in y_test.index]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        X_train_tensor = tf.convert_to_tensor(X_train_s.reshape(1, X_train.shape[0], X_train.shape[1]), dtype = 'float')
        X_test_tensor = tf.convert_to_tensor(X_test_s.reshape(1, X_test.shape[0], X_test.shape[1]), dtype = 'float')
        y_train_tensor = tf.convert_to_tensor(y_train.values, dtype = 'float')
        y_test_tensor = tf.convert_to_tensor(y_test.values, dtype = 'float')

        data_dict = dict()
        data_dict['df_train'] = df_train
        data_dict['df_test'] = df_test
        data_dict['y_train'] = y_train
        data_dict['y_test'] = y_test
        data_dict['X_train_tensor'] = X_train_tensor
        data_dict['X_test_tensor'] = X_test_tensor
        data_dict['y_train_tensor'] = y_train_tensor
        data_dict['y_test_tensor'] = y_test_tensor
        data_dict['train_dates'] = train_dates
        data_dict['test_dates'] = test_dates

        return data_dict

    
    def init_agent(self, init_agent_weights):


        '''
        This method initializes neural networks for predicing stock positions
        
        Parameter:
        ------------


        Returns:
        -----------

        ''' 

        if isinstance(init_agent_weights, float):
            self.agent = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = (self.X_train_tensor.shape[1], self.X_train_tensor.shape[2])),
                tf.keras.layers.SimpleRNN(1, activation = self.output_activation, return_sequences=True, kernel_initializer=tf.keras.initializers.Constant(value = init_agent_weights), recurrent_initializer=tf.keras.initializers.Constant(value = init_agent_weights)),
                tf.keras.layers.Flatten()
            ])
        else:
            self.agent = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = (self.X_train_tensor.shape[1], self.X_train_tensor.shape[2])),
                tf.keras.layers.SimpleRNN(1, activation = self.output_activation, return_sequences=True),
                tf.keras.layers.Flatten()
            ])

        self.agent_test = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.X_test_tensor.shape[1], self.X_test_tensor.shape[2])),
            tf.keras.layers.SimpleRNN(1, activation = self.output_activation, return_sequences=True),
            tf.keras.layers.Flatten()
        ])

        self.agent_test.set_weights(self.agent.get_weights())


    def get_gradients(self, learn_from_ticker = None, with_penalty = False, use_sim_data_epsilon = 0.0, l1_regularization = False, penalty = 0.001):

        '''
        This method derives the gradient for a training step
        
        Parameter:
        ------------

        with_penalty: boolean which declares if cumulative underperformance of buy and hold is added in the denominator to prevent the agent from complete avoidance of stock positions
        use_sim_data_epsilon: float in [0, 1], if higher than 0, a uniform random variable is drawn and simulated data is used for gradient derivation if the random realization is below this value


        Returns:
        -----------

        grads: list with gradient information
        loss: current value of the loss function

        ''' 

        if learn_from_ticker:
            X_train_tensor = self.external_training_data[learn_from_ticker]['X_train_tensor']
            y_train_tensor = self.external_training_data[learn_from_ticker]['y_train_tensor']
        elif (use_sim_data_epsilon > 0) & (np.random.rand() < use_sim_data_epsilon):
            sim_tensor = self.simdata[np.random.choice(list(self.simdata.keys()), 1)[0]]
            X_train_tensor = sim_tensor['X_train_tensor_sim'] 
            y_train_tensor = sim_tensor['y_train_tensor_sim'] 
        else:
            X_train_tensor = self.X_train_tensor
            y_train_tensor = self.y_train_tensor
        
        with tf.GradientTape() as tape:
            tape.watch(self.agent.trainable_variables)
            holdings = self.agent(X_train_tensor)
            holdings = tf.reshape(holdings, shape = (self.X_train_tensor.shape[1]))

            # determine the delte trading positions for calculating the trading costs
            holdings_shifted = tf.concat([holdings[1:], tf.convert_to_tensor([0], dtype = 'float')], 0)
            abs_pos_change = tf.abs(tf.subtract(holdings_shifted, holdings))

            # trading_return = (1 + h_t * r_t)*(1 - transaction_costs * position_change) - 1
            returns = (1 + holdings * y_train_tensor) * (1 - self.transaction_costs * abs_pos_change) - 1
            excess_returns = tf.subtract(returns, y_train_tensor) 

            cum_returns_agent = tf.math.reduce_sum(returns)
            cum_returns_bh = tf.math.reduce_sum(y_train_tensor)
            penalty = tf.maximum(cum_returns_bh - cum_returns_agent, 0)

            if self.performance_benchmark == 'buy_and_hold':
                numerator_returns = excess_returns 
            else:
                numerator_returns = returns 

            if self.risk_benchmark == 'buy_and_hold':
                denominator_returns = excess_returns 
            else:
                denominator_returns = returns 

            if self.risk_measure == 'none': 
                loss = -tf.math.reduce_mean(numerator_returns)
            elif self.risk_measure == 'std':
                loss = -tf.divide(tf.math.reduce_mean(numerator_returns), tf.math.reduce_std(denominator_returns) + penalty * with_penalty)
            elif self.risk_measure == 'lpm1':
                loss = -tf.divide(tf.math.reduce_mean(numerator_returns), tf.math.reduce_mean(tf.maximum(0-denominator_returns, 0)) + penalty * with_penalty)
            elif self.risk_measure == 'lpm2':
                loss = -tf.divide(tf.math.reduce_mean(numerator_returns), tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.maximum(0-denominator_returns, 0)))) + penalty * with_penalty)

            if l1_regularization:
                state_weights = np.concatenate((self.agent.get_weights()[0], self.agent.get_weights()[1]))
                loss += penalty * tf.convert_to_tensor(np.abs(state_weights).sum(), dtype = 'float')

        grads = tape.gradient(loss, self.agent.trainable_variables)
        return grads, loss.numpy().flatten()[0]
    

    def get_avg_gradients_from_all(self, with_penalty = False, penalty = 0.001):
        '''
        
        This method determines gradients for the stock and all stocks provided as trainer_tickers; the average gradients over all gradients are returned

        '''
        gradients_list = []
        grads, loss = self.get_gradients(with_penalty = with_penalty, use_sim_data_epsilon = 0.0, l1_regularization = False, penalty = penalty)
        gradients_list.append(grads)
        for ticker in self.trainer_tickers:
            external_grads, _ = self.get_gradients(learn_from_ticker=ticker, with_penalty = with_penalty, use_sim_data_epsilon = 0.0, l1_regularization = False, penalty = penalty)
            gradients_list.append(external_grads)

        avg_gradients = []
        for i in range(len(grads)):
            avg_gradients.append(tf.divide(tf.add_n([grad[i] for grad in gradients_list]), len(gradients_list)))

        return avg_gradients, loss
    

    def update_weights(self, gradients):

        '''
        This method updates the weights of the neural network using gradient information
        
        Parameter:
        ------------

        gradients: list with gradient information


        Returns:
        -----------

        '''
        

        self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))    


    def learn(self, epochs, learn_from_tickers = False, reset_weights = True, with_penalty = False, use_sim_data_epsilon = 0.0, l1_regularization = False, penalty = 0.001):
        
        '''
        This method trains the agent, repeating the gradient derivation and weights updata step
        
        Parameter:
        ------------

        epochs: int, the number of training steps
        reset_weights: boolean, if True starts with new training else continues training
        learn_from_tickers: boolean, if True average gradients from all companies are used for gradient update
        with_penalty: boolean, if penalty should be included when deriving the gradients
        use_sim_data_epsilon: float in [0, 1], if higher than 0, a uniform random variable is drawn and simulated data is used for gradient derivation if the random realization is below this value


        Returns:
        -----------

        '''

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.optimizer_lr)


        if reset_weights:
            self.init_agent(self.init_agent_weights)
        losses = []
        for e in range(epochs):
            if learn_from_tickers:
                grads, loss = self.get_avg_gradients_from_all(with_penalty = with_penalty, penalty = penalty)
            else:
                grads, loss = self.get_gradients(with_penalty = with_penalty, use_sim_data_epsilon = use_sim_data_epsilon, l1_regularization = l1_regularization, penalty = penalty)
            self.update_weights(grads)
            losses.append(loss)
            if e%20 == 0:
                print(f'{e*100/epochs:.2f}% of training is finished.')
        
        self.agent_test.set_weights(self.agent.get_weights())
        return losses
        

    def get_returns_and_holdings(self, discrete_holdings = False):
        
        '''
        This method gets holdings and returns for the current agentt
        
        Parameter:
        ------------

        discrete_holdings: boolean if set to True, the agent either is fully invested (if the predicted holding is greater than 0.50) or not


        Returns:
        -----------

        holdings_train: numpy array with stock position of training period
        holdings_test: numpy array with stock position of test period
        returns_train: numpy array with discrete returns of the agent's training period
        returns_test: numpy array with discrete returns of the agent's test period

        '''

        holdings_train = self.agent(self.X_train_tensor)
        holdings_train = tf.reshape(holdings_train, shape = (self.X_train_tensor.shape[1]))
        if discrete_holdings:
            holdings_train = tf.where(holdings_train > 0.5, 1., 0.)
        holdings_train_shifted = tf.concat([holdings_train[1:], tf.convert_to_tensor([0.], dtype = 'float')], 0)
        abs_pos_change_train = tf.abs(tf.subtract(holdings_train_shifted, holdings_train))
        returns_train = (1 + holdings_train * self.y_train.values) * (1 - self.transaction_costs * abs_pos_change_train) - 1


        holdings_test = self.agent_test(self.X_test_tensor)
        holdings_test = tf.reshape(holdings_test, shape = (self.X_test_tensor.shape[1]))
        if discrete_holdings:
            holdings_test = tf.where(holdings_test > 0.5, 1., 0.)            
        holdings_test_shifted = tf.concat([holdings_test[1:], tf.convert_to_tensor([0.], dtype = 'float')], 0)
        abs_pos_change_test = tf.abs(tf.subtract(holdings_test_shifted, holdings_test))
        returns_test = (1 + holdings_test * self.y_test.values) * (1 - self.transaction_costs * abs_pos_change_test) - 1

        return holdings_train.numpy(), holdings_test.numpy(), returns_train.numpy(), returns_test.numpy()


    def plot_trading(self, discrete_holdings = False, save_path = None, figure_name = None):

        '''
        This method gets plots trading performance of the agent
        
        Parameter:
        ------------

        discrete_holdings: boolean if set to True, the agent either is fully invested (if the predicted holding is greater than 0.50) or not


        Returns:
        -----------

        fig, axs instance from matplotlib.pylab 

        '''

        holdings_train, holdings_test, agent_returns_train, agent_returns_test = self.get_returns_and_holdings(discrete_holdings=discrete_holdings)
 
        fig, axs = plt.subplots(2, 3, figsize = (12, 6))

        axs[0, 0].scatter(self.y_train.values, agent_returns_train)
        axs[0, 0].set_xlabel('buy and hold returns (train)')
        axs[0, 0].set_ylabel('trading agent returns (train)')
        axs[0, 1].plot(self.train_dates, np.cumprod(1 + agent_returns_train), label = 'trading strategy (train)')
        axs[0, 1].plot(self.train_dates, np.cumprod(1 + self.y_train.values), label = 'buy and hold (train)')
        axs[0, 1].set_title('Wealth development (train)')
        axs[0, 1].set_xticks(axs[0, 1].get_xticks(), axs[0, 1].get_xticklabels(), rotation=45, ha='right')
        axs[0, 1].legend()
        axs[0, 2].plot(self.train_dates, holdings_train)
        axs[0, 2].set_title('Stock position (train)')
        axs[0, 2].set_xticks(axs[0, 2].get_xticks(), axs[0, 2].get_xticklabels(), rotation=45, ha='right')
        axs[1, 0].scatter(self.y_test.values, agent_returns_test)
        axs[1, 0].set_xlabel('buy and hold returns (test)')
        axs[1, 0].set_ylabel('trading agent returns (test)')
        axs[1, 1].plot(self.test_dates, np.cumprod(1 + agent_returns_test), label = 'trading strategy (test)')
        axs[1, 1].plot(self.test_dates, np.cumprod(1 + self.y_test.values), label = 'buy and hold (test)')
        axs[1, 1].set_title('Wealth development (test)')
        axs[1, 1].set_xticks(axs[1, 1].get_xticks(), axs[1, 1].get_xticklabels(), rotation=45, ha='right')
        axs[1, 1].legend()
        axs[1, 2].plot(self.test_dates, holdings_test)
        axs[1, 2].set_title('Stock position (test)')
        axs[1, 2].set_xticks(axs[1, 2].get_xticks(), axs[1, 2].get_xticklabels(), rotation=45, ha='right')
        fig.suptitle(f'{self.ticker} - training and test results ')
        fig.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, figure_name))
        return fig, axs

    def get_feature_importance(self):

        '''
        This method gets feature importance for the agent's holdings
        
        Parameter:
        ------------

        Returns:
        -----------

       sum: pandas dataframe with first and second order feature importanes
       sum_c: pandas dataframw with corresponding conformity of feature importances

        '''

        fnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.X_train_tensor.shape[2] + 1, )),
            tf.keras.layers.Dense(1, activation = self.output_activation)
        ])

        agent_weights = self.agent.get_weights()
        fnn.set_weights([np.concatenate((agent_weights[0], agent_weights[1])), agent_weights[2]])

        X_train_tensor_flat = tf.reshape(self.X_train_tensor, (self.X_train_tensor.shape[1], self.X_train_tensor.shape[2]) )
        holdings = self.agent(self.X_train_tensor)
        holdings_tf_shift = tf.reshape(tf.concat([tf.constant([0.], dtype = 'float'), holdings[0, :-1]], axis = 0), shape = (self.X_train_tensor.shape[1], 1))
        X_train_fnn = tf.concat([X_train_tensor_flat, holdings_tf_shift], axis = 1)
        sum, sum_c = feature_analysis(X_train_fnn, fnn, self.df_train.drop(['return_t', 'ticker'], axis = 1).columns.tolist() + ['pos_t_t1'])
        return sum, sum_c
    

    def get_performance(self, discrete_holdings = False):

        '''
        This method summarizes the agent's performance
        
        Parameter:
        ------------

        discrete_holdings: boolean if set to True, the agent either is fully invested (if the predicted holding is greater than 0.50) or not


        Returns:
        -----------

        performance_results: pandas dataframe

        '''

        holdings_train, holdings_test, returns_train, returns_test = self.get_returns_and_holdings(discrete_holdings=discrete_holdings)
        means = [returns_train.mean(), returns_test.mean(), self.y_train.mean(), self.y_test.mean()]
        stds = [returns_train.std(ddof = 1), returns_test.std(ddof = 1), self.y_train.std(ddof = 1), self.y_test.std(ddof = 1)]
        lpms_train = self._get_lpms(returns_train)
        lpms_train_bh = self._get_lpms(self.y_train.values)
        lpms_test = self._get_lpms(returns_train)
        lpms_test_bh = self._get_lpms(self.y_test.values)
        lpm1s = lpms_train[0], lpms_test[0], lpms_train_bh[0], lpms_test_bh[0]
        lpm2s = lpms_train[1], lpms_test[1], lpms_train_bh[1], lpms_test_bh[1]

        performance_results = pd.DataFrame(columns = ['train', 'test', 'train (bnh)', 'test (bnh)'])
        performance_results.loc['mean'] = means
        performance_results.loc['std'] = stds
        performance_results.loc['lpm1'] = lpm1s
        performance_results.loc['lpm2 (sqrt)'] = lpm2s
        performance_results.loc['mean/std'] = performance_results.loc['mean'] / performance_results.loc['std']
        performance_results.loc['mean/lmp1'] = performance_results.loc['mean'] / performance_results.loc['lpm1']
        performance_results.loc['mean/lpm2 (sqrt)'] = performance_results.loc['mean'] / performance_results.loc['lpm2 (sqrt)']

        return performance_results


    def fit_ts_model(self):

        '''
        This method estimates a AR(1)-GARCH(1, 1) model with a skewwed student t distribution for the return time series
        
        Parameter:
        ------------

        Returns:
        -----------
        
        '''

        self.ts_model = ARX(self.ts_returns.values*100, lags=[1])
        self.ts_model.volatility = GARCH()
        self.ts_model.distribution = SkewStudent()
        self.ts_parameters = self.ts_model.fit(disp = 'off')


    def generate_sim_data(self):

        '''
        This method uses the estimated AR(1)-GARCH(1, 1) model to generate simulated time series data which can additionaly be used for training the agent
        
        Parameter:
        ------------

        Returns:
        -----------
        
        '''

        self.simdata = dict()
        for sim in range(self.n_sims):
            df_sim_tmp = self.ts_model.simulate(params = self.ts_parameters.params, nobs = self.ts_returns.shape[0]).data.divide(100).to_frame('return_t')
            
            # add lagged returns
            for lag in range(1, self.lags+1):
                    df_sim_tmp.loc[:, f'return_t_{lag}'] = df_sim_tmp.return_t.shift(lag)
            # add lagged absolute returns
            for lag in range(1, self.lags+1):
                    df_sim_tmp.loc[:, f'return_abs_t_{lag}'] = df_sim_tmp.return_t.abs().shift(lag)
            colnames = df_sim_tmp.columns.tolist()
            colnames.reverse()
            df_sim_tmp_extended = df_sim_tmp.loc[:, colnames].dropna()

            X_train_sim, y_train_sim = df_sim_tmp_extended.drop(['return_t'], axis = 1), df_sim_tmp_extended['return_t']
            scaler_tmp = StandardScaler()
            X_train_sim_s = scaler_tmp.fit_transform(X_train_sim)
            X_train_sim_s = X_train_sim_s.reshape(1, self.X_train_tensor.shape[1], self.X_train_tensor.shape[2])
            X_train_sim_tensor = tf.convert_to_tensor(X_train_sim_s, dtype='float')
            y_train_sim_tensor = tf.convert_to_tensor(y_train_sim.values, dtype = 'float')

            self.simdata[sim] = dict()
            self.simdata[sim]['X_train_tensor_sim'] = X_train_sim_tensor
            self.simdata[sim]['y_train_tensor_sim'] = y_train_sim_tensor


    @staticmethod
    def _get_lpms(returns):
        lpm_scores = np.maximum(-returns, 0)
        lpm1 = np.mean(lpm_scores)
        lpm2_sq = np.sqrt(np.mean(lpm_scores**2))

        return lpm1, lpm2_sq





class DynamicPortfolioManager:
    def __init__(self, course_data_manager, pf_tickers, optimizer_lr, start_train, end_train, start_test, end_test, rolling_window = 20, transaction_costs = 0.001, non_uniform_weight_penalty = 0.01):
    
        '''
        This class intializes a numerical dynamic portfolio optimization agent

        Parameter:
        ------------
        course_data_manager: CourseDataManager instance handling course data
        pf_tickers: a list of strings with stock symbols
        optimizer_lr: float, a learning rate which is used for the Adam optimizer when making the gradient update
        start_train: str in %Y-%m-%d format, e.g., 2020-01-01
        end_train: str in %Y-%m-%d format, e.g., 2020-01-01
        start_test: str in %Y-%m-%d format, e.g., 2020-01-01
        end_est: str in %Y-%m-%d format, e.g., 2020-01-01
        rolling_window: int determining the number of past days to calculate the rolling means, standard deviations and correlations
        transaction_costs: float which is multiplied by the sum of absolute stock position changes to determine overall transaction costs
        non_uniform_weight_penalty: float which can be used to prevent the agent from extreme positions in single stocks

        Returns:
        ----------
        
        '''
    
        self.pf_tickers = pf_tickers
        self.optimizer_lr = optimizer_lr
        self.n_companies = len(self.pf_tickers)
        self.train_start = pd.to_datetime(start_train)
        self.train_end = pd.to_datetime(end_train)
        self.test_start = pd.to_datetime(start_test)
        self.test_end = pd.to_datetime(end_test)
        self.rolling_window = rolling_window
        self.transaction_costs = transaction_costs
        self.non_uniform_weight_penalty = non_uniform_weight_penalty

        self.df_returns = course_data_manager.df_returns.loc[:, self.pf_tickers]
        self.prepare_train_test_data()
        self.init_agent()


    def prepare_train_test_data(self):

        '''
        This function is used internally to prepare feature and target data. It calculates rolling means, standard deviations and correlations
        to determine feature variables. These features are standardized using mean-std standarizdation.
        
        '''

        roll_mu = self.df_returns.rolling(self.rolling_window).mean()
        roll_mu.columns = [f'mu_{ticker}' for ticker in self.pf_tickers]
        roll_std = self.df_returns.rolling(self.rolling_window).std()
        roll_std.columns = [f'sigma_{ticker}' for ticker in self.pf_tickers]
        roll_corr = self.df_returns.rolling(self.rolling_window).corr()
        cols = []
        i_idx, j_idx = np.tril_indices(n = self.n_companies, k = -1)
        for i, j in zip(i_idx, j_idx):
            cols.append(f'rho_{self.pf_tickers[i]}-{self.pf_tickers[j]}')
        roll_corr_t = pd.DataFrame(columns = cols)
        for dt in roll_corr.index.get_level_values(0).unique():
            roll_corr_t.loc[dt] = roll_corr.loc[(dt, self.pf_tickers), :].values[np.tril_indices(n = self.n_companies, k = -1)]

        self.X = pd.concat((roll_mu, roll_std, roll_corr_t), axis = 1)
        self.X = self.X.iloc[self.rolling_window-1:]
        self.X.index = pd.to_datetime(self.X.index)
        self.Y = self.df_returns.iloc[self.rolling_window-1:]
        self.Y.index = pd.to_datetime(self.Y.index)

        self.X_train = self.X.loc[self.train_start:self.train_end]
        self.X_train = self.X_train.iloc[:-1, :]
        self.Y_train = self.Y.loc[self.train_start:self.train_end]
        self.Y_train = self.Y_train.iloc[1:, :]

        self.X_test = self.X.loc[self.test_start:self.test_end]
        self.X_test = self.X_test.iloc[:-1, :]
        self.Y_test = self.Y.loc[self.test_start:self.test_end]
        self.Y_test = self.Y_test.iloc[1:, :]

        
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train_s, self.X_test_s = self.scaler.transform(self.X_train), self.scaler.transform(self.X_test)

        self.X_train_tensor = tf.reshape(tf.convert_to_tensor(self.X_train_s, dtype = 'float'), shape = (1, self.X_train.shape[0], self.X_train.shape[1]))
        self.X_test_tensor = tf.reshape(tf.convert_to_tensor(self.X_test_s, dtype = 'float'), shape = (1, self.X_test.shape[0], self.X_test.shape[1]))
        self.Y_train_tensor = tf.convert_to_tensor(self.Y_train, dtype = 'float')
        self.Y_test_tensor = tf.convert_to_tensor(self.Y_test, dtype = 'float')


    def init_agent(self):
        '''
        This function initializes weights for the agent.

        '''
        self.agent = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.X_train.shape[0], self.X_train.shape[1])),
            tf.keras.layers.SimpleRNN(self.n_companies, activation = 'softmax', return_sequences=True, kernel_initializer=tf.keras.initializers.Constant(value = 0.1), recurrent_initializer=tf.keras.initializers.Constant(value = 0.1))
        ])

        self.agent_test = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.X_test.shape[0], self.X_test.shape[1])),
            tf.keras.layers.SimpleRNN(self.n_companies, activation = 'softmax', return_sequences=True)
        ])

        self.agent_test.set_weights(self.agent.get_weights())


    def get_gradients(self, mean_only = False):
        ''''
        This function determines the gradient for the average mean to standard deviation loss w.r.t. the agent's parameters.
        Furthermore, it adds a penalty in the denominator which is higher, the more extreme single stock positions are and the higher
        the non_uniform_weight_penalty variable is set.
        
        '''
        naive_holdings = tf.convert_to_tensor(np.array(([1 / self.n_companies] * self.X_train.shape[0] * self.n_companies)).reshape(self.X_train.shape[0] , self.n_companies), dtype = 'float')
        with tf.GradientTape() as tape:
            tape.watch(self.agent.trainable_variables)
            holdings = self.agent(self.X_train_tensor)
            holdings = tf.reshape(holdings, shape = (self.X_train.shape[0], -1))
            pf_returns = tf.reduce_sum(tf.multiply(holdings, self.Y_train_tensor), axis = 1)
            holdings_shifted = tf.concat([holdings[1:], tf.expand_dims(tf.zeros(self.n_companies), 0)], axis = 0)
            abs_pos_change = tf.reduce_sum(tf.abs(tf.subtract(holdings_shifted, holdings)), axis = 1)
            delta_to_naive = tf.reduce_mean(tf.abs(tf.subtract(holdings, naive_holdings)))

            returns = (1 + pf_returns) * (1 - abs_pos_change * self.transaction_costs) - 1
            if mean_only:
                loss = -tf.reduce_mean(returns) + self.non_uniform_weight_penalty * delta_to_naive
            else:
                loss = -tf.divide(tf.reduce_mean(returns), tf.math.reduce_std(returns) + self.non_uniform_weight_penalty * delta_to_naive) 
        
        grads = tape.gradient(loss, self.agent.trainable_variables)
        return grads, loss.numpy().flatten()[0]


    def update_weights(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))    


    def learn(self, epochs, reset_weights = True, mean_only = False):

        ''''
        
        This function iteratively determines the training loss, the gradients and uses the Adam optimizer to update parameters based on gradient information.

        '''

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.optimizer_lr)
        if reset_weights:
            self.init_agent()
        losses = []
        for e in range(epochs):
            grads, loss = self.get_gradients(mean_only = mean_only)
            self.update_weights(grads)
            losses.append(loss)
            if e%20 == 0:
                print(f'{e*100/epochs:.2f}% of training is finished.')
        
        self.agent_test.set_weights(self.agent.get_weights())
        return losses
    

    def get_holdings_and_returns(self):
        '''
        
        Get the holdings during time for every stock.


        '''
        holdings_train = self.agent(self.X_train_tensor).numpy()
        holdings_train = holdings_train.reshape(self.X_train.shape[0], self.n_companies)
        holdings_shifted_train = np.concatenate([holdings_train[1:], np.zeros(self.n_companies).reshape(1, -1)], axis = 0)
        abs_pos_change_train = np.sum(np.abs(holdings_shifted_train - holdings_train), axis = 1)
        pf_returns_train = (holdings_train * self.Y_train).sum(axis = 1)
        returns_train = (1 + pf_returns_train) * (1 - abs_pos_change_train * self.transaction_costs) - 1

        holdings_test = self.agent_test(self.X_test_tensor).numpy()
        holdings_test = holdings_test.reshape(self.X_test.shape[0], self.n_companies)
        holdings_shifted_test = np.concatenate([holdings_test[1:], np.zeros(self.n_companies).reshape(1, -1)], axis = 0)
        abs_pos_change_test = np.sum(np.abs(holdings_shifted_test - holdings_test), axis = 1)
        pf_returns_test = (holdings_test * self.Y_test).sum(axis = 1)
        returns_test = (1 + pf_returns_test) * (1 - abs_pos_change_test * self.transaction_costs) - 1

        return holdings_train, holdings_test, returns_train, returns_test


    def get_performance(self):

        '''
        
        Get performance for training and test data.
        
        '''

        holdings_train, holdings_test, returns_train, returns_test = self.get_holdings_and_returns()
        
        naive_returns_train = self.Y_train.mean(axis = 1)
        naive_returns_test = self.Y_test.mean(axis = 1)

        all_names = ['pf', 'naive'] + self.pf_tickers

        cols = []
        for mode in ['train', 'test']:
            for name in all_names:
                cols.append(mode + '-' + name)

        train_means = [returns_train.mean(), naive_returns_train.mean()] + self.Y_train.mean().tolist()
        test_means = [returns_test.mean(), naive_returns_test.mean()] + self.Y_test.mean().tolist()
        means = train_means + test_means

        train_stds = [returns_train.std(ddof = 1), naive_returns_train.std(ddof = 1)] + self.Y_train.std(ddof = 1).tolist()
        test_stds = [returns_test.std(ddof = 1), naive_returns_test.std(ddof = 1)] + self.Y_test.std(ddof = 1).tolist()
        stds = train_stds + test_stds

        performance_results = pd.DataFrame(columns = cols)
        performance_results.loc['mean'] = means
        performance_results.loc['std'] = stds
        performance_results.loc['mean/std'] = performance_results.loc['mean'] / performance_results.loc['std']
        return performance_results


    def plot_performance(self):
        '''
        
        Visualize performance for training and test data.
        
        '''
        holdings_train, holdings_test, returns_train, returns_test = self.get_holdings_and_returns()
        naive_returns_train = self.Y_train.mean(axis = 1)
        naive_returns_test = self.Y_test.mean(axis = 1)

        pf_all_train = pd.concat((self.Y_train.add(1.).cumprod(), returns_train.to_frame('portfolio').add(1.).cumprod(), naive_returns_train.add(1.).cumprod().to_frame('naive')), axis = 1)
        pf_all_test = pd.concat((self.Y_test.add(1.).cumprod(), returns_test.to_frame('portfolio').add(1.).cumprod(), naive_returns_test.add(1.).cumprod().to_frame('naive')), axis = 1)

        holdings_train_df = pd.DataFrame(data = holdings_train, columns = self.pf_tickers, index = self.X_train.index)
        holdings_test_df = pd.DataFrame(data = holdings_test, columns = self.pf_tickers, index = self.X_test.index)


        fig, axs = plt.subplots(2, 2, figsize = (12, 8))
        plot_weights_over_time(holdings_train_df, legend = True, ax = axs[0, 0])
        #pd.DataFrame(holdings_train, columns = self.pf_tickers, index = self.X_train.index).plot(ax = axs[0, 0], title = 'Potfolio holdings')
        pf_all_train.plot(ax = axs[0, 1], title = 'Wealth development')
        plot_weights_over_time(holdings_test_df, legend = False, ax = axs[1, 0])
        #pd.DataFrame(holdings_test, columns = self.pf_tickers, index = self.X_test.index).plot(ax = axs[1, 0], title = 'Portfolio holdings')
        pf_all_test.plot(ax = axs[1, 1], title = 'Wealth development', legend = False)
        fig.suptitle(f'{self.pf_tickers} - training and test results ')
        fig.tight_layout()


    def plot_features(self):
        '''
        
        Visualize the non-standardized features over time.

        '''
        fig, axs = plt.subplots(2, 3, figsize = (14, 6))
        self.X_train.iloc[:, :self.n_companies].plot(ax = axs[0, 0], title = 'rolling means (train)')
        self.X_train.iloc[:, self.n_companies:2*self.n_companies].plot(ax = axs[0, 1], title = 'rolling stds (train)')
        self.X_train.iloc[:, 2*self.n_companies:].plot(ax = axs[0, 2], title = 'rolling corr. (train)', cmap = 'Set2')

        self.X_test.iloc[:, :self.n_companies].plot(ax = axs[1, 0], title = 'rolling means (test)', legend = False)
        self.X_test.iloc[:, self.n_companies:2*self.n_companies].plot(ax = axs[1, 1], title = 'rolling stds (test)', legend = False)
        self.X_test.iloc[:, 2*self.n_companies:].plot(ax = axs[1, 2], title = 'rolling corr. (test)', cmap = 'Set2', legend = False)
        fig.tight_layout()
        return fig, axs
    

    

#########################
# old
#########################


class WealthManager:
    def __init__(self, ticker, course_data_manager, agent_args, start_train, end_train, start_test, end_test, performance_benchmark = 'buy_and_hold', risk_benchmark = 'buy_and_hold', risk_measure = 'none', reward_scaler = 100, **kwargs):
        
        assert isinstance(course_data_manager, CourseDataManager), 'Please provide data using the CourseDataManager class'
        self.ticker = ticker
        self.data = course_data_manager.df_returns_extended[course_data_manager.df_returns_extended.ticker == self.ticker]
        
        self.agent_args = agent_args
        self.optimizer = self.agent_args['optimizer']
        
        self.start_train = start_train
        self.end_train = end_train
        self.start_test = start_test
        self.end_test = end_test

        assert performance_benchmark in ['buy_and_hold', 'none'], 'Please select buy_and_hold or None as a performance benchmark'
        self.performance_benchmark = performance_benchmark
        assert risk_benchmark in ['buy_and_hold', 'none'], 'Please select buy_and_hold or None as a performance benchmark'
        self.risk_benchmark = risk_benchmark
        assert risk_measure in ['none', 'std', 'lpm1', 'lpm2'], 'Please select None, std, lpm1 or lpm2 as a risk measure'
        self.risk_measure = risk_measure
        self.reward_scaler = reward_scaler

        self.prepare_train_test_data()
        self.init_agent()   

    def prepare_train_test_data(self):
        self.df_train, self.df_test = self.data.loc[self.start_train:self.end_train], self.data.loc[self.start_test:self.end_test]
        self.X_train, self.y_train = self.df_train.drop(['return_t', 'ticker'], axis = 1), self.df_train['return_t']
        self.X_test, self.y_test = self.df_test.drop(['return_t', 'ticker'], axis = 1), self.df_test['return_t']
        self.train_dates = [pd.to_datetime(dt) for dt in self.y_train.index]
        self.test_dates = [pd.to_datetime(dt) for dt in self.y_test.index]

        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train_s = self.scaler.transform(self.X_train)
        self.X_test_s = self.scaler.transform(self.X_test)

        self.X_train_tensor = tf.convert_to_tensor(self.X_train_s, dtype = 'float')
        self.X_test_tensor = tf.convert_to_tensor(self.X_test_s, dtype = 'float')
        self.y_train_tensor = tf.expand_dims(tf.convert_to_tensor(self.y_train.values, dtype = 'float'), 1)
        self.y_test_tensor = tf.expand_dims(tf.convert_to_tensor(self.y_test.values, dtype = 'float'), 1)


    def init_agent(self):
        self.agent = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dense(self.agent_args['hidden_dim'], activation = self.agent_args['hidden_activation']),
            tf.keras.layers.Dense(1, activation = self.agent_args['output_activation'])
        ])


    def get_gradients(self, with_penalty = False):
        with tf.GradientTape() as tape:
            tape.watch(self.agent.trainable_variables)
            holdings = self.agent(self.X_train_tensor)
            returns = tf.multiply(holdings, self.y_train_tensor)
            excess_returns = tf.subtract(returns, self.y_train_tensor) 

            cum_returns_agent = tf.math.reduce_sum(returns)
            cum_returns_bh = tf.math.reduce_sum(self.y_train_tensor)
            penalty = tf.maximum(cum_returns_bh - cum_returns_agent, 0)
            
            if self.performance_benchmark == 'buy_and_hold':
                numerator_returns = excess_returns 
            else:
                numerator_returns = returns 

            if self.risk_benchmark == 'buy_and_hold':
                denominator_returns = excess_returns 
            else:
                denominator_returns = returns 

            if self.risk_measure == 'none': 
                loss = -tf.math.reduce_mean(numerator_returns)
            elif self.risk_measure == 'std':
                loss = -tf.divide(tf.math.reduce_mean(numerator_returns), tf.math.reduce_std(denominator_returns) + penalty * with_penalty)
            elif self.risk_measure == 'lpm1':
                loss = -tf.divide(tf.math.reduce_mean(numerator_returns), tf.math.reduce_mean(tf.maximum(0-denominator_returns, 0)) + penalty * with_penalty)
            elif self.risk_measure == 'lpm2':
                loss = -tf.divide(tf.math.reduce_mean(numerator_returns), tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.maximum(0-denominator_returns, 0)))) + penalty * with_penalty)

        grads = tape.gradient(loss, self.agent.trainable_variables)
        return grads, loss.numpy().flatten()[0]
    

    def update_weights(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))    


    def learn(self, epochs, reset_weights = True):
        if reset_weights:
            self.init_agent()
        losses = []
        for _ in range(epochs):
            grads, loss = self.get_gradients()
            self.update_weights(grads)
            losses.append(loss)
        plt.plot(losses)


    def get_returns(self):
        holdings_train = self.agent(self.X_train_tensor).numpy().flatten()
        holdings_test = self.agent(self.X_test_tensor).numpy().flatten()
        returns_train = holdings_train * self.y_train.values
        returns_test = holdings_test * self.y_test.values

        return returns_train, returns_test
    

    def plot_trading(self):

        agent_returns_train, agent_returns_test = self.get_returns()
        holdings_train = self.agent(self.X_train_tensor).numpy().flatten()
        holdings_test = self.agent(self.X_test_tensor).numpy().flatten()

        fig, axs = plt.subplots(2, 3, figsize = (14, 8))

        axs[0, 0].scatter(self.y_train.values, agent_returns_train)
        axs[0, 0].set_xlabel('buy and hold returns (train)')
        axs[0, 0].set_ylabel('trading agent returns (train)')
        axs[0, 1].plot(self.train_dates, np.cumprod(1 + agent_returns_train), label = 'trading strategy (train)')
        axs[0, 1].plot(self.train_dates, np.cumprod(1 + self.y_train.values), label = 'buy and hold (train)')
        axs[0, 1].set_title('Wealth development (train)')
        axs[0, 1].set_xticks(axs[0, 1].get_xticks(), axs[0, 1].get_xticklabels(), rotation=45, ha='right')
        axs[0, 1].legend()
        axs[0, 2].plot(self.train_dates, holdings_train)
        axs[0, 2].set_title('Stock position (train)')
        axs[0, 2].set_xticks(axs[0, 2].get_xticks(), axs[0, 2].get_xticklabels(), rotation=45, ha='right')
        axs[1, 0].scatter(self.y_test.values, agent_returns_test)
        axs[1, 0].set_xlabel('buy and hold returns (test)')
        axs[1, 0].set_ylabel('trading agent returns (test)')
        axs[1, 1].plot(self.test_dates, np.cumprod(1 + agent_returns_test), label = 'trading strategy (test)')
        axs[1, 1].plot(self.test_dates, np.cumprod(1 + self.y_test.values), label = 'buy and hold (test)')
        axs[1, 1].set_title('Wealth development (test)')
        axs[1, 1].set_xticks(axs[1, 1].get_xticks(), axs[1, 1].get_xticklabels(), rotation=45, ha='right')
        axs[1, 1].legend()
        axs[1, 2].plot(self.test_dates, holdings_test)
        axs[1, 2].set_title('Stock position (test)')
        axs[1, 2].set_xticks(axs[1, 2].get_xticks(), axs[1, 2].get_xticklabels(), rotation=45, ha='right')
        fig.suptitle(f'{self.single_data_manager.ticker} - training and test results ')
        fig.tight_layout()


    def get_sharpe_ratios(self):
        agent_returns_train, agent_returns_test = self.get_returns()
        train_sr = agent_returns_train.mean() / agent_returns_train.std()
        test_sr = agent_returns_test.mean() / agent_returns_test.std()

        bh_train_sr = self.y_train.mean() / self.y_train.std()
        bh_test_sr = self.y_test.mean() / self.y_test.std()

        return train_sr, test_sr, bh_train_sr, bh_test_sr
    

    def get_feature_importance(self):
        feature_importance, conformity = feature_analysis(self.X_train_tensor, self.agent, self.X_train.columns.tolist())
        return feature_importance, conformity

