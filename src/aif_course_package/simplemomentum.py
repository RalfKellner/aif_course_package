import gym
import pandas as pd
import numpy as np
import random
import os
import pickle
from .utils import read_pickle

class SimpleMomentumEnv(gym.Env):

    def __init__(
            self,
            start_date,
            end_date,
            momentum_window,
            initial_balance,
            tickers = None,
            n_companies = None,
            transaction_costs_pct = 0.001,
            compare_to_bh = True,
            start_with_stock_pos = False,
            use_binary_momentum_only = True
        ):

        if tickers and n_companies:
            print('Please either provide a list of tickers or the number of companies to be selected randomly. If both are provided, the ticker list is prioritized.')
        if not(tickers) and not(n_companies):
            print('All available companies are going to be used. If you desire to use less companies provide n_companies or tickers.')

        self.start_date = start_date
        self.end_date = end_date
        self.momentum_window = momentum_window
        self.initial_balance = initial_balance
        self.transaction_costs_pct = transaction_costs_pct
        self.compare_to_bh = compare_to_bh
        self.start_with_stock_pos = start_with_stock_pos
        self.use_binary_momentum_only = use_binary_momentum_only
        self.state_to_meaning = {(0,0): ['neg. momentum', 'cash pos.'], (0,1): ['neg. momentum', 'stock pos.'], (1,0): ['pos. momentum', 'cash pos.'], (1,1): ['pos. momentum', 'stock pos.']}
        self.action_to_meaning = {0: 'hold pos.', 1: 'change pos.'}
        
        self.stocks_data = self._load_data()
        self.stocks_data = self.stocks_data.reset_index().rename(columns = {'index': 'date'})
        self.stocks_data_in_time = self.stocks_data[(self.stocks_data.date > self.start_date) & (self.stocks_data.date < self.end_date)]
        self.all_tickers = self.stocks_data_in_time.ticker.unique().tolist()
        if tickers:
            self.tickers = [ticker for ticker in tickers if ticker in self.all_tickers]
            not_in_stocks_data = [ticker for ticker in tickers if ticker not in self.all_tickers]
            if len(not_in_stocks_data) > 0:
                print(f'The following tickers are not in the stocks data and will not be used: {not_in_stocks_data}')
        elif n_companies:
            self.tickers = random.sample(self.all_tickers, k=n_companies)
        else:
            self.tickers = self.all_tickers
        self.current_ticker = 0

        self.action_space = gym.spaces.Discrete(2)
        if self.use_binary_momentum_only:
            self.observation_space = gym.spaces.MultiBinary(2)
        else:
            self.observation_space = gym.spaces.Box(low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0]), high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0]))


    def reset(self):
        self.data = self.stocks_data[(self.stocks_data.ticker==self.tickers[self.current_ticker]) & (self.stocks_data.date > self.start_date) & (self.stocks_data.date < self.end_date)]
        print(f'Episode runs with ticker {self.tickers[self.current_ticker]}')
        if self.current_ticker == len(self.tickers) - 1:
            self.current_ticker = 0
        else:
            self.current_ticker += 1
        self._prepare_features()
        self.data.reset_index(inplace = True, drop = True)
        self.feature_data.reset_index(inplace = True, drop = True)
        self.current_step = 0
        self.shares_buy_and_hold = self.initial_balance / self.data['Close'][0]
        if self.start_with_stock_pos:
            self.shares_held = self.shares_buy_and_hold
            self.balance = 0
        else:
            self.balance = self.initial_balance
            self.shares_held = 0
        return self._next_observation()
        

    def step(self, action):        
        
        previous_value = self.balance + (self.shares_held * self.data['Close'][self.current_step]) 
        starting_balance = self.balance
        starting_shares = self.shares_held

        self.current_step += 1

        if action == 1:
            if self.shares_held == 0:
                self._buy_stock()
            else:
                self._sell_stock()
        else:
            pass

        current_value = self.balance + self.shares_held * self.data['Close'][self.current_step]
        reward = np.log(current_value) - np.log(previous_value) 
        reward_buy_and_hold = np.log(self.shares_buy_and_hold * self.data['Close'][self.current_step]) - np.log(self.shares_buy_and_hold * self.data['Close'][self.current_step-1])
        if self.compare_to_bh:
            agent_reward = reward - reward_buy_and_hold
        else:
            agent_reward = reward
        done = self.current_step == len(self.data) - 1
        obs = self._next_observation()
        additional_info = pd.concat((self.data.iloc[self.current_step-1].to_frame('t'), self.data.iloc[self.current_step].to_frame('t+1')), axis = 1)
        additional_info.loc['shares_held'] = [starting_shares, self.shares_held]
        additional_info.loc['balance'] = [starting_balance, self.balance]
        info = {}
        info['action'] = action
        info['step_data'] = additional_info
        return obs, agent_reward, done, info


    def render(self, mode = 'human'):
        print(f'Current step: {self.current_step}, current value: {self.balance + self.shares_held*self.data["Close"][self.current_step]}')            
    

    def play_with_ticker(self, ticker):
        assert ticker in self.tickers, 'Ticker is not in the data set, use the print_available_tickers method to get a list of available tickers.'
        self.data = self.stocks_data[(self.stocks_data.ticker==ticker) & (self.stocks_data.date > self.start_date) & (self.stocks_data.date < self.end_date)]
        print(f'Episode runs with ticker {ticker}')
        self._prepare_features()
        self.data.reset_index(inplace = True, drop = True)
        self.feature_data.reset_index(inplace = True, drop = True)
        self.current_step = 0
        self.shares_buy_and_hold = self.initial_balance / self.data['Close'][0]
        if self.start_with_stock_pos:
            self.shares_held = self.shares_buy_and_hold
            self.balance = 0
        else:
            self.balance = self.initial_balance
            self.shares_held = 0
        return self._next_observation()


    def print_available_tickers(self):
        return self.tickers


    def _prepare_features(self):

        self.feature_data = self.data.Close.diff(self.momentum_window).to_frame('momentum')
        self.feature_data.loc[:, 'binary_momentum'] = (self.feature_data.momentum > 0) * 1
        self.feature_data.loc[:, 'return_t'] = self.data.Close.pct_change()
        for lag in range(1, self.momentum_window):
            col = f'return_t-{lag}'
            self.feature_data[col] = self.feature_data['return_t'].shift(lag)
        feature_names = self.feature_data.columns.tolist()
        feature_names.reverse()
        self.feature_data = self.feature_data.loc[:, feature_names]
        self.feature_data.dropna(inplace = True)

        #self.feature_data = self.data.loc[:, 'Close'].diff(self.momentum_window).dropna().to_frame('momentum')
        #self.feature_data.loc[:, 'binary_momentum'] = (self.feature_data.momentum > 0) * 1
        #self.feature_data = self.data.loc[:, 'Close'].diff(self.momentum_window).dropna().apply(lambda x: 1 if x > 0 else 0).to_frame('momentum')
        self.data = self.data.iloc[self.momentum_window:]


    def _buy_stock(self):
        # buy all shares
        share_price = self.data['Open'][self.current_step]
        max_trades = self.balance / share_price
        max_trading_volume = max_trades * share_price
        available_trading_volume = (1 - self.transaction_costs_pct) * max_trading_volume
        stocks_to_buy = available_trading_volume / share_price
        self.balance = 0
        self.shares_held += stocks_to_buy


    def _sell_stock(self):
        # sell all shares
        share_price = self.data['Open'][self.current_step]
        self.balance += (self.shares_held * share_price) * (1 - self.transaction_costs_pct)
        self.shares_held = 0
    
    
    def _next_observation(self):
        # 1 if all money is invested in stock, -1 if all money is in cash
        hold_pos = 1 if self.shares_held > 0 else 0
        if self.use_binary_momentum_only:
            obs = [self.feature_data.iloc[self.current_step]['binary_momentum'], hold_pos]
        else:
            obs = self.feature_data.drop(['binary_momentum'], axis = 1).iloc[self.current_step].values.tolist() + [hold_pos]
        return obs


    @staticmethod
    def _load_data():
        '''
        This function is made for internal usage.
        '''
        this_dir, _ = os.path.split(__file__)
        file_path = os.path.join(this_dir, 'data', 'yfinance_data.pickle')
        with open(file_path, 'rb') as handle:
            df = pickle.load(handle)

        return df


class SimpleMomentumAgent:
    def __init__(self):
        self.policy = {}
        self.policy[1, 0] = 1
        self.policy[1, 1] = 0
        self.policy[0, 1] = 1
        self.policy[0, 0] = 0

    def predict(self, s):
        return self.policy[(s[0], s[1])]
    

