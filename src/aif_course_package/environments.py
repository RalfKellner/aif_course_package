import gym
import pandas as pd
import numpy as np
import random
import os

class SimpleMomentumEnv(gym.Env):

    def __init__(self, start_date, end_date, momentum_window, initial_balance, tickers = None, n_companies = None):

        if tickers and n_companies:
            print('Please either provide a list of tickers or the number of companies to be selected randomly. If both are provided, the ticker list is prioritized.')
        if not(tickers) and not(n_companies):
            raise ValueError('Please either provide a list of tickers or the number of companies to be selected randomly.')

        self.start_date = start_date
        self.end_date = end_date
        self.momentum_window = momentum_window
        self.initial_balance = initial_balance
        self.state_to_meaning = {(0,0): ['no momentum', 'cash pos.'], (0,1): ['no momentum', 'stock pos.'], (1,0): ['momentum', 'cash pos.'], (1,1): ['momentum', 'stock pos.']}
        self.action_to_meaning = {0: 'hold pos.', 1: 'change pos.'}
        
        self.stocks_data = self._load_data()
        self.stocks_data_in_time = self.stocks_data[(self.stocks_data.date > self.start_date) & (self.stocks_data.date < self.end_date)]
        self.all_tickers = self.stocks_data_in_time.ticker.unique().tolist()
        if tickers:
            self.tickers = [ticker for ticker in tickers if ticker in self.all_tickers]
            not_in_stocks_data = [ticker for ticker in tickers if ticker not in self.all_tickers]
            if len(not_in_stocks_data) > 0:
                print(f'The following tickers are not in the stocks data and will not be used: {not_in_stocks_data}')
        else:
            self.tickers = random.sample(self.all_tickers, k=n_companies)
        self.current_ticker = 0

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.MultiBinary(2)


    def reset(self):
        self.data = self.stocks_data[(self.stocks_data.ticker==self.tickers[self.current_ticker]) & (self.stocks_data.date > self.start_date) & (self.stocks_data.date < self.end_date)]
        print(f'Episode runs with ticker {self.tickers[self.current_ticker]}')
        self.current_ticker += 1
        self._prepare_features()
        self.data.reset_index(inplace = True, drop = True)
        self.feature_data.reset_index(inplace = True, drop = True)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.shares_buy_and_hold = self.balance / self.data['close'][0]
        return self._next_observation()
        

    def step(self, action):        
        
        previous_value = self.balance + self.shares_held * self.data['close'][self.current_step]
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

        current_value = self.balance + self.shares_held * self.data['close'][self.current_step]
        reward = np.log(current_value) - np.log(previous_value) 
        reward_buy_and_hold = np.log(self.shares_buy_and_hold * self.data['close'][self.current_step]) - np.log(self.shares_buy_and_hold * self.data['close'][self.current_step-1])
        done = self.current_step == len(self.data) - 1
        obs = self._next_observation()
        additional_info = pd.concat((self.data.iloc[self.current_step-1].to_frame('t'), self.data.iloc[self.current_step].to_frame('t+1')), axis = 1)
        additional_info.loc['shares_held'] = [starting_shares, self.shares_held]
        additional_info.loc['balance'] = [starting_balance, self.balance]
        info = {}
        info['action'] = action
        info['step_data'] = additional_info
        return obs, reward-reward_buy_and_hold, done, info


    def render(self, mode = 'human'):
        print(f'Current step: {self.current_step}, current value: {self.balance + self.shares_held*self.data["close"][self.current_step]}')            
    

    def _prepare_features(self):
        self.feature_data = self.data.loc[:, 'close'].diff(self.momentum_window).dropna().apply(lambda x: 1 if x > 0 else 0).to_frame('momentum')
        self.data = self.data.iloc[self.momentum_window:]


    def _buy_stock(self):
        # buy all shares
        share_price = self.data['open'][self.current_step]
        stocks_to_buy = self.balance / share_price
        self.shares_held += stocks_to_buy
        self.balance -= stocks_to_buy * share_price


    def _sell_stock(self):
        # sell all shares
        share_price = self.data['open'][self.current_step]
        self.balance += self.shares_held * share_price
        self.shares_held = 0
    
    
    def _next_observation(self):
        # 1 if all money is invested in stock, -1 if all money is in cash
        hold_pos = 1 if self.shares_held > 0 else 0
        obs = [self.feature_data.iloc[self.current_step]['momentum'], hold_pos]
        return obs
    
    @staticmethod
    def _load_data():
        '''
        This function is made for internal usage.
        '''
        this_dir, _ = os.path.split(__file__)
        data_path = os.path.join(this_dir, 'data', 'aif_course_asset_data.csv')
        df = pd.read_csv(data_path)

        return df