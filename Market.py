import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
matplotlib.use("Agg")


# state: 持仓情况+当日close+技术指标们(特征工程)
# action: 买入全仓, 持仓不动, 卖空全仓
# reward: reward_future_day 后的收益率
# epoch: 跑完一次全交易数据
class StockTradingEnv(gym.Env):
    def __init__(
        self, df, buy_cost_pct, sell_cost_pct, state_space, tech_indicator_list,
        day=0, initial=True, seed=0, reward_future_day=5
    ):
        self.day = day                              # 交易第几天
        self.df = df                                # 交易数据
        self.buy_cost_pct = buy_cost_pct            # 买入手续费
        self.sell_cost_pct = sell_cost_pct          # 卖出手续费
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list  # 技术指标(factor)名字的list
        self.action_space = spaces.Discrete(2)      # action 为0(空),1(多)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )                                          # state为任意连续值
        self.data = self.df.loc[self.day, :]
        self.terminal = False                      # 是否到达终点
        self.initial = initial
        self.reward_future_day = reward_future_day
        self.state = self._initiate_state()
        # initialize reward
        self.reward = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.date_memory = [self._get_date()]
        self._seed(seed=seed)
        self.return_list = [0]                       # 收益率list

    def _buy_and_sell(self, actions):
        """
        做交易函数
        :param actions: int -1表示sell,0表示hold,1表示buy
        """
        # state为 [0]:仓位情况(1表示全仓做多,0表示空仓) [1]:当日close价格 [1:tech_indicator+1]: 技术指标(特征工程)
        if self.state[0] > 0:
            if actions == 0:  # 当前持多仓，发出卖出信号
                self.state[0] = 0
                self.trades += 1
            # 其他情况说明持多仓,继续买入或者继续看多，仓位不变
        else:
            if actions == 1:  # 当前持空仓，发出买入信号
                self.state[0] = 1
                self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1  # 需要确保index为日期,否则unique之后并不是所有日期,会有重复
        # actions = actions - 1  # actions is -1, 0, 1
        # print(f"now state: {self.state}; action: {actions}")
        if self.terminal:
            # 如果 one episode trade 结束
            print(f"Episode: {self.episode} end;  total trades: {self.trades}")
            df = pd.DataFrame(self.date_memory, columns=['date'])
            df['daily_return'] = self.return_list
            if df["daily_return"].std() != 0:
                sharpe = ((252**0.5) * df["daily_return"].mean() / df["daily_return"].std())
            df['daily_return'] = df['daily_return'].cumsum()
            if df["daily_return"].std() != 0:
                print(f"end_return: {df['daily_return'].iloc[-1]:0.2f};  Sharpe: {sharpe:0.3f}")
        else:
            # 如果决策没有碰到数据集结束
            self._buy_and_sell(actions)             # 收盘时间为决策点
            self.day += 1
            self.data = self.df.loc[self.day, :]
            close_record = self.state[1]
            # 利用当日收盘的state和几日后的refer_state比较，计算reward
            try:
                self.refer_state = self._update_state(future_date=self.reward_future_day)  # 更新n天后的收盘价情况
            except:
                self.refer_state = self._update_state(future_date=1)  # 如果n天后的数据没有导致错误, 使用正常1天后的数据
            if self.state[0] > 0:
                if actions == 0:  # 当前持多仓，发出卖出信号
                    self.reward = (1 - self.sell_cost_pct) * (2 - self.refer_state[1]/self.state[1]) -1
                elif actions == 1:
                    self.reward = self.refer_state[1]/self.state[1] - 1
            else:
                if actions == 1:  # 当前持空仓，发出买入信号
                    self.reward = (1 - self.buy_cost_pct) * self.refer_state[1]/self.state[1] - 1
                elif actions == 0:
                    self.reward = 1 - self.refer_state[1]/self.state[1]
            # state更新了一天, 昨日收盘给出action,买入卖出后(此时仓位就变化了),更新时间,然后进行收盘结算
            self.state = self._update_state(future_date=1)
            self.date_memory.append(self._get_date())
            if self.state[0] > 0:
                self.return_list.append(self.state[1]/close_record-1)
            else:
                self.return_list.append(0)
        return self.state[2:], self.reward, self.terminal, self.refer_state[2:]  # return state, reward, 是否结束(Bool), info

    def reset(self):
        """
        初始化环境
        """
        self.state = self._initiate_state()
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.return_list = [0]
        self.trades = 0
        self.reward = 0
        self.terminal = False
        self.date_memory = [self._get_date()]
        self.episode += 1
        return self.state[2:]

    def render(self):
        if self.state[0] > 0:
            print(f"交易日:{self.df.iloc[self.day]['date']}, 持仓: 全仓")
        else:
            print(f"交易日:{self.df.iloc[self.day]['date']}, 持仓: 空仓")
        return self.state

    def _initiate_state(self):
        """
        初始化 state 变量
        state = 持仓情况+当日close+技术指标们(特征工程)
        """
        state = [0] + self.df.loc[0, :][['close']+self.tech_indicator_list].tolist()
        return state

    def _update_state(self, future_date=1):
        """
        更新future_date日之后的账户状态(账户现金和持仓股数不变,only变化close价格)
        :param future_date: int
        :return state
        """
        state = [self.state[0]] + self.df.loc[self.day+future_date-1, :][['close']+self.tech_indicator_list].tolist()
        return state

    def _get_date(self):
        return self.data.date

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
