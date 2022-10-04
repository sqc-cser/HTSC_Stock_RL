from Market import StockTradingEnv
from Data import train, INDICATORS
import visdom
import numpy as np
import torch
# 项目参数（超参数）
TRAIN_ITER = 10               # 学习迭代次数
BATCH_SIZE = 16               # 随机抽取BATCH_SIZE条数据。
LR = 0.001                    # 学习率 （learning rate）
EPSILON_START = 0.9           # 最优选择动作百分比 （greedy policy）,取值[0,1], 越大随机性越强
EPSILON_END = 0.05
EPSILON_DECAY = 500
GAMMA = 0.5                   # 奖励递减参数 （reward discount）
TARGET_REPLACE_ITER = 252     # Q 现实网络的更新频率 （target update frequency）
MEMORY_CAPACITY = 32          # 记忆库大小
REWARD_FUTURE_DATE = 10       # 根据之后第n天的收盘价定奖励


# env 配置
state_space = len(INDICATORS) # + 2
N_ACTIONS = 2                 # env.action_space.n
N_STATES = len(INDICATORS)    # env.observation_space.shape[0]
ENV_A_SHAPE = 0               # if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
EPSILON_START = EPSILON_START/ N_ACTIONS + 1 - EPSILON_START      # 根据研报修正EPSILON
EPSILON_END = EPSILON_END/N_ACTIONS + 1 - EPSILON_END


# 可视化配置
vis = visdom.Visdom(env=u'FinRL', use_incoming_socket=False)