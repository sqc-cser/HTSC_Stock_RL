import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from Config import *
import matplotlib.pyplot as plt


# 定义神经网络class
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # N_STATES 与 图像的特征值个数有关
        self.fc1 = nn.Linear(N_STATES, 128)    # N_ACTIONS 与 能做的动作个数有关
        self.fc1.weight.data.normal_(0, 0.1)   # 初始化权重，用二值分布来随机生成参数的值
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, N_ACTIONS)    # 做出每个动作后，每个动作的价值作为输出。
        self.out.weight.data.normal_(0, 0.1)    # 初始化权重，用二值分布来随机生成参数的值
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.out(x)
        actions_value = self.softmax_out(x)
        return actions_value


class DQN(object):
    def __init__(self, seed=0):
        self.eval_net, self.target_net = Net(), Net()                   # eval_net: (actor) target_net: (critic)
        self.eval_net.train()                                           # 初始化为训练网络
        self.target_net.eval()                                          # 初始化为只评估(参数都是照搬eval_net的)
        self.learn_step_counter = 0                                     # 用来记录学习到第几步了
        self.memory_counter = 0                                         # 用来记录当前指到数据库的第几个数据了
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR, betas=(0.9, 0.999)
        )                                                               # 优化器，优化评估神经网络 仅优化eval_net
        self.loss_func = nn.SmoothL1Loss()
        self.loss = 0
        self.EPSILON = 0

    def choose_action(self, x):
        # 获取输入
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 选择 max - value
        self.EPSILON = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-self.learn_step_counter/EPSILON_DECAY)
        if np.random.uniform() < self.EPSILON:   # greedy # 随机结果是否大于EPSILON（0.9）
            self.eval_net.eval()            # 调为eval模式，仅用于评估单状态
            actions_value = self.eval_net.forward(x) # if 取max方法选择执行动作
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # 变异情况
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        """
        存储数据
        :param s: 本次状态
        :param a: 执行的动作
        :param r: 获得的奖励
        :param s_: 完成动作后产生的下一个状态
        """
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY                   # index 是 这一次录入的数据在 3000 的哪一个位置
        self.memory[index, :] = transition                              # 如果记忆超过上线,我们重新索引,即覆盖老的记忆
        self.memory_counter += 1

    def learn(self):
        """
        从存储学习数据
        target net(actor): 达到次数后更新
        eval net(critic): 每次learn 就进行更新
        """
        self.eval_net.train()
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:           # 如果步数达到一定程度, 让actor成为目前的critic
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 把critic的所有参数赋值到actor中
        self.learn_step_counter += 1
        #  eval net是每次learn 就进行更新
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 从 数据库中随机抽取 BATCH_SIZE条数据
        b_memory = self.memory[sample_index, :]                       # 把这BATCH_SIZE个数据打包
        # 下面这些变量是 32个数据打包的变量
        # ===========================================================================
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])                         # 当时的状态
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))    # 当时的动作
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])             # 当初的奖励
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])                       # 执行动作后的下一个动作状态
        # ===========================================================================
        # q_eval的学习过程
        # self.eval_net(b_s).gather(1, b_a)  输入我们包（32条）中的所有状态 并得到（32条）所有状态的所有动作价值， .gather(1,b_a) 只取这32个状态中 的 每一个状态的最大值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # eval_net 根据state预测动作价值, .gather(1, b_a) 选择动作价值最大的动作
        # 输入下一个状态 进入critic 输出下一个动作的价值  .detach() 阻止网络反向传递，我们的target需要自己定义该如何更新，它的更新在learn那一步
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_target 实际价值的计算  ==  当前价值 + GAMMA（未来价值递减参数） * 未来的价值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # TD算法, 引入了真实奖励b_r信息
        # q_eval预测值， q_target真实值
        self.loss = self.loss_func(q_eval, q_target)         # 根据误差，去优化我们eval网
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()                           # eval_net 优化一次
        self._visualize()

    def _visualize(self, step=100):
        if self.learn_step_counter % step == 0:
            # vis.line(Y=torch.tensor([self.EPSILON]), X=torch.tensor([self.learn_step_counter]), win="epsilon", env="FinRL", update="append", opts={"title": "epsilon"})
            vis.line(Y=torch.tensor([self.loss]), X=torch.tensor([self.learn_step_counter]), win="loss", env="FinRL", update="append", name="train loss", opts={"title": "loss"})


def plot(env, save_place='DQN.png'):
    plot_df = pd.DataFrame(env.date_memory, columns=['date']);
    plot_df['return'] = env.return_list
    plot_df['close'] = env.df['close']
    plot_df['close'] = plot_df['close'].shift(-1) / plot_df['close'] - 1
    plot_df['return'] = plot_df['return'].shift(-1)
    plot_df.set_index('date', inplace=True)
    plot_df = plot_df.cumsum()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(plot_df.index, plot_df['close'], label='origin')
    ax.plot(plot_df.index, plot_df['return'], label='strategy')
    ax.plot(plot_df.index, plot_df['return'] - plot_df['close'], label='excess return')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_place)


def train_DQN(env: StockTradingEnv, seed=0):
    dqn = DQN(seed=seed)
    print('\nCollecting experience...')
    for i_episode in range(TRAIN_ITER):
        s = env.reset()  # 获得初始化 observation 环境特征
        ep_r = 0         # 作为一个计数变量，来统计我第n次训练。 完成所有动作的分的总和
        while True:
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)    # s_为后一天的情况, info为往后看n天的state
            # 存储数据  每完成一个动作，记忆存储数据一次
            dqn.store_transition(s, a, r, info)
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:    # 数据库中多于memory size时候就会学习
                dqn.learn()
                if done:
                    print('Ep: ', i_episode+1, '| Ep_reward: ', round(ep_r, 2))  # 打印这是i_episode次训练 ， Ep_reward代表这次的总分
                    print('==============================')
            if done:
                break
            s = s_
        if i_episode % 5 == 4:
            plot(env, save_place=f"results/DQN_train_{i_episode+1}_ep.png")
    return dqn


def test_DQN(env_config, test_data: pd.DataFrame, dqn: DQN):
    """
    :param env_config: dict
    :param test_data: trade data about test time
    :param dqn: DQN object
    """
    print("start predicting...")
    env = StockTradingEnv(test_data, **env_config)
    s = env.reset()
    ep_r = 0
    while True:
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        ep_r += r
        if done:
            break
        s = s_
    print(f'reward: {ep_r}')
    # plot
    plot(env=env, save_place="results/DQN_TEST.png")

