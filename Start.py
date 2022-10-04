import warnings
import pandas as pd
import pickle
import torch
import numpy as np
from Market import StockTradingEnv
from Agent import train_DQN, test_DQN, plot
from Data import train, INDICATORS, test
from Config import REWARD_FUTURE_DATE
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None


def create_model(save_place: str, seed):
    """
    :param save_place: string, name of .pkl file
    :param seed: random seed
    """
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(SEED)   # 为所有GPU设置随机种子
    np.random.seed(seed)
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(INDICATORS))
    # state为 [0]:仓位情况(1表示全仓做多,0表示空仓) [1]:当日close价格 [1:tech_indicator+1]: 技术指标(特征工程)
    print(f"training model with seed {seed}, save to {save_place}")
    env_kwargs = {
        "buy_cost_pct": 0.0005,
        "sell_cost_pct": 0.0005,
        "state_space": state_space,
        "tech_indicator_list": INDICATORS,
        "reward_future_day": REWARD_FUTURE_DATE,
        "seed": seed
    }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    dqn = train_DQN(e_train_gym, seed=seed)
    if save_place.endswith('.pkl'):
        out_put = open(save_place, 'wb')
        dqn_pkl = pickle.dumps(dqn)
        out_put.write(dqn_pkl)
        out_put.close()
    print("hit end!\n")
    test_DQN(env_kwargs, test, dqn)


def load_model(save_place: str, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    with open(save_place, 'rb') as file:
        dqn = pickle.loads(file.read())
    env_kwargs = {
        "buy_cost_pct": 0.0005,
        "sell_cost_pct": 0.0005,
        "state_space": len(INDICATORS),
        "tech_indicator_list": INDICATORS,
        "reward_future_day": REWARD_FUTURE_DATE,
        "seed": seed
    }
    env = StockTradingEnv(test, **env_kwargs)
    s = env.reset()
    action_list = []
    while True:
        a = dqn.choose_action(s)
        action_list.append(a)
        s_, r, done, info = env.step(a)
        if done:
            break
        s = s_
    plot(env, f'results/DQN_model_seed_{seed}')
    return action_list


if __name__ == "__main__":
    create_model(f"models/dqn_seed_{0}.pkl", seed=0)
    # for i in range(0, 100, 10):
    #     create_model(f"models/dqn_seed_{i}.pkl", seed=i)

    # model2signal = pd.DataFrame(None)
    # for i in range(0, 100, 10):
    #     model2signal[f'seed{i}'] = load_model(f"models/dqn_seed_{i}.pkl", seed=i)
    # final_signal = model2signal.apply(lambda x: x.value_counts(), axis=1)
    # final_signal = final_signal.apply(lambda x: x.idxmax(), axis=1).tolist()
    # env_kwargs = {
    #     "buy_cost_pct": 0.0005,
    #     "sell_cost_pct": 0.0005,
    #     "state_space": len(INDICATORS),
    #     "tech_indicator_list": INDICATORS,
    #     "reward_future_day": REWARD_FUTURE_DATE,
    # }
    # env = StockTradingEnv(test, **env_kwargs)
    # s = env.reset()
    # step = 0
    # while True:
    #     a = final_signal[step]
    #     s_, r, done, info = env.step(a)
    #     if done:
    #         break
    #     step += 1
    # plot(env, "results/combine_signal.png")