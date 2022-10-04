# 数据部分
import pandas as pd
import tushare as ts

token = '******************************'
train_start_date = '2006-01-04'
train_stop_date = '2017-01-04'
val_start_date = '2017-01-04'
val_stop_date = '2022-09-21'
look_back = 5
INDICATORS = [f'look_back_{i}_open' for i in range(look_back)] + [f'look_back_{i}_close' for i in range(look_back)] +\
             [f'look_back_{i}_high' for i in range(look_back)] + [f'look_back_{i}_low' for i in range(look_back)]


def data_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def get_data(code='399300.SZ')->pd.DataFrame:
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.index_daily(ts_code=code, start_date=train_start_date, end_date=val_stop_date)
    df.sort_values(by='trade_date', inplace=True)
    # 特征工程 Z-score
    for i in range(look_back):
        df[f'look_back_{i}_open'] = (df['open'].shift(i) - df['close'].expanding(252).mean()) / df['close'].expanding(
            252).std()
        df[f'look_back_{i}_close'] = (df['close'].shift(i) - df['close'].expanding(252).mean()) / df['close'].expanding(
            252).std()
        df[f'look_back_{i}_high'] = (df['high'].shift(i) - df['close'].expanding(252).mean()) / df['close'].expanding(
            252).std()
        df[f'look_back_{i}_low'] = (df['low'].shift(i) - df['close'].expanding(252).mean()) / df['close'].expanding(
            252).std()
    df.dropna(inplace=True)
    return df

# if __name__ == "__main__":
#     df = get_data('000001.SH')
#     df.to_csv('data.csv')
df = pd.read_csv('data.csv')
df['trade_date'] = df['trade_date'].astype(str)
df.rename(columns={"ts_code": "tic", "trade_date": "date"}, inplace=True)
train = data_split(df, train_start_date, train_stop_date)
test = data_split(df, val_start_date, val_stop_date)
