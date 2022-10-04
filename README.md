## DQN研报复现
### 项目介绍
复现华泰证券《强化学习初探与DQN择时》研报中的DQN模型与效果

项目架构如下:
- models(文件夹): 存储不同组随机数产生的DQN模型
- results(文件夹): 存储复现择时结果
- Agent: DQN类
- Config: 项目配置
- Data: 数据文件
- Market: StockTradeEnv
- Start: 运行开始的文件
- requirements: 安装所需库
### 模型介绍
受到华泰上述研报和GitHub高星仓库FinRL的影响，开发者希望通过强化学习方法做择时策略

### 项目使用
运行Start.py文件

启动visdom可视化

```terminal
python -m visdom.server
# nohup python -m visdom.server &    # 后台开启
```
在后台开启visdom后，实时过程查看网址: localhost:8097
### 项目进度

v0.1.0(已完成)  完成交易员,交易环境,强化学习基本算法的开发

v0.1.1(未完成)  调试超参数,尽可能复现模型

v0.1.0(未完成)  特征工程,将遗传规划等方法用于因子挖掘,将因子加入Env的state中

v0.2.0(未完成)  将单标的扩展到多标的算法

