import pickle as pkl
import numpy as np

with open('../dense_log_240.pkl', 'rb') as f:
    dense = pkl.load(f)

states = dense['states']
coins = []

for m in range(min(5, len(states))):
    for aid, s in states[m].items():
        if str(aid) != 'p' and isinstance(s, dict):
            coin = s.get('inventory', {}).get('Coin', 0)
            coins.append(coin)

coins_arr = np.array(coins)
print(f'前5个月的Coin样本数: {len(coins_arr)}')
print(f'最小值: {coins_arr.min()}')
print(f'最大值: {coins_arr.max()}')
print(f'均值: {coins_arr.mean():.4f}')
print(f'标准差: {coins_arr.std():.4f}')
print(f'非零样本数: {(coins_arr != 0).sum()}')
print(f'唯一值数量: {np.unique(coins_arr).shape[0]}')

