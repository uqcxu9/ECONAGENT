# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('panel_data_v2.csv')

print("=" * 70)
print("pw_i 数据诊断")
print("=" * 70)

print(f"\n总行数: {len(df)}")
print(f"Agent数量: {df['agent'].nunique()}")

print("\n=== pw_i 统计 ===")
print(f"均值: {df['pw_i'].mean():.6f}")
print(f"标准差: {df['pw_i'].std():.6f}")
print(f"最小值: {df['pw_i'].min():.6f}")
print(f"最大值: {df['pw_i'].max():.6f}")
print(f"唯一值数量: {df['pw_i'].nunique()}")

print("\n=== pw_i 值分布（前20个） ===")
print(df['pw_i'].value_counts().head(20))

print("\n=== pc_i 统计 ===")
print(f"均值: {df['pc_i'].mean():.6f}")
print(f"标准差: {df['pc_i'].std():.6f}")
print(f"最小值: {df['pc_i'].min():.6f}")
print(f"最大值: {df['pc_i'].max():.6f}")

print("\n=== 各 agent 的 pw_i 方差 ===")
agent_pw_var = df.groupby('agent')['pw_i'].agg(['mean', 'std', 'min', 'max'])
print(agent_pw_var.head(10))
print(f"\npw_i 标准差为0的agent数量: {(agent_pw_var['std'] == 0).sum()}")
print(f"pw_i 标准差<0.01的agent数量: {(agent_pw_var['std'] < 0.01).sum()}")
print(f"pw_i 标准差<0.1的agent数量: {(agent_pw_var['std'] < 0.1).sum()}")

print("\n=== 关键诊断 ===")
print(f"pw_i 几乎是常数？ {'是' if df['pw_i'].std() < 0.01 else '否'}")
print(f"pw_i 全为0？ {'是' if (df['pw_i'] == 0).all() else '否'}")
print(f"pw_i 全为1？ {'是' if (df['pw_i'] == 1).all() else '否'}")
print(f"pw_i=0的比例: {(df['pw_i'] == 0).sum() / len(df) * 100:.2f}%")
print(f"pw_i=1的比例: {(df['pw_i'] == 1).sum() / len(df) * 100:.2f}%")



