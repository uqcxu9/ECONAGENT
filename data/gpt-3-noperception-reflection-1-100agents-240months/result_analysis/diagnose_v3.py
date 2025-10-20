# -*- coding: utf-8 -*-
"""诊断 v3 版本的数据"""
import pandas as pd
import numpy as np

df = pd.read_csv('panel_data_v3_dialog.csv')

print("=" * 70)
print("v3 数据诊断")
print("=" * 70)

print(f"\n总行数: {len(df)}")
print(f"Agent数量: {df['agent'].nunique()}")

print("\n=== pw_i 统计 ===")
print(f"均值: {df['pw_i'].mean():.6f}")
print(f"标准差: {df['pw_i'].std():.6f}")
print(f"最小值: {df['pw_i'].min():.6f}")
print(f"最大值: {df['pw_i'].max():.6f}")
print(f"唯一值数量: {df['pw_i'].nunique()}")

print("\n=== pc_i 统计 ===")
print(f"均值: {df['pc_i'].mean():.6f}")
print(f"标准差: {df['pc_i'].std():.6f}")
print(f"最小值: {df['pc_i'].min():.6f}")
print(f"最大值: {df['pc_i'].max():.6f}")
print(f"唯一值数量: {df['pc_i'].nunique()}")

print("\n=== 自变量统计 ===")
for var in ['vi', 'ci_hat', 'T_zi', 'zr', 'P', 'si', 'r']:
    print(f"\n{var}:")
    print(f"  均值: {df[var].mean():.4f}")
    print(f"  标准差: {df[var].std():.4f}")
    print(f"  最小值: {df[var].min():.4f}")
    print(f"  最大值: {df[var].max():.4f}")
    print(f"  NaN数量: {df[var].isna().sum()}")
    print(f"  全为0?: {(df[var] == 0).all()}")
    print(f"  全为NaN?: {df[var].isna().all()}")

print("\n=== 按 agent 检查 pw_i 方差 ===")
agent_pw_stats = df.groupby('agent')['pw_i'].agg(['mean', 'std', 'count'])
print(agent_pw_stats.head(10))
print(f"\npw_i 标准差为0的agent数量: {(agent_pw_stats['std'] == 0).sum()}")
print(f"pw_i 标准差为NaN的agent数量: {agent_pw_stats['std'].isna().sum()}")
print(f"每个agent的样本数: {agent_pw_stats['count'].unique()}")

print("\n=== 检查前10行数据 ===")
print(df.head(10)[['agent', 'month', 'pw_i', 'pc_i', 'vi', 'si', 'ci_hat']])

print("\n=== 关键诊断 ===")
print(f"pw_i 全为常数?: {df['pw_i'].std() < 1e-10}")
print(f"pc_i 全为常数?: {df['pc_i'].std() < 1e-10}")
print(f"vi 全为0?: {(df['vi'] == 0).all()}")
print(f"si 全为0?: {(df['si'] == 0).all()}")



