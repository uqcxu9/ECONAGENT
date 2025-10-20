#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查回归数据为什么会导致Singular matrix
"""

import pickle as pkl
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"
P_DENSE = DATA_DIR + r"\dense_log.pkl"

def check_variance():
    """检查各个变量的方差"""
    
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    states = dense.get("states", [])
    actions = dense.get("actions", [])
    
    print("=" * 70)
    print("检查Agent 0的数据（前50个月）")
    print("=" * 70)
    
    agent_id = '0'
    data_points = []
    
    for m in range(min(50, len(actions))):
        if m >= len(states):
            break
            
        month_states = states[m]
        month_actions = actions[m]
        
        if agent_id not in month_states:
            continue
        
        agent_state = month_states[agent_id]
        if not isinstance(agent_state, dict):
            continue
        
        # 提取变量
        state_dict = agent_state.get('state', {})
        inventory = agent_state.get('inventory', {})
        endogenous = agent_state.get('endogenous', {})
        
        skill = state_dict.get('skill', 0)
        savings = inventory.get('Coin', 0)
        
        # 获取action
        agent_action = month_actions.get(agent_id, {})
        if isinstance(agent_action, dict):
            work_decision = agent_action.get('SimpleLabor', 0)
            consumption_decision = agent_action.get('SimpleConsumption', 0)
        else:
            continue
        
        # 价格和利率（假设为常数或从planner获取）
        price = 1.0  # 占位
        interest_rate = 0.05  # 占位
        
        data_point = {
            'month': m + 1,
            'skill': skill,
            'savings': savings,
            'work': work_decision,
            'consumption': consumption_decision,
            'v_i': skill * 168,  # 预期收入
            'price': price,
            'interest_rate': interest_rate
        }
        
        data_points.append(data_point)
    
    if not data_points:
        print("没有数据！")
        return
    
    df = pd.DataFrame(data_points)
    
    print(f"\n数据点数: {len(df)}")
    print("\n前10行:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("变量统计")
    print("=" * 70)
    
    variables = ['skill', 'savings', 'work', 'consumption', 'v_i']
    
    for var in variables:
        values = df[var].values
        print(f"\n{var}:")
        print(f"  均值: {np.mean(values):.4f}")
        print(f"  标准差: {np.std(values):.4f}")
        print(f"  最小值: {np.min(values):.4f}")
        print(f"  最大值: {np.max(values):.4f}")
        print(f"  唯一值数量: {len(np.unique(values))}")
        print(f"  唯一值: {np.unique(values)[:10]}")  # 前10个
    
    # 检查是否有常数列
    print("\n" + "=" * 70)
    print("常数列检查")
    print("=" * 70)
    
    for var in variables:
        if df[var].std() == 0:
            print(f"⚠️  {var} 是常数列！值={df[var].iloc[0]}")
        else:
            print(f"✓  {var} 有变化")
    
    # 检查相关性
    print("\n" + "=" * 70)
    print("变量相关性矩阵")
    print("=" * 70)
    
    corr_matrix = df[variables].corr()
    print(corr_matrix)
    
    # 检查完全共线性
    print("\n" + "=" * 70)
    print("共线性检查")
    print("=" * 70)
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                corr = corr_matrix.loc[var1, var2]
                if abs(corr) > 0.99:
                    print(f"⚠️  {var1} 和 {var2} 高度相关: {corr:.4f}")

if __name__ == "__main__":
    check_variance()

