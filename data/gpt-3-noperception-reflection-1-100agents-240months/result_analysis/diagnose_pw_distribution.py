# -*- coding: utf-8 -*-
"""
诊断工作倾向（pw_i）的分布
"""
import pickle as pkl
import numpy as np
import pandas as pd

DATA = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\dense_log.pkl"

with open(DATA, "rb") as f:
    dense = pkl.load(f)

actions = dense.get("actions", [])

print("="*70)
print("工作决策分布诊断")
print("="*70)

# 统计每个agent的工作频率
work_rates = {}
for agent_id in range(100):
    agent_id_str = str(agent_id)
    work_count = 0
    total_count = 0
    
    for m in range(len(actions)):
        if agent_id_str in actions[m]:
            action = actions[m][agent_id_str]
            labor = action.get('SimpleLabor', 0)
            total_count += 1
            if labor >= 1:
                work_count += 1
    
    if total_count > 0:
        work_rates[agent_id] = work_count / total_count

work_rates_array = np.array(list(work_rates.values()))

print(f"\n工作频率统计（240个月）:")
print(f"  均值: {work_rates_array.mean():.2%}")
print(f"  标准差: {work_rates_array.std():.2%}")
print(f"  最小值: {work_rates_array.min():.2%}")
print(f"  最大值: {work_rates_array.max():.2%}")

print(f"\n分布:")
print(f"  从不工作 (0%): {(work_rates_array == 0).sum()} 个agent")
print(f"  很少工作 (<25%): {(work_rates_array < 0.25).sum()} 个agent")
print(f"  偶尔工作 (25-50%): {((work_rates_array >= 0.25) & (work_rates_array < 0.5)).sum()} 个agent")
print(f"  经常工作 (50-75%): {((work_rates_array >= 0.5) & (work_rates_array < 0.75)).sum()} 个agent")
print(f"  总是工作 (>75%): {(work_rates_array >= 0.75).sum()} 个agent")
print(f"  100%工作: {(work_rates_array == 1.0).sum()} 个agent")

print("\n" + "="*70)
if work_rates_array.std() < 0.1:
    print("问题: 工作决策缺乏变化！")
    print("  -> 大部分agent要么总是工作，要么总不工作")
    print("  -> 这会导致回归无法找到有意义的模式")
elif (work_rates_array == 0).sum() > 10 or (work_rates_array == 1.0).sum() > 10:
    print("警告: 存在极端agent（从不工作或总是工作）")
    print("  -> 这些agent的回归会失败或不可靠")
else:
    print("OK: 工作决策有合理的变化")
print("="*70)

# 与储蓄的关系
print("\n检查储蓄与工作频率的关系...")
states = dense.get("states", [])

# 计算平均储蓄
avg_savings = {}
for agent_id in range(100):
    agent_id_str = str(agent_id)
    savings_list = []
    
    for m in range(1, len(states)):  # 跳过初始状态
        if agent_id_str in states[m]:
            state = states[m][agent_id_str]
            inventory = state.get('inventory', {}) or {}
            coin = inventory.get('Coin', 0)
            savings_list.append(coin)
    
    if savings_list:
        avg_savings[agent_id] = np.mean(savings_list)

if len(avg_savings) > 0 and len(work_rates) > 0:
    # 排序
    sorted_agents = sorted(work_rates.keys(), key=lambda x: work_rates[x])
    
    print("\n工作频率最低的10个agent:")
    print("Agent | 工作率 | 平均储蓄")
    for agent_id in sorted_agents[:10]:
        if agent_id in avg_savings:
            print(f"{agent_id:5d} | {work_rates[agent_id]:6.1%} | {avg_savings[agent_id]:10.0f}")
    
    print("\n工作频率最高的10个agent:")
    print("Agent | 工作率 | 平均储蓄")
    for agent_id in sorted_agents[-10:]:
        if agent_id in avg_savings:
            print(f"{agent_id:5d} | {work_rates[agent_id]:6.1%} | {avg_savings[agent_id]:10.0f}")
    
    # 相关性
    work_list = []
    saving_list = []
    for agent_id in work_rates.keys():
        if agent_id in avg_savings:
            work_list.append(work_rates[agent_id])
            saving_list.append(avg_savings[agent_id])
    
    if len(work_list) > 0:
        correlation = np.corrcoef(work_list, saving_list)[0, 1]
        print(f"\n储蓄与工作频率的相关性: {correlation:.3f}")
        
        if correlation < -0.5:
            print("  -> 强负相关：储蓄越少，越需要工作（生存模式）")
        elif correlation > 0.5:
            print("  -> 强正相关：储蓄越多，越愿意工作（投资模式）")
        else:
            print("  -> 弱相关：储蓄与工作决策关系不大")



