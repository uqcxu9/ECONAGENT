# -*- coding: utf-8 -*-
"""
检查individual_regression.py中vi的实际值
"""

import pickle as pkl
import numpy as np

DATA = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\dense_log.pkl"

with open(DATA, "rb") as f:
    dense = pkl.load(f)

states = dense.get("states", [])
actions = dense.get("actions", [])

print("="*70)
print("检查 vi (预期收入) 的实际值")
print("="*70)

# 检查第1个月（跳过初始月）
if len(states) > 1 and len(actions) > 0:
    month1_states = states[1]
    month1_actions = actions[0]
    
    vi_values = []
    skill_values = []
    expected_skill_values = []
    
    for agent_id in range(100):
        agent_id_str = str(agent_id)
        
        if agent_id_str in month1_states and agent_id_str in month1_actions:
            state = month1_states[agent_id_str]
            
            # 使用和individual_regression.py相同的逻辑
            expected_skill = state.get("expected skill", state.get("skill", 0.0))
            vi = float(expected_skill) * 168.0
            
            skill = state.get("skill", 0.0)
            exp_skill = state.get("expected skill", 0.0)
            
            vi_values.append(vi)
            skill_values.append(skill)
            expected_skill_values.append(exp_skill)
    
    if vi_values:
        vi_array = np.array(vi_values)
        skill_array = np.array(skill_values)
        exp_skill_array = np.array(expected_skill_values)
        
        print("\n第1个月的数据统计:")
        print(f"Agent数量: {len(vi_values)}")
        
        print(f"\nvi (预期月收入):")
        print(f"  均值: {vi_array.mean():.2f}")
        print(f"  标准差: {vi_array.std():.2f}")
        print(f"  变异系数(CV): {vi_array.std()/vi_array.mean():.4f}" if vi_array.mean() > 0 else "  CV: N/A")
        print(f"  范围: [{vi_array.min():.2f}, {vi_array.max():.2f}]")
        print(f"  零值数量: {(vi_array == 0).sum()}")
        
        print(f"\nskill:")
        print(f"  均值: {skill_array.mean():.4f}")
        print(f"  标准差: {skill_array.std():.4f}")
        print(f"  范围: [{skill_array.min():.4f}, {skill_array.max():.4f}]")
        print(f"  零值数量: {(skill_array == 0).sum()}")
        
        print(f"\nexpected skill:")
        print(f"  均值: {exp_skill_array.mean():.4f}")
        print(f"  标准差: {exp_skill_array.std():.4f}")
        print(f"  范围: [{exp_skill_array.min():.4f}, {exp_skill_array.max():.4f}]")
        print(f"  零值数量: {(exp_skill_array == 0).sum()}")
        
        print("\n" + "="*70)
        if vi_array.std() / vi_array.mean() < 0.05:
            print("问题: vi的变化太小（CV < 5%）！")
            print("  -> 这会导致vi对决策没有解释力")
        elif (vi_array == 0).sum() > 50:
            print("问题: 超过一半的agent的vi为0！")
            print("  -> skill或expected skill数据有问题")
        else:
            print("OK: vi有合理的个体差异")
        print("="*70)



