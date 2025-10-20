#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细检查states的数据结构
"""

import pickle as pkl
import json

DATA_DIR = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"
P_DENSE = DATA_DIR + r"\dense_log.pkl"

def inspect_structure():
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    states = dense.get("states", [])
    
    print("=" * 70)
    print("检查 states[10] 的结构（第11个月）")
    print("=" * 70)
    
    if len(states) < 11:
        print("数据不足")
        return
    
    month_10 = states[10]
    
    # 1. 检查planner
    print("\n1. Planner ('p') 的数据结构:")
    print("-" * 70)
    if 'p' in month_10:
        planner = month_10['p']
        print(f"Type: {type(planner)}")
        if isinstance(planner, dict):
            print(f"Keys: {list(planner.keys())}")
            for key in list(planner.keys())[:10]:  # 前10个key
                value = planner[key]
                print(f"  {key}: {type(value)} = {value if not isinstance(value, (list, dict)) else f'{type(value)} (len={len(value)})'}")
    
    # 2. 检查agent '0'
    print("\n2. Agent '0' 的数据结构:")
    print("-" * 70)
    if '0' in month_10:
        agent_0 = month_10['0']
        print(f"Type: {type(agent_0)}")
        if isinstance(agent_0, dict):
            print(f"\n顶层 Keys: {list(agent_0.keys())}")
            
            # 检查每个顶层key
            for key in agent_0.keys():
                value = agent_0[key]
                if isinstance(value, dict):
                    print(f"\n  '{key}': dict with {len(value)} keys")
                    print(f"    子Keys: {list(value.keys())[:20]}")  # 前20个
                    # 显示一些具体值
                    for sub_key in list(value.keys())[:5]:
                        sub_value = value[sub_key]
                        if not isinstance(sub_value, (dict, list)):
                            print(f"      {sub_key}: {sub_value}")
                elif isinstance(value, (list, tuple)):
                    print(f"\n  '{key}': {type(value).__name__} (len={len(value)})")
                else:
                    print(f"\n  '{key}': {type(value).__name__} = {value}")
    
    # 3. 寻找skill
    print("\n3. 寻找 skill 信息:")
    print("-" * 70)
    agent_0 = month_10.get('0', {})
    if isinstance(agent_0, dict):
        # 递归搜索skill
        def find_key(d, target_key, path=""):
            results = []
            if isinstance(d, dict):
                for k, v in d.items():
                    current_path = f"{path}.{k}" if path else k
                    if target_key.lower() in k.lower():
                        results.append((current_path, v))
                    results.extend(find_key(v, target_key, current_path))
            return results
        
        skill_results = find_key(agent_0, 'skill')
        if skill_results:
            print("找到skill相关字段:")
            for path, value in skill_results:
                print(f"  {path}: {value}")
        else:
            print("⚠️  未找到skill字段")
    
    # 4. 寻找价格和利率
    print("\n4. 寻找全局变量 (价格、利率):")
    print("-" * 70)
    planner = month_10.get('p', {})
    if isinstance(planner, dict):
        # 搜索价格
        price_keys = [k for k in planner.keys() if 'price' in str(k).lower()]
        print(f"包含'price'的keys: {price_keys}")
        for key in price_keys[:3]:
            print(f"  {key}: {planner[key]}")
        
        # 搜索利率
        rate_keys = [k for k in planner.keys() if 'rate' in str(k).lower() or k in ['r', 'interest']]
        print(f"\n包含'rate'或'r'的keys: {rate_keys}")
        for key in rate_keys[:3]:
            print(f"  {key}: {planner[key]}")

if __name__ == "__main__":
    inspect_structure()

