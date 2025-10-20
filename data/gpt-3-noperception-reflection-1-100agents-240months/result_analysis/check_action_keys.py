# -*- coding: utf-8 -*-
"""检查 actions 中实际的键名"""
import pickle as pkl
import os
from pathlib import Path

DATA = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"

# 加载 dense_log
with open(os.path.join(DATA, "dense_log_240.pkl"), "rb") as f:
    dense = pkl.load(f)

actions = dense["actions"]

print("=" * 70)
print("检查 actions 中的键名")
print("=" * 70)

# 检查第1个月，第1个agent
month_0 = actions[0]
print(f"\n第1个月的 agent 数量: {len(month_0)}")

# 找一个非 'p' 的 agent
for agent_id, action in month_0.items():
    if str(agent_id) != 'p':
        print(f"\nAgent {agent_id} 的 action 键:")
        print(f"类型: {type(action)}")
        if isinstance(action, dict):
            print(f"所有键: {list(action.keys())}")
            print(f"\n完整内容:")
            for k, v in action.items():
                print(f"  {k}: {v} (类型: {type(v).__name__})")
        else:
            print(f"内容: {action}")
        break

# 统计所有月份中 action 的键名
print("\n" + "=" * 70)
print("统计所有键名出现频率")
print("=" * 70)

all_keys = {}
for m in range(min(10, len(actions))):  # 只检查前10个月
    month_actions = actions[m]
    for agent_id, action in month_actions.items():
        if str(agent_id) == 'p':
            continue
        if isinstance(action, dict):
            for k in action.keys():
                all_keys[k] = all_keys.get(k, 0) + 1

print("\n出现的键名（按频率排序）:")
for k, count in sorted(all_keys.items(), key=lambda x: -x[1]):
    print(f"  {k}: {count} 次")



