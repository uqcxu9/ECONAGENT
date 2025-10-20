# -*- coding: utf-8 -*-
"""寻找倾向值可能存储的位置"""
import pickle as pkl
import os
import json

DATA = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"

print("=" * 70)
print("搜索倾向值")
print("=" * 70)

# 1. 检查 dense_log 的所有键
print("\n1. 检查 dense_log 的顶层键:")
with open(os.path.join(DATA, "dense_log_240.pkl"), "rb") as f:
    dense = pkl.load(f)

print(f"   dense_log 的键: {list(dense.keys())}")

# 2. 检查 states 中是否有倾向值
print("\n2. 检查 states[0][agent_0] 的所有键:")
states = dense["states"]
for agent_id, state in states[1].items():  # 第1个月（index 1）
    if str(agent_id) != 'p' and isinstance(state, dict):
        print(f"\n   Agent {agent_id} 的 state 键:")
        print(f"   顶层键: {list(state.keys())}")
        
        # 递归打印所有嵌套键
        def print_nested(d, prefix="   "):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"{prefix}{k}:")
                    print_nested(v, prefix + "  ")
                else:
                    print(f"{prefix}{k}: {type(v).__name__}")
        
        print("\n   完整结构:")
        print_nested(state)
        break

# 3. 检查 dialog 文件
print("\n3. 检查 dialog 文件:")
dialog_files = [
    "dialog_240.pkl",
    "dialog4ref_240.pkl",
    "dialogs/0"  # Agent 0 的对话
]

for fname in dialog_files:
    fpath = os.path.join(DATA, fname)
    if os.path.exists(fpath):
        print(f"\n   ✓ 找到: {fname}")
        try:
            with open(fpath, "rb") as f:
                data = pkl.load(f)
            print(f"     类型: {type(data)}")
            if isinstance(data, dict):
                print(f"     键: {list(data.keys())[:10]}")
            elif isinstance(data, list):
                print(f"     长度: {len(data)}")
                if len(data) > 0:
                    print(f"     第1个元素类型: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"     第1个元素的键: {list(data[0].keys())}")
                    elif isinstance(data[0], str):
                        print(f"     第1个元素（前200字符）: {data[0][:200]}")
        except Exception as e:
            print(f"     ✗ 读取失败: {e}")
    else:
        print(f"   ✗ 不存在: {fname}")

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
print("如果以上都没有倾向值，说明:")
print("1. 该模拟版本确实没有记录倾向值")
print("2. 只能使用 SimpleLabor (0/1) 和 SimpleConsumption (0-50)")
print("3. 需要用 Logit 回归来处理二元的 pw_i")



