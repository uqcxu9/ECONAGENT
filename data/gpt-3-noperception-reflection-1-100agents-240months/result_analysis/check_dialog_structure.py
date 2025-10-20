# -*- coding: utf-8 -*-
"""详细检查 dialog 结构"""
import pickle as pkl
import os

DATA = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"

with open(os.path.join(DATA, "dialog_240.pkl"), "rb") as f:
    dialogs = pkl.load(f)

print("=" * 70)
print("dialog 结构详细检查")
print("=" * 70)

print(f"\n总agent数: {len(dialogs)}")

# 检查 agent 0
agent_0 = list(dialogs[0])
print(f"\nAgent 0 的对话数量: {len(agent_0)}")

print("\n前10条对话的 role:")
for i, msg in enumerate(agent_0[:10]):
    role = msg.get('role', 'UNKNOWN') if isinstance(msg, dict) else 'NOT_DICT'
    print(f"  {i}: {role}")

# 统计 assistant 回复数量
assistant_count = sum(1 for msg in agent_0 if isinstance(msg, dict) and msg.get('role') == 'assistant')
user_count = sum(1 for msg in agent_0 if isinstance(msg, dict) and msg.get('role') == 'user')

print(f"\nAgent 0 统计:")
print(f"  assistant 回复数: {assistant_count}")
print(f"  user 提示数: {user_count}")
print(f"  总对话数: {len(agent_0)}")

# 提取所有 assistant 回复
assistant_msgs = [msg for msg in agent_0 if isinstance(msg, dict) and msg.get('role') == 'assistant']

print(f"\n前5个 assistant 回复的内容:")
for i, msg in enumerate(assistant_msgs[:5]):
    content = msg.get('content', '')
    print(f"\n  月{i+1}: {content[:100]}")

# 检查是否有月份对应关系
print("\n" + "=" * 70)
print("月份对应关系检查")
print("=" * 70)

print("\n假设: assistant 回复按顺序对应月份")
print(f"  如果有240个月，应该有240个 assistant 回复")
print(f"  实际 assistant 回复数: {assistant_count}")

if assistant_count == 240:
    print("\n  ✓ 完美匹配！")
elif assistant_count == 3:
    print("\n  ✗ 只有3个回复 - 这是 3个月的模拟数据！")
    print("  ⚠️  你可能用错了数据文件！")
elif assistant_count < 240:
    print(f"\n  ✗ 回复数不足，缺少 {240 - assistant_count} 个月")
elif assistant_count > 240:
    print(f"\n  ⚠️ 回复数过多，可能包含其他信息")

print("\n检查所有agent的回复数:")
all_counts = []
for agent_id in range(min(10, len(dialogs))):
    agent_msgs = list(dialogs[agent_id])
    assistant_cnt = sum(1 for msg in agent_msgs if isinstance(msg, dict) and msg.get('role') == 'assistant')
    all_counts.append(assistant_cnt)
    print(f"  Agent {agent_id}: {assistant_cnt} 个 assistant 回复")

print(f"\n是否所有agent都一样: {len(set(all_counts)) == 1}")



