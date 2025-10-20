# -*- coding: utf-8 -*-
"""从 dialog 文件中提取倾向值"""
import pickle as pkl
import os
import re
import json

DATA = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"

print("=" * 70)
print("从 dialog 提取倾向值")
print("=" * 70)

# 加载 dialog
with open(os.path.join(DATA, "dialog_240.pkl"), "rb") as f:
    dialogs = pkl.load(f)

print(f"\n总 agent 数: {len(dialogs)}")
print(f"类型: {type(dialogs)}")

# 检查第一个 agent 的对话
agent_0_dialog = dialogs[0]
print(f"\nAgent 0 的 dialog 类型: {type(agent_0_dialog)}")
print(f"长度: {len(agent_0_dialog)}")

# 打印前3条对话
print("\n前3条对话:")
for i, msg in enumerate(list(agent_0_dialog)[:3]):
    print(f"\n--- 对话 {i} ---")
    print(f"类型: {type(msg)}")
    if isinstance(msg, dict):
        print(f"键: {list(msg.keys())}")
        for k, v in msg.items():
            if isinstance(v, str):
                print(f"{k}: {v[:200] if len(v) > 200 else v}")
            else:
                print(f"{k}: {v}")
    elif isinstance(msg, str):
        print(f"内容（前500字符）:\n{msg[:500]}")
    else:
        print(f"内容: {msg}")

# 尝试查找倾向相关的关键词
print("\n" + "=" * 70)
print("搜索可能的倾向值关键词")
print("=" * 70)

keywords = [
    "propensity", "tendency", "probability", "likelihood",
    "work", "labor", "consumption", "consume",
    "0.", "1.", "score", "rating"
]

found_patterns = {}
for i, msg in enumerate(list(agent_0_dialog)[:10]):  # 只检查前10条
    msg_str = str(msg)
    for keyword in keywords:
        if keyword.lower() in msg_str.lower():
            if keyword not in found_patterns:
                found_patterns[keyword] = []
            # 提取上下文（前后50字符）
            pos = msg_str.lower().find(keyword.lower())
            context = msg_str[max(0, pos-50):min(len(msg_str), pos+len(keyword)+50)]
            found_patterns[keyword].append((i, context))

print(f"\n找到的关键词:")
for keyword, occurrences in found_patterns.items():
    print(f"\n  {keyword}: {len(occurrences)} 次")
    for idx, context in occurrences[:2]:  # 只显示前2个
        print(f"    对话{idx}: ...{context}...")

# 尝试解析 JSON 格式的决策
print("\n" + "=" * 70)
print("尝试解析决策格式")
print("=" * 70)

for i, msg in enumerate(list(agent_0_dialog)[:5]):
    msg_str = str(msg)
    
    # 尝试查找 JSON
    json_patterns = [
        r'\{[^}]*"[Ww]ork"[^}]*\}',
        r'\{[^}]*"[Ll]abor"[^}]*\}',
        r'\{[^}]*"[Cc]onsumption"[^}]*\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, msg_str)
        if matches:
            print(f"\n对话 {i} 找到可能的决策 JSON:")
            for match in matches[:2]:
                print(f"  {match}")
                try:
                    parsed = json.loads(match)
                    print(f"  解析成功: {parsed}")
                except:
                    pass

print("\n" + "=" * 70)
print("结论")
print("=" * 70)



