# -*- coding: utf-8 -*-
"""
诊断4个关键问题：
1. 从env.world.wage反推skill
2. 检查GPT-4模型是否考虑了skill差异
3. 价格的变异系数（CV）是否太小
4. Agent的prompt中是否强调了价格信息
"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径设置
BASE = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent"
DATA_DIR = os.path.join(BASE, "data", "gpt-3-noperception-reflection-1-100agents-240months")
OUT = os.path.join(DATA_DIR, "result_analysis")

# DummyUnpickler
class DummyModule:
    def __getattr__(self, name):
        return DummyModule()

class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

def load_pickle(filepath):
    """使用DummyUnpickler加载pickle文件"""
    with open(filepath, "rb") as f:
        return DummyUnpickler(f).load()

print("="*70)
print("关键问题诊断")
print("="*70)

# ========== 问题1: 从env.world.wage反推skill ==========
print("\n" + "="*70)
print("问题1: 从 env.world.wage 反推 skill")
print("="*70)

env_file = os.path.join(DATA_DIR, "env_240.pkl")
dense_file = os.path.join(DATA_DIR, "dense_log.pkl")

try:
    env = load_pickle(env_file)
    print(f"成功加载: {env_file}")
    
    # 尝试获取wage数据
    wage_data = None
    if hasattr(env, 'world'):
        if hasattr(env.world, 'wage'):
            wage_data = env.world.wage
            print(f"找到 env.world.wage，长度: {len(wage_data)}")
        else:
            print("env.world 存在，但没有 wage 属性")
            print(f"env.world 的属性: {dir(env.world)}")
    else:
        print("env 没有 world 属性")
        print(f"env 的属性: {dir(env)}")
    
    if wage_data is not None:
        wage_array = np.array(wage_data)
        
        print(f"\n工资统计:")
        print(f"  均值: {wage_array.mean():.2f}")
        print(f"  标准差: {wage_array.std():.2f}")
        print(f"  变异系数(CV): {wage_array.std()/wage_array.mean():.4f}")
        print(f"  范围: [{wage_array.min():.2f}, {wage_array.max():.2f}]")
        
        # 反推skill: wage / 168小时
        skill_inferred = wage_array / 168.0
        print(f"\n反推的 skill (wage/168):")
        print(f"  均值: {skill_inferred.mean():.4f}")
        print(f"  标准差: {skill_inferred.std():.4f}")
        print(f"  变异系数(CV): {skill_inferred.std()/skill_inferred.mean():.4f}")
        print(f"  范围: [{skill_inferred.min():.4f}, {skill_inferred.max():.4f}]")
        
        if skill_inferred.std() < 0.01:
            print("\n警告: skill 变化极小！几乎所有agent的skill都一样")
            print("  -> 这会导致 vi (预期收入) 对决策没有解释力")
        else:
            print("\nOK: skill 有合理的变化范围")
    
except Exception as e:
    print(f"错误: {e}")

# ========== 问题2: 检查GPT-4模型是否考虑了skill差异 ==========
print("\n" + "="*70)
print("问题2: 检查 GPT-4 模型是否考虑了 skill 差异")
print("="*70)

try:
    dense = pkl.load(open(dense_file, "rb"))
    states = dense.get("states", [])
    
    print(f"总月份数: {len(states)}")
    
    # 检查第1个月（跳过第0个月的初始状态）
    if len(states) > 1:
        month1_states = states[1]
        
        skills_month1 = []
        for agent_id, agent_state in month1_states.items():
            if agent_id == 'p':
                continue
            
            # 尝试多种可能的skill字段
            skill = None
            if isinstance(agent_state, dict):
                # 方法1: state['skill']
                skill = agent_state.get('skill', None)
                
                # 方法2: state['state']['skill']
                if skill is None and 'state' in agent_state:
                    skill = agent_state['state'].get('skill', None)
                
                # 方法3: state['endogenous']['skill']
                if skill is None and 'endogenous' in agent_state:
                    skill = agent_state['endogenous'].get('skill', None)
                
                # 方法4: state['expected skill']
                if skill is None:
                    skill = agent_state.get('expected skill', None)
            
            if skill is not None:
                skills_month1.append(float(skill))
        
        if skills_month1:
            skills_array = np.array(skills_month1)
            print(f"\n第1个月的 skill 数据（{len(skills_month1)} 个agent）:")
            print(f"  均值: {skills_array.mean():.4f}")
            print(f"  标准差: {skills_array.std():.4f}")
            print(f"  变异系数(CV): {skills_array.std()/skills_array.mean():.4f}" if skills_array.mean() > 0 else "  CV: N/A (均值为0)")
            print(f"  范围: [{skills_array.min():.4f}, {skills_array.max():.4f}]")
            print(f"  唯一值数量: {len(np.unique(skills_array))}")
            
            if skills_array.std() < 0.01:
                print("\n问题: skill 几乎没有差异！")
                print("  -> GPT-4 可能没有考虑个体差异")
                print("  -> 或者 skill 字段没有被正确设置")
            elif len(np.unique(skills_array)) < 5:
                print(f"\n警告: skill 只有 {len(np.unique(skills_array))} 种不同的值")
                print("  -> 可能是离散化的skill等级，而非连续分布")
            else:
                print("\nOK: skill 有合理的个体差异")
        else:
            print("\n问题: 在 states 中找不到 skill 数据！")
            print("尝试的字段:")
            print("  - state['skill']")
            print("  - state['state']['skill']")
            print("  - state['endogenous']['skill']")
            print("  - state['expected skill']")
            
            # 打印第一个agent的结构
            first_agent_id = [k for k in month1_states.keys() if k != 'p'][0]
            print(f"\n第一个agent (ID={first_agent_id}) 的数据结构:")
            print(f"  顶层键: {list(month1_states[first_agent_id].keys())}")
            
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

# ========== 问题3: 价格的变异系数（CV）是否太小 ==========
print("\n" + "="*70)
print("问题3: 价格的变异系数（CV）是否太小")
print("="*70)

try:
    env = load_pickle(env_file)
    
    price_data = None
    if hasattr(env, 'world'):
        if hasattr(env.world, 'price'):
            price_data = env.world.price
            print(f"找到 env.world.price，长度: {len(price_data)}")
        else:
            print("env.world 存在，但没有 price 属性")
    
    if price_data is not None:
        price_array = np.array(price_data)
        
        print(f"\n价格统计:")
        print(f"  均值: {price_array.mean():.2f}")
        print(f"  标准差: {price_array.std():.2f}")
        print(f"  变异系数(CV): {price_array.std()/price_array.mean():.4f}")
        print(f"  范围: [{price_array.min():.2f}, {price_array.max():.2f}]")
        
        # 月环比变化
        monthly_changes = np.diff(price_array) / price_array[:-1] * 100
        print(f"\n月环比变化:")
        print(f"  均值: {monthly_changes.mean():.2f}%")
        print(f"  标准差: {monthly_changes.std():.2f}%")
        print(f"  范围: [{monthly_changes.min():.2f}%, {monthly_changes.max():.2f}%]")
        
        # 年度平均价格的变化
        yearly_prices = []
        for y in range(0, len(price_array), 12):
            chunk = price_array[y:y+12]
            if len(chunk) == 12:
                yearly_prices.append(chunk.mean())
        
        yearly_prices = np.array(yearly_prices)
        yearly_cv = yearly_prices.std() / yearly_prices.mean()
        print(f"\n年度平均价格统计:")
        print(f"  均值: {yearly_prices.mean():.2f}")
        print(f"  标准差: {yearly_prices.std():.2f}")
        print(f"  变异系数(CV): {yearly_cv:.4f}")
        
        # 判断
        if price_array.std() / price_array.mean() < 0.05:
            print("\n问题: 价格波动太小（CV < 5%）！")
            print("  -> agent 感受不到明显的价格变化")
            print("  -> 价格对决策的影响会很弱")
        elif monthly_changes.std() < 1.0:
            print("\n警告: 月度价格变化很稳定（标准差 < 1%）")
            print("  -> 可能影响 agent 对价格的敏感度")
        else:
            print("\nOK: 价格有合理的波动")
            
except Exception as e:
    print(f"错误: {e}")

# ========== 问题4: Agent的prompt中是否强调了价格信息 ==========
print("\n" + "="*70)
print("问题4: Agent 的 prompt 中是否强调了价格信息")
print("="*70)

# 检查 simulate.py 中的 prompt
simulate_file = os.path.join(BASE, "simulate.py")

if os.path.exists(simulate_file):
    print(f"读取: {simulate_file}")
    
    with open(simulate_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 搜索关键词
    keywords = {
        'price': 'price 或 Price',
        'wage': 'wage 或 Wage',
        'skill': 'skill',
        'expected': 'expected',
        'income': 'income',
        'salary': 'salary',
        'cost': 'cost'
    }
    
    print("\n关键词搜索结果:")
    for key, desc in keywords.items():
        count = content.lower().count(key.lower())
        print(f"  {desc}: 出现 {count} 次")
    
    # 尝试找到GPT prompt的定义
    print("\n尝试提取 GPT prompt 定义...")
    
    # 查找可能包含prompt的函数
    import re
    
    # 搜索包含"prompt"或"message"的行
    prompt_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        if 'prompt' in line.lower() or ('message' in line.lower() and 'gpt' in content.split('\n')[max(0, i-10):i+10]):
            prompt_lines.append((i, line.strip()))
    
    if prompt_lines:
        print(f"\n找到 {len(prompt_lines)} 行包含 prompt 相关的代码:")
        for line_num, line in prompt_lines[:10]:  # 只显示前10行
            print(f"  第{line_num}行: {line[:80]}...")
    
    # 搜索 gpt_actions 或 complex_actions 函数
    gpt_actions_match = re.search(r'def gpt_actions\(.*?\):(.*?)(?=\ndef |\Z)', content, re.DOTALL)
    if gpt_actions_match:
        gpt_func = gpt_actions_match.group(1)
        
        print("\n在 gpt_actions 函数中检查是否提到价格:")
        price_mentioned = 'price' in gpt_func.lower()
        wage_mentioned = 'wage' in gpt_func.lower()
        skill_mentioned = 'skill' in gpt_func.lower()
        
        print(f"  提到 price: {'是' if price_mentioned else '否'}")
        print(f"  提到 wage: {'是' if wage_mentioned else '否'}")
        print(f"  提到 skill: {'是' if skill_mentioned else '否'}")
        
        if not price_mentioned:
            print("\n警告: gpt_actions 中可能没有强调价格信息！")
            print("  -> agent 可能不会考虑价格变化")
        
        if not skill_mentioned:
            print("\n警告: gpt_actions 中可能没有强调 skill 信息！")
            print("  -> agent 可能不会考虑自身技能水平")
    else:
        print("\n未找到 gpt_actions 函数定义")
    
    print("\n建议: 手动检查 simulate.py 中 gpt_actions 函数的 prompt 内容")
    print(f"  文件路径: {simulate_file}")
    
else:
    print(f"错误: 找不到 {simulate_file}")

# ========== 总结 ==========
print("\n" + "="*70)
print("诊断总结")
print("="*70)

print("\n建议下一步:")
print("1. 如果 skill 变化极小：")
print("   - 检查 one_step_economy.py 中 skill 的初始化")
print("   - 确认是否使用了 profiles.json 中的 skill 数据")
print("")
print("2. 如果价格波动太小：")
print("   - 检查供需机制是否正常运作")
print("   - 可能需要调整经济参数以增加波动性")
print("")
print("3. 如果 prompt 中没有强调价格/skill：")
print("   - 修改 simulate.py 中的 gpt_actions 函数")
print("   - 在 prompt 中明确告知 agent 当前的价格和自身skill")
print("")
print("4. 核心问题可能是：")
print("   - 失业率过高（13-26%）导致劳动力市场失灵")
print("   - 储蓄成为主导因素，而非价格/收入")

print("\n" + "="*70)
print("诊断完成")
print("="*70)



