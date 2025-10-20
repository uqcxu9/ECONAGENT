#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断失业率波动过大的原因
"""

import pickle as pkl
import numpy as np

DATA_DIR = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"
P_DENSE = DATA_DIR + r"\dense_log.pkl"

def analyze_unemployment_volatility():
    """分析失业率波动的原因"""
    
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    states = dense.get("states", [])
    actions = dense.get("actions", [])
    
    print("=" * 70)
    print("失业率波动诊断")
    print("=" * 70)
    
    # 1. 计算月度失业率
    monthly_unemp_rate = []
    monthly_unemp_count = []
    
    for m in range(1, len(states)):
        unemployed = sum(1 for k, v in states[m].items() 
                        if k != 'p' and isinstance(v, dict) 
                        and v.get('endogenous', {}).get('job') == 'Unemployment')
        total = sum(1 for k in states[m].keys() if k != 'p')
        
        monthly_unemp_rate.append(unemployed / total if total > 0 else 0)
        monthly_unemp_count.append(unemployed)
    
    # 2. 计算年度失业率
    yearly_unemp_rate = []
    yearly_unemp_count = []
    
    for y in range(0, len(monthly_unemp_rate), 12):
        chunk_rate = monthly_unemp_rate[y:y+12]
        chunk_count = monthly_unemp_count[y:y+12]
        if len(chunk_rate) == 12:
            yearly_unemp_rate.append(sum(chunk_rate) / 12)
            yearly_unemp_count.append(sum(chunk_count) / 12)
    
    # 3. 计算失业率增长
    unemp_growth = [yearly_unemp_rate[i] - yearly_unemp_rate[i-1] 
                   for i in range(1, len(yearly_unemp_rate))]
    
    print("\n1. 基本统计")
    print("-" * 70)
    print(f"失业率水平:")
    print(f"  平均: {np.mean(yearly_unemp_rate)*100:.2f}%")
    print(f"  最小: {np.min(yearly_unemp_rate)*100:.2f}%")
    print(f"  最大: {np.max(yearly_unemp_rate)*100:.2f}%")
    print(f"  标准差: {np.std(yearly_unemp_rate)*100:.2f}%")
    
    print(f"\n失业率增长:")
    print(f"  平均: {np.mean(unemp_growth)*100:+.2f} 百分点")
    print(f"  最小: {np.min(unemp_growth)*100:+.2f} 百分点")
    print(f"  最大: {np.max(unemp_growth)*100:+.2f} 百分点")
    print(f"  标准差: {np.std(unemp_growth)*100:.2f} 百分点")
    
    # 4. 分析极端变化年份
    print("\n2. 极端变化年份分析")
    print("-" * 70)
    
    # 最大上升
    max_increase_idx = np.argmax(unemp_growth)
    print(f"\n最大上升: 第{max_increase_idx+2}年")
    print(f"  变化: {unemp_growth[max_increase_idx]*100:+.2f} 百分点")
    print(f"  从 {yearly_unemp_rate[max_increase_idx]*100:.2f}% → {yearly_unemp_rate[max_increase_idx+1]*100:.2f}%")
    print(f"  失业人数: {yearly_unemp_count[max_increase_idx]:.1f} → {yearly_unemp_count[max_increase_idx+1]:.1f}")
    
    # 最大下降
    max_decrease_idx = np.argmin(unemp_growth)
    print(f"\n最大下降: 第{max_decrease_idx+2}年")
    print(f"  变化: {unemp_growth[max_decrease_idx]*100:+.2f} 百分点")
    print(f"  从 {yearly_unemp_rate[max_decrease_idx]*100:.2f}% → {yearly_unemp_rate[max_decrease_idx+1]*100:.2f}%")
    print(f"  失业人数: {yearly_unemp_count[max_decrease_idx]:.1f} → {yearly_unemp_count[max_decrease_idx+1]:.1f}")
    
    # 5. 月度波动分析
    print("\n3. 月度波动分析（前24个月）")
    print("-" * 70)
    print("月份 | 失业率 | 月环比变化")
    print("-" * 40)
    
    for m in range(min(24, len(monthly_unemp_rate))):
        rate = monthly_unemp_rate[m] * 100
        if m > 0:
            mom_change = (monthly_unemp_rate[m] - monthly_unemp_rate[m-1]) * 100
            print(f"{m+1:3d}  | {rate:6.2f}% | {mom_change:+6.2f} pp")
        else:
            print(f"{m+1:3d}  | {rate:6.2f}% | {'N/A':>6}")
    
    # 6. 劳动力市场动态
    print("\n4. 劳动力市场动态（年度）")
    print("-" * 70)
    print("年份 | 平均失业人数 | 就业人数 | 就业率")
    print("-" * 50)
    
    for y, (urate, ucount) in enumerate(zip(yearly_unemp_rate, yearly_unemp_count), 1):
        employed = 100 - ucount
        employment_rate = (1 - urate) * 100
        print(f"{y:3d}  | {ucount:8.1f}      | {employed:6.1f}   | {employment_rate:6.2f}%")
    
    # 7. 对比论文预期
    print("\n5. 与论文预期对比")
    print("-" * 70)
    print("指标                   | 论文预期 | 你的结果")
    print("-" * 50)
    print(f"失业率范围             | 2-12%    | {np.min(yearly_unemp_rate)*100:.2f}-{np.max(yearly_unemp_rate)*100:.2f}%")
    print(f"失业率增长范围         | ±2pp     | {np.min(unemp_growth)*100:+.2f} 到 {np.max(unemp_growth)*100:+.2f}pp")
    print(f"失业率标准差           | <3%      | {np.std(yearly_unemp_rate)*100:.2f}%")
    
    # 8. 可能原因总结
    print("\n6. 可能原因分析")
    print("-" * 70)
    
    if np.max(yearly_unemp_rate) > 0.20:
        print("⚠️  失业率过高 (>20%)")
        print("    → 可能原因: 劳动力市场匹配效率低、工资过高、需求不足")
    
    if np.std(unemp_growth) > 0.03:
        print("⚠️  失业率增长波动过大 (标准差 >{:.2f}pp)".format(np.std(unemp_growth)*100))
        print("    → 可能原因: 外部冲击大、政策不稳定、Agent决策不稳定")
    
    # 计算连续上升/下降的周期
    consecutive_increase = 0
    consecutive_decrease = 0
    max_consecutive_increase = 0
    max_consecutive_decrease = 0
    
    for g in unemp_growth:
        if g > 0:
            consecutive_increase += 1
            consecutive_decrease = 0
            max_consecutive_increase = max(max_consecutive_increase, consecutive_increase)
        else:
            consecutive_decrease += 1
            consecutive_increase = 0
            max_consecutive_decrease = max(max_consecutive_decrease, consecutive_decrease)
    
    if max_consecutive_increase > 3:
        print(f"⚠️  连续上升周期长 (最长{max_consecutive_increase}年)")
        print("    → 可能原因: 经济衰退、结构性失业、负反馈循环")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_unemployment_volatility()

