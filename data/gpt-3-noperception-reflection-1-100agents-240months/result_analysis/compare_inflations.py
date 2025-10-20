#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比价格通胀率和工资通胀率
"""

import os
import pickle as pkl

# DummyUnpickler
class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

DATA_DIR = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"
ENV_FILE = os.path.join(DATA_DIR, "env_240.pkl")

def compute_inflation(history, name):
    """计算年度通胀率"""
    # 计算年度平均
    yearly_avg = []
    for y in range(0, len(history), 12):
        chunk = history[y:y+12]
        if len(chunk) == 12:
            yearly_avg.append(sum(chunk) / 12)
    
    # 计算通胀率
    inflation_rates = []
    for i in range(1, len(yearly_avg)):
        if yearly_avg[i-1] > 0:
            inflation = (yearly_avg[i] - yearly_avg[i-1]) / yearly_avg[i-1]
            inflation_rates.append(inflation)
    
    return yearly_avg, inflation_rates

def main():
    print("\n" + "=" * 70)
    print("对比价格通胀率 vs 工资通胀率")
    print("=" * 70)
    
    # 加载 env
    with open(ENV_FILE, "rb") as f:
        env = DummyUnpickler(f).load()
    
    world = env.world
    
    # 1. 价格数据（含P0）
    prices = list(map(float, world.price))
    print(f"\n价格数据长度: {len(prices)}")
    print(f"  P0 (初始价格): {prices[0]:.2f}")
    print(f"  月度价格 (240个月): {len(prices[1:])}")
    
    p0 = prices[0]
    monthly_prices = prices[1:]  # 240个月
    
    # 计算价格年度平均和通胀
    price_yearly_avg, price_inflation = compute_inflation(monthly_prices, "价格")
    
    # 第1年特殊处理（相对P0）
    price_inflation_full = [(price_yearly_avg[0] - p0) / p0] + price_inflation
    
    # 2. 工资数据（无W0）
    wages = list(map(float, world.wage))
    print(f"\n工资数据长度: {len(wages)}")
    print(f"  月度工资 (240个月): {len(wages)}")
    
    # 计算工资年度平均和通胀
    wage_yearly_avg, wage_inflation = compute_inflation(wages, "工资")
    
    # 3. 对比结果
    print("\n" + "=" * 70)
    print("年度通胀率对比")
    print("=" * 70)
    print(f"\n{'年份':<6} | {'价格通胀率':<12} | {'工资通胀率':<12} | {'差值':<10}")
    print("-" * 70)
    
    # 价格有20年数据，工资有19年数据
    print(f"  1   | {price_inflation_full[0]*100:>10.2f}% | {'N/A':>10} | {'N/A':>10}")
    
    for i in range(len(wage_inflation)):
        year = i + 2
        price_infl = price_inflation_full[i+1] * 100
        wage_infl = wage_inflation[i] * 100
        diff = price_infl - wage_infl
        
        print(f"  {year:<3} | {price_infl:>10.2f}% | {wage_infl:>10.2f}% | {diff:>9.2f}%")
    
    # 4. 统计对比
    print("\n" + "=" * 70)
    print("统计对比")
    print("=" * 70)
    
    # 对齐数据（使用第2-20年）
    price_inflation_aligned = price_inflation_full[1:]  # 第2-20年
    
    import numpy as np
    
    print(f"\n价格通胀率（第2-20年）:")
    print(f"  平均值: {np.mean(price_inflation_aligned)*100:.2f}%")
    print(f"  范围: {np.min(price_inflation_aligned)*100:.2f}% - {np.max(price_inflation_aligned)*100:.2f}%")
    
    print(f"\n工资通胀率（第2-20年）:")
    print(f"  平均值: {np.mean(wage_inflation)*100:.2f}%")
    print(f"  范围: {np.min(wage_inflation)*100:.2f}% - {np.max(wage_inflation)*100:.2f}%")
    
    # 相关性
    correlation = np.corrcoef(price_inflation_aligned, wage_inflation)[0, 1]
    print(f"\n价格通胀与工资通胀的相关系数: {correlation:.3f}")
    
    if correlation > 0.7:
        print("  → 强正相关：价格和工资同步上涨")
    elif correlation > 0.3:
        print("  → 中等正相关：价格和工资有一定同步性")
    elif correlation < -0.3:
        print("  → 负相关：价格和工资反向变化")
    else:
        print("  → 弱相关或无相关")

if __name__ == "__main__":
    main()

