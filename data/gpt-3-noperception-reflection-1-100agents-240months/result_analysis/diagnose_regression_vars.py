#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断回归变量的分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\result_analysis\panel_data.csv"

def diagnose():
    print("=" * 70)
    print("回归变量诊断")
    print("=" * 70)
    
    # 读取数据
    df = pd.read_csv(CSV_FILE)
    
    print(f"\n数据行数: {len(df)}")
    print(f"变量列表: {df.columns.tolist()}")
    
    # 关键变量
    vars_to_check = ['vi', 'ci_hat', 'T_zi', 'zr', 'P_dev', 'si', 'r_dev', 'pw_i', 'pc_i']
    
    print("\n" + "=" * 70)
    print("变量统计")
    print("=" * 70)
    
    stats_list = []
    
    for var in vars_to_check:
        if var not in df.columns:
            print(f"\n⚠️  变量 {var} 不存在")
            continue
        
        values = df[var].dropna()
        
        stats = {
            '变量': var,
            '均值': f"{values.mean():.4f}",
            '标准差': f"{values.std():.4f}",
            '最小值': f"{values.min():.4f}",
            '最大值': f"{values.max():.4f}",
            '唯一值数': len(values.unique()),
            '缺失率': f"{df[var].isna().sum()/len(df)*100:.1f}%"
        }
        stats_list.append(stats)
        
        print(f"\n{var}:")
        print(f"  均值: {values.mean():.4f}")
        print(f"  标准差: {values.std():.4f}")
        print(f"  变异系数(CV): {values.std()/values.mean():.4f}" if values.mean() != 0 else "  CV: N/A (均值为0)")
        print(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
        print(f"  唯一值数: {len(values.unique())}")
        print(f"  缺失率: {df[var].isna().sum()/len(df)*100:.1f}%")
        
        # 检查是否为常数
        if values.std() < 1e-10:
            print(f"  ⚠️  {var} 几乎是常数！")
        
        # 检查异常值
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        outliers = ((values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)).sum()
        if outliers > 0:
            print(f"  ⚠️  异常值: {outliers} ({outliers/len(values)*100:.1f}%)")
    
    # 保存统计表
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(CSV_FILE.replace("panel_data.csv", "variable_statistics.csv"), index=False)
    print(f"\n✓ 保存统计表")
    
    # 相关性分析
    print("\n" + "=" * 70)
    print("变量间相关性（可能的共线性问题）")
    print("=" * 70)
    
    numeric_vars = [v for v in vars_to_check if v in df.columns and df[v].dtype in ['float64', 'int64']]
    corr_matrix = df[numeric_vars].corr()
    
    # 找出高相关（>0.8）的变量对
    high_corr_pairs = []
    for i in range(len(numeric_vars)):
        for j in range(i+1, len(numeric_vars)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((numeric_vars[i], numeric_vars[j], corr_val))
    
    if high_corr_pairs:
        print("\n高度相关的变量对 (|r| > 0.8):")
        for var1, var2, corr in high_corr_pairs:
            print(f"  {var1} <-> {var2}: {corr:.3f}")
    else:
        print("\n✓ 没有发现高度共线性问题")
    
    # 对比论文预期
    print("\n" + "=" * 70)
    print("与论文结果对比")
    print("=" * 70)
    
    print("\n关键差异分析:")
    
    # vi 影响弱
    if 'vi' in df.columns:
        vi_cv = df['vi'].std() / df['vi'].mean() if df['vi'].mean() != 0 else 0
        print(f"\n1. vi (预期收入) 影响较弱:")
        print(f"   - 变异系数: {vi_cv:.4f}")
        if vi_cv < 0.2:
            print(f"   ⚠️  vi 变化太小！agent收入差异不大")
            print(f"   → 建议: 检查skill分布是否合理")
    
    # P_dev 影响弱
    if 'P_dev' in df.columns:
        p_cv = df['P_dev'].std() / abs(df['P_dev'].mean()) if df['P_dev'].mean() != 0 else 0
        print(f"\n2. P_dev (价格偏离) 影响较弱:")
        print(f"   - 价格波动CV: {p_cv:.4f}")
        if p_cv < 0.5:
            print(f"   ⚠️  价格变化太稳定！")
            print(f"   → 建议: 检查价格是否正常波动")
    
    # si 影响过强
    if 'si' in df.columns:
        si_cv = df['si'].std() / df['si'].mean() if df['si'].mean() != 0 else 0
        print(f"\n3. si (储蓄) 对工作倾向影响过强:")
        print(f"   - 储蓄变异系数: {si_cv:.4f}")
        if si_cv > 1.0:
            print(f"   ⚠️  储蓄波动非常大！")
            print(f"   → 这可能导致储蓄成为主导因素")
            print(f"   → 失业率高时，穷人被迫工作效应明显")
    
    # 绘制分布图
    print("\n" + "=" * 70)
    print("绘制变量分布图...")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, var in enumerate(vars_to_check):
        if var in df.columns and idx < 9:
            ax = axes[idx]
            values = df[var].dropna()
            ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(f'{var}\n(mean={values.mean():.2f}, std={values.std():.2f})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CSV_FILE.replace("panel_data.csv", "variable_distributions.png"), dpi=300)
    print("✓ 保存分布图")
    
    print("\n" + "=" * 70)
    print("诊断完成！")
    print("=" * 70)

if __name__ == "__main__":
    diagnose()

