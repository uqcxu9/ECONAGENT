# -*- coding: utf-8 -*-
"""
测试3：心理账户（短期vs长期）

假设：Agent更受短期损益主导，而非长期财富水平
检验方法：
  回归：labor_decision ~ α + β₁·wealth_t + β₂·recent_profit + controls
  
期望结果（心理账户假说）：
  - β₁ 不显著（长期财富影响小）
  - β₂ 显著且大（短期损益影响大）
  - |β₂| > |β₁|
"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
import re
import statsmodels.api as sm
import matplotlib.pyplot as plt

BASE = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent"
MODEL = "gpt-3-noperception-reflection-1-100agents-240months"
DATA = os.path.join(BASE, "data", MODEL)
OUT = os.path.join(DATA, "result_analysis")

def load_data():
    """加载数据"""
    dense_path = os.path.join(DATA, "dense_log_240.pkl")
    with open(dense_path, "rb") as f:
        dense = pkl.load(f)
    return dense

def extract_panel(dense):
    """提取面板数据"""
    states = dense["states"]
    actions = dense["actions"]
    n_months = len(actions)
    
    # 先提取所有agent的时间序列
    agent_series = {}
    
    for m in range(n_months + 1):
        state_t = states[m]
        
        for agent_id in state_t.keys():
            if str(agent_id) == 'p':
                continue
            if not (isinstance(agent_id, (int, np.integer)) or str(agent_id).isdigit()):
                continue
            
            agent_id_int = int(agent_id)
            
            if agent_id_int not in agent_series:
                agent_series[agent_id_int] = {
                    'month': [],
                    'wealth': [],
                    'worked': [],
                    'skill': None
                }
            
            s = state_t.get(agent_id_int, {}) or {}
            
            # 财富
            wealth = float(((s.get('inventory', {}) or {}).get('Coin', 0.0)) or 0.0)
            
            # 技能（只需记录一次）
            if agent_series[agent_id_int]['skill'] is None:
                skill = float(((s.get('endogenous', {}) or {}).get('skill', np.nan)) or np.nan)
                agent_series[agent_id_int]['skill'] = skill
            
            # 工作决策（从actions中提取）
            if m < n_months:
                act = actions[m].get(agent_id_int, {}) or {}
                labor_action = int(act.get('SimpleLabor', 0))
                worked = 1 if labor_action == 1 else 0
            else:
                worked = 0
            
            agent_series[agent_id_int]['month'].append(m)
            agent_series[agent_id_int]['wealth'].append(wealth)
            agent_series[agent_id_int]['worked'].append(worked)
    
    # 计算短期损益和构造面板
    rows = []
    for agent_id, data in agent_series.items():
        wealth_arr = np.array(data['wealth'])
        worked_arr = np.array(data['worked'])
        skill = data['skill']
        
        # 从第3期开始（需要计算短期利润）
        for t in range(2, len(wealth_arr) - 1):
            # 当期财富
            wealth_t = wealth_arr[t]
            
            # 短期损益（最近1-3期的财富变化）
            # 方案1：最近1期变化
            recent_profit_1 = wealth_arr[t] - wealth_arr[t-1]
            
            # 方案2：最近3期平均变化
            recent_profit_3 = (wealth_arr[t] - wealth_arr[t-3]) / 3.0
            
            # 当期工作决策
            worked_t = worked_arr[t]
            
            rows.append({
                'agent': agent_id,
                'month': t + 1,
                'worked': worked_t,
                'wealth': wealth_t,
                'recent_profit_1': recent_profit_1,
                'recent_profit_3': recent_profit_3,
                'skill': skill
            })
    
    df = pd.DataFrame(rows)
    return df

def test_mental_accounting(df):
    """测试心理账户效应"""
    print("=" * 70)
    print("测试3：心理账户（短期vs长期）")
    print("=" * 70)
    
    print(f"\n总样本数: {len(df)}")
    
    # 过滤有效数据
    df_valid = df[np.isfinite(df['skill'])].copy()
    
    print(f"有效样本: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("\n⚠️  警告：没有有效样本！")
        return None
    
    # 描述统计
    print("\n" + "-" * 70)
    print("描述统计")
    print("-" * 70)
    print(df_valid[['wealth', 'recent_profit_1', 'recent_profit_3', 'worked']].describe())
    
    # 检查变化
    wealth_std = df_valid['wealth'].std()
    profit1_std = df_valid['recent_profit_1'].std()
    profit3_std = df_valid['recent_profit_3'].std()
    
    print(f"\n变量标准差:")
    print(f"  wealth: {wealth_std:.6f}")
    print(f"  recent_profit_1: {profit1_std:.6f}")
    print(f"  recent_profit_3: {profit3_std:.6f}")
    
    if wealth_std < 1e-6 and profit1_std < 1e-6:
        print("\n⚠️  警告：财富和短期损益都没有变化！")
        print("可能原因：纯决策模拟，财富始终为0")
        return None
    
    # 回归1：labor ~ wealth + recent_profit_1 + skill
    print("\n" + "-" * 70)
    print("Logit回归1：P(work) ~ wealth + recent_profit_1 + skill")
    print("-" * 70)
    
    model1 = run_logit_regression(df_valid, profit_var='recent_profit_1')
    
    # 回归2：labor ~ wealth + recent_profit_3 + skill
    print("\n" + "-" * 70)
    print("Logit回归2：P(work) ~ wealth + recent_profit_3 + skill")
    print("-" * 70)
    
    model2 = run_logit_regression(df_valid, profit_var='recent_profit_3')
    
    return model1, model2

def run_logit_regression(df, profit_var='recent_profit_1'):
    """运行Logit回归"""
    X = df[['wealth', profit_var, 'skill']].copy()
    X = sm.add_constant(X)
    y = df['worked']
    
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        print(model.summary2().tables[1])
        
        # 关键结论
        beta_wealth = model.params['wealth']
        beta_profit = model.params[profit_var]
        
        pval_wealth = model.pvalues['wealth']
        pval_profit = model.pvalues[profit_var]
        
        print("\n" + "=" * 70)
        print("结论")
        print("=" * 70)
        print(f"长期财富系数 β₁: {beta_wealth:.6f} (p={pval_wealth:.4g})")
        print(f"短期损益系数 β₂: {beta_profit:.6f} (p={pval_profit:.4g})")
        
        # 判断心理账户效应
        wealth_sig = pval_wealth < 0.05
        profit_sig = pval_profit < 0.05
        
        print("\n心理账户假说检验:")
        print(f"  β₁ 显著？ {'是' if wealth_sig else '否'}")
        print(f"  β₂ 显著？ {'是' if profit_sig else '否'}")
        
        if not wealth_sig and profit_sig:
            print("\n✓ 支持心理账户假说：")
            print("  短期损益显著影响决策，而长期财富不显著")
        elif wealth_sig and profit_sig and abs(beta_profit) > abs(beta_wealth):
            print("\n✓ 部分支持心理账户假说：")
            print(f"  短期损益影响 (|{beta_profit:.4f}|) > 长期财富影响 (|{beta_wealth:.4f}|)")
        else:
            print("\n✗ 不支持心理账户假说")
        
        return model
    except Exception as e:
        print(f"✗ 回归失败: {e}")
        return None

def visualize_mental_accounting(df):
    """可视化心理账户效应"""
    df_valid = df[np.isfinite(df['skill'])].copy()
    
    if len(df_valid) == 0:
        return
    
    # 检查是否有变化
    if df_valid['recent_profit_1'].std() < 1e-6:
        print("\n⚠️  跳过可视化：数据无变化")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：短期损益 vs 工作率
    df_valid['profit_bin'] = pd.qcut(df_valid['recent_profit_1'], q=10, duplicates='drop')
    grouped_profit = df_valid.groupby('profit_bin').agg(
        profit_mean=('recent_profit_1', 'mean'),
        work_rate=('worked', 'mean')
    ).reset_index(drop=True)
    
    axes[0].plot(grouped_profit['profit_mean'], grouped_profit['work_rate'], 
                marker='o', linewidth=2)
    axes[0].set_xlabel('短期损益', fontsize=12)
    axes[0].set_ylabel('工作率', fontsize=12)
    axes[0].set_title('短期损益 vs 工作率', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 右图：长期财富 vs 工作率
    if df_valid['wealth'].std() > 1e-6:
        df_valid['wealth_bin'] = pd.qcut(df_valid['wealth'], q=10, duplicates='drop')
        grouped_wealth = df_valid.groupby('wealth_bin').agg(
            wealth_mean=('wealth', 'mean'),
            work_rate=('worked', 'mean')
        ).reset_index(drop=True)
        
        axes[1].plot(grouped_wealth['wealth_mean'], grouped_wealth['work_rate'], 
                    marker='o', linewidth=2, color='orange')
        axes[1].set_xlabel('长期财富', fontsize=12)
        axes[1].set_ylabel('工作率', fontsize=12)
        axes[1].set_title('长期财富 vs 工作率', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "财富无变化\n无法绘制", 
                    ha='center', va='center', fontsize=14, 
                    transform=axes[1].transAxes)
    
    plt.tight_layout()
    fig_path = os.path.join(OUT, "mental_accounting.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存图表: {fig_path}")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("心理账户测试（短期vs长期）")
    print("=" * 70)
    
    dense = load_data()
    df = extract_panel(dense)
    
    # 保存数据
    csv_path = os.path.join(OUT, "mental_accounting_panel.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存面板数据: {csv_path}")
    
    # 测试心理账户
    models = test_mental_accounting(df)
    
    # 可视化
    visualize_mental_accounting(df)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()



