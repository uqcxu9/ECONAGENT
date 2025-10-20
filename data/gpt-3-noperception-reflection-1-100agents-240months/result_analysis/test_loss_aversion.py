# -*- coding: utf-8 -*-
"""
测试2：损失厌恶

假设：Agent对财富损失的反应比对等量增益更强烈
检验方法：
  1. 计算财富变化 Δwealth_t = wealth_t - wealth_{t-1}
  2. 计算消费变化 Δconsumption_t = consumption_t - consumption_{t-1}
  3. 回归：Δconsumption ~ α + β₊·max(Δwealth,0) + β₋·min(Δwealth,0)
  4. 检验：|β₋| > |β₊| (损失的边际效应 > 收益的边际效应)
"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
import re
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

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

def extract_changes(dense):
    """提取财富和消费的变化"""
    states = dense["states"]
    actions = dense["actions"]
    n_months = len(actions)
    
    # 先提取所有agent的时间序列
    agent_series = {}
    
    for m in range(n_months + 1):  # states有n_months+1个
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
                    'consumption': []
                }
            
            s = state_t.get(agent_id_int, {}) or {}
            
            # 财富
            wealth = float(((s.get('inventory', {}) or {}).get('Coin', 0.0)) or 0.0)
            
            # 消费决策（从actions中提取，只有前n_months个月有）
            if m < n_months:
                act = actions[m].get(agent_id_int, {}) or {}
                consumption = float(act.get('SimpleConsumption', 0.0))
            else:
                consumption = 0.0
            
            agent_series[agent_id_int]['month'].append(m)
            agent_series[agent_id_int]['wealth'].append(wealth)
            agent_series[agent_id_int]['consumption'].append(consumption)
    
    # 计算变化量
    rows = []
    for agent_id, data in agent_series.items():
        wealth_arr = np.array(data['wealth'])
        cons_arr = np.array(data['consumption'])
        
        # 计算变化（从第2期开始）
        for t in range(1, len(wealth_arr) - 1):  # -1因为消费序列少一个
            delta_wealth = wealth_arr[t] - wealth_arr[t-1]
            delta_cons = cons_arr[t] - cons_arr[t-1]
            
            # 分解为损失和收益
            wealth_gain = max(delta_wealth, 0)
            wealth_loss = min(delta_wealth, 0)
            
            rows.append({
                'agent': agent_id,
                'month': t + 1,
                'wealth_t': wealth_arr[t],
                'wealth_t1': wealth_arr[t-1],
                'delta_wealth': delta_wealth,
                'wealth_gain': wealth_gain,
                'wealth_loss': wealth_loss,
                'consumption_t': cons_arr[t],
                'consumption_t1': cons_arr[t-1],
                'delta_cons': delta_cons
            })
    
    df = pd.DataFrame(rows)
    return df

def test_loss_aversion(df):
    """测试损失厌恶"""
    print("=" * 70)
    print("测试2：损失厌恶")
    print("=" * 70)
    
    print(f"\n总样本数: {len(df)}")
    
    # 过滤有变化的样本
    df_change = df[
        (df['delta_wealth'] != 0) | (df['delta_cons'] != 0)
    ].copy()
    
    print(f"有变化的样本: {len(df_change)}")
    
    if len(df_change) == 0:
        print("\n⚠️  警告：没有财富或消费变化的样本！")
        print("可能原因：财富和消费都始终为0（纯决策模拟）")
        return None
    
    # 描述统计
    print("\n" + "-" * 70)
    print("描述统计")
    print("-" * 70)
    
    print(f"财富增加样本数: {(df_change['wealth_gain'] > 0).sum()}")
    print(f"财富减少样本数: {(df_change['wealth_loss'] < 0).sum()}")
    print(f"平均 Δwealth: {df_change['delta_wealth'].mean():.4f}")
    print(f"平均 Δconsumption: {df_change['delta_cons'].mean():.4f}")
    
    # 分组比较
    gain_group = df_change[df_change['wealth_gain'] > 0]
    loss_group = df_change[df_change['wealth_loss'] < 0]
    
    if len(gain_group) > 0 and len(loss_group) > 0:
        print("\n按财富变化方向分组:")
        print(f"  财富增加时，平均 Δconsumption: {gain_group['delta_cons'].mean():.4f}")
        print(f"  财富减少时，平均 Δconsumption: {loss_group['delta_cons'].mean():.4f}")
        
        # 计算弹性
        gain_elasticity = gain_group['delta_cons'].mean() / gain_group['wealth_gain'].mean() if gain_group['wealth_gain'].mean() != 0 else 0
        loss_elasticity = loss_group['delta_cons'].mean() / loss_group['wealth_loss'].mean() if loss_group['wealth_loss'].mean() != 0 else 0
        
        print(f"\n  收益弹性: {gain_elasticity:.4f}")
        print(f"  损失弹性: {loss_elasticity:.4f}")
        print(f"  |损失弹性| / |收益弹性|: {abs(loss_elasticity) / abs(gain_elasticity) if abs(gain_elasticity) > 1e-6 else np.inf:.4f}")
    
    # 回归：Δcons ~ wealth_gain + wealth_loss
    print("\n" + "-" * 70)
    print("回归：Δconsumption ~ wealth_gain + wealth_loss")
    print("-" * 70)
    
    df_reg = df_change.copy()
    
    if len(df_reg) > 100:
        X = df_reg[['wealth_gain', 'wealth_loss']].copy()
        X = sm.add_constant(X)
        y = df_reg['delta_cons']
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HC1')
            print(model.summary2().tables[1])
            
            # 关键结论
            beta_gain = model.params['wealth_gain']
            beta_loss = model.params['wealth_loss']
            
            pval_gain = model.pvalues['wealth_gain']
            pval_loss = model.pvalues['wealth_loss']
            
            print("\n" + "=" * 70)
            print("结论")
            print("=" * 70)
            print(f"收益系数 β₊: {beta_gain:.6f} (p={pval_gain:.4g})")
            print(f"损失系数 β₋: {beta_loss:.6f} (p={pval_loss:.4g})")
            print(f"|β₋| / |β₊|: {abs(beta_loss) / abs(beta_gain) if abs(beta_gain) > 1e-6 else np.inf:.4f}")
            
            if abs(beta_loss) > abs(beta_gain) and pval_loss < 0.05:
                print("\n✓ 检测到损失厌恶：")
                print(f"  财富损失的边际效应 ({abs(beta_loss):.4f}) > 财富收益的边际效应 ({abs(beta_gain):.4f})")
            else:
                print("\n✗ 未检测到显著的损失厌恶")
            
            return model
        except Exception as e:
            print(f"✗ 回归失败: {e}")
            return None
    else:
        print("✗ 有效样本不足，无法进行回归")
        return None

def visualize_loss_aversion(df):
    """可视化损失厌恶"""
    df_change = df[
        (df['delta_wealth'] != 0) | (df['delta_cons'] != 0)
    ].copy()
    
    if len(df_change) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：散点图（Δwealth vs Δconsumption）
    axes[0].scatter(df_change['delta_wealth'], df_change['delta_cons'], 
                   alpha=0.3, s=10)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Δ财富', fontsize=12)
    axes[0].set_ylabel('Δ消费', fontsize=12)
    axes[0].set_title('财富变化 vs 消费变化', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 右图：分组箱线图
    gain_group = df_change[df_change['wealth_gain'] > 0]['delta_cons']
    loss_group = df_change[df_change['wealth_loss'] < 0]['delta_cons']
    
    if len(gain_group) > 0 and len(loss_group) > 0:
        axes[1].boxplot([gain_group, loss_group], 
                       labels=['财富增加时', '财富减少时'])
        axes[1].set_ylabel('Δ消费', fontsize=12)
        axes[1].set_title('消费变化分布（按财富变化方向）', fontsize=14, fontweight='bold')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(OUT, "loss_aversion.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存图表: {fig_path}")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("损失厌恶测试")
    print("=" * 70)
    
    dense = load_data()
    df = extract_changes(dense)
    
    # 保存数据
    csv_path = os.path.join(OUT, "loss_aversion_panel.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存面板数据: {csv_path}")
    
    # 测试损失厌恶
    model = test_loss_aversion(df)
    
    # 可视化
    visualize_loss_aversion(df)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()



