# -*- coding: utf-8 -*-
"""
测试1：EMA（期望相对收益）与工作倾向

假设：EMA < 0 的agent更倾向于不工作
检验：
  1. 计算每个agent每月的 EMA
  2. 按 EMA 正负分组，比较工作率
  3. Logit回归：P(work) ~ EMA + controls
"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.stats import ttest_ind
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
    
    obs_files = sorted(Path(DATA).glob("obs_*.pkl"),
                      key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    obs_list = []
    for p in obs_files:
        with open(p, "rb") as f:
            obs_list.append(pkl.load(f))
    
    world = dense.get("world", [])
    
    return dense, obs_list, world

def extract_panel(dense, obs_list, world):
    """提取面板数据：agent-month级别"""
    states = dense["states"]
    actions = dense["actions"]
    n_months = len(actions)
    
    rows = []
    for m in range(n_months):
        state_t0 = states[m]
        state_t1 = states[m + 1]
        action_t = actions[m]
        obs_t = obs_list[m] if m < len(obs_list) else {}
        world_t = world[m] if m < len(world) else {}
        
        for agent_id in action_t.keys():
            if str(agent_id) == 'p':
                continue
            if not (isinstance(agent_id, (int, np.integer)) or str(agent_id).isdigit()):
                continue
            
            agent_id_int = int(agent_id)
            
            s0 = state_t0.get(agent_id_int, {}) or {}
            s1 = state_t1.get(agent_id_int, {}) or {}
            act = action_t.get(agent_id_int, {}) or {}
            obs_agent = obs_t.get(agent_id_int, {}) or {}
            
            # 工作决策 (0/1)
            labor_action = int(act.get('SimpleLabor', 0))
            worked = 1 if labor_action == 1 else 0
            
            # EMA (期望相对收益)
            ema_val = float(obs_agent.get('PeriodicBracketTax-ema', np.nan) or np.nan)
            
            # 技能
            skill = float(((s0.get('endogenous', {}) or {}).get('skill', np.nan)) or np.nan)
            
            # 财富
            wealth = float(((s0.get('inventory', {}) or {}).get('Coin', 0.0)) or 0.0)
            
            rows.append({
                'month': m + 1,
                'agent': agent_id_int,
                'worked': worked,
                'ema': ema_val,
                'skill': skill,
                'wealth': wealth
            })
    
    df = pd.DataFrame(rows)
    return df

def test_ema_effect(df):
    """测试EMA对工作倾向的影响"""
    print("=" * 70)
    print("测试1：EMA与工作倾向")
    print("=" * 70)
    
    # 过滤有效EMA数据
    df_valid = df[np.isfinite(df['ema'])].copy()
    
    print(f"\n总样本数: {len(df)}")
    print(f"有效EMA样本: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("\n⚠️  警告：没有有效的EMA数据！")
        print("可能原因：obs文件中没有记录'PeriodicBracketTax-ema'")
        return None
    
    # 按EMA正负分组
    df_valid['ema_negative'] = (df_valid['ema'] < 0).astype(int)
    
    neg_group = df_valid[df_valid['ema_negative'] == 1]
    pos_group = df_valid[df_valid['ema_negative'] == 0]
    
    print(f"\nEMA < 0 样本数: {len(neg_group)}")
    print(f"EMA ≥ 0 样本数: {len(pos_group)}")
    
    # 描述统计
    print("\n" + "-" * 70)
    print("描述统计：工作率（按EMA正负分组）")
    print("-" * 70)
    
    work_rate_neg = neg_group['worked'].mean()
    work_rate_pos = pos_group['worked'].mean()
    
    print(f"EMA < 0 组工作率: {work_rate_neg:.4f}")
    print(f"EMA ≥ 0 组工作率: {work_rate_pos:.4f}")
    print(f"差异: {work_rate_neg - work_rate_pos:.4f}")
    
    # t检验
    if len(neg_group) > 0 and len(pos_group) > 0:
        t_stat, p_val = ttest_ind(neg_group['worked'], pos_group['worked'])
        print(f"\nt检验: t={t_stat:.4f}, p={p_val:.4g}")
    
    # Logit回归：P(work) ~ EMA + skill + wealth
    print("\n" + "-" * 70)
    print("Logit回归：P(work) ~ EMA + skill + wealth")
    print("-" * 70)
    
    df_reg = df_valid[np.isfinite(df_valid['skill'])].copy()
    
    if len(df_reg) > 100:
        X = df_reg[['ema', 'skill', 'wealth']].copy()
        X = sm.add_constant(X)
        y = df_reg['worked']
        
        try:
            model = sm.Logit(y, X).fit(disp=0, maxiter=100)
            print(model.summary2().tables[1])
            
            # 关键结论
            ema_coef = model.params['ema']
            ema_pval = model.pvalues['ema']
            
            print("\n" + "=" * 70)
            print("结论")
            print("=" * 70)
            print(f"EMA系数: {ema_coef:.6f}")
            print(f"p值: {ema_pval:.4g}")
            
            if ema_coef > 0 and ema_pval < 0.05:
                print("✓ EMA显著正向影响工作倾向（EMA越高，越倾向工作）")
            elif ema_coef < 0 and ema_pval < 0.05:
                print("✓ EMA显著负向影响工作倾向（EMA越低，越倾向不工作）")
            else:
                print("✗ EMA对工作倾向无显著影响")
            
            return model
        except Exception as e:
            print(f"✗ 回归失败: {e}")
            return None
    else:
        print("✗ 有效样本不足，无法进行回归")
        return None

def visualize_ema_effect(df):
    """可视化EMA与工作率的关系"""
    df_valid = df[np.isfinite(df['ema'])].copy()
    
    if len(df_valid) == 0:
        return
    
    # 按EMA分组，计算工作率
    df_valid['ema_bin'] = pd.qcut(df_valid['ema'], q=20, duplicates='drop')
    grouped = df_valid.groupby('ema_bin').agg(
        ema_mean=('ema', 'mean'),
        work_rate=('worked', 'mean'),
        count=('worked', 'count')
    ).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(grouped['ema_mean'], grouped['work_rate'], 
              s=grouped['count']/10, alpha=0.6)
    ax.set_xlabel('EMA (期望相对收益)', fontsize=12)
    ax.set_ylabel('工作率', fontsize=12)
    ax.set_title('EMA与工作率关系', fontsize=14, fontweight='bold')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='EMA=0')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUT, "ema_work_propensity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存图表: {fig_path}")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("EMA与工作倾向测试")
    print("=" * 70)
    
    dense, obs_list, world = load_data()
    df = extract_panel(dense, obs_list, world)
    
    # 保存面板数据
    csv_path = os.path.join(OUT, "ema_work_panel.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存面板数据: {csv_path}")
    
    # 测试EMA效应
    model = test_ema_effect(df)
    
    # 可视化
    visualize_ema_effect(df)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()



