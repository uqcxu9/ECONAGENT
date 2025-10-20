# -*- coding: utf-8 -*-
"""
单体回归实验 - 混合回归版本
- pw_i: Logit 回归（因为是 0/1 二元变量）
- pc_i: OLS 回归（因为是连续变量）
"""

import os
import pickle as pkl
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from scipy import stats

BASE = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent"
MODEL = "gpt-3-noperception-reflection-1-100agents-240months"
DATA = os.path.join(BASE, "data", MODEL)
OUT  = os.path.join(DATA, "result_analysis")
os.makedirs(OUT, exist_ok=True)

def load_dense_log(run_dir):
    """加载 dense_log"""
    d = Path(run_dir)
    
    for name in ["dense_log_240.pkl", "dense_log.pkl"]:
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            print(f"✓ 加载: {p}")
            with open(p, "rb") as f:
                return pkl.load(f)
    
    raise FileNotFoundError(f"找不到 dense_log: {run_dir}")

def extract_panel_data(dense, run_dir):
    """提取面板数据（使用离散动作）"""
    print("\n" + "=" * 70)
    print("提取回归面板数据（使用离散动作）")
    print("=" * 70)
    
    world = dense["world"]
    actions = dense["actions"]
    states = dense["states"]
    n_months = len(actions)
    
    print(f"总月数: {n_months}")
    
    # 读取 obs 文件
    print("\n步骤1: 读取税收数据...")
    obs_files = sorted(Path(run_dir).glob("obs_*.pkl"),
                      key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    
    obs_list = []
    for m in range(n_months):
        if m < len(obs_files):
            with open(obs_files[m], "rb") as f:
                obs_list.append(pkl.load(f))
        else:
            obs_list.append({})
    
    # 提取宏观变量
    print("\n步骤2: 提取宏观变量...")
    price_by_month, rate_by_month = [], []
    for m in range(n_months):
        w_m = world[m]
        price = w_m.get("Price", np.nan)
        rate = w_m.get("Interest Rate", np.nan)
        
        if price is None or np.isnan(price):
            st_m = states[m] if m < len(states) else {}
            planner = st_m.get('p', {}) if isinstance(st_m, dict) else {}
            price = planner.get('price', np.nan)
        
        if rate is None or rate == 0 or np.isnan(rate):
            st_m = states[m] if m < len(states) else {}
            planner = st_m.get('p', {}) if isinstance(st_m, dict) else {}
            rate = planner.get('interest_rate', np.nan)
        
        price_by_month.append(float(price) if price is not None else np.nan)
        rate_by_month.append(float(rate) if rate is not None else np.nan)
    
    price_series = pd.Series(price_by_month).ffill()
    rate_series = pd.Series(rate_by_month).replace(0, np.nan).ffill()
    
    # 提取 agent 数据
    print("\n步骤3: 提取agent数据...")
    rows = []
    
    for m in range(n_months):
        month_actions = actions[m]
        state_pre_all = states[m]
        month_obs = obs_list[m] if m < len(obs_list) else {}
        
        prev_state_all = states[m-1] if m > 0 else None
        prev_obs = obs_list[m-1] if m > 0 and (m-1) < len(obs_list) else None
        
        for agent_id in month_actions.keys():
            if str(agent_id) == "p":
                continue
            if not (isinstance(agent_id, (int, np.integer)) or str(agent_id).isdigit()):
                continue
            agent_id_int = int(agent_id)
            
            action = month_actions.get(agent_id, {}) or {}
            state_pre = state_pre_all.get(agent_id, {}) or {}
            
            # === 因变量：使用离散动作 ===
            pw_i = float(action.get("SimpleLabor", 0))  # 0/1
            pc_i = float(action.get("SimpleConsumption", 0)) / 50.0  # 0-1
            
            # === 自变量 ===
            endo = state_pre.get("endogenous", {}) or {}
            exp_wage = (state_pre.get("expected wage")
                       or endo.get("wage_expectation")
                       or endo.get("wage")
                       or state_pre.get("wage")
                       or world[m].get("Wage"))
            
            if exp_wage is None:
                exp_wage = state_pre.get("skill", 0.0)
            
            exp_wage = float(exp_wage) if exp_wage is not None else 0.0
            vi = exp_wage * 168.0
            
            inventory = state_pre.get("inventory", {}) or {}
            si = float(inventory.get("Coin", 0.0))
            
            P = float(price_series.iloc[m])
            r = float(rate_series.iloc[m])
            
            # === 滞后变量 ===
            ci_hat = 0.0
            if prev_state_all is not None and isinstance(prev_state_all, dict):
                prev_state = prev_state_all.get(agent_id, {}) or {}
                if isinstance(prev_state, dict):
                    cons_prev = prev_state.get('consumption', {})
                    if isinstance(cons_prev, dict):
                        ci_hat = float(cons_prev.get('Coin', 0.0))
            
            T_zi = 0.0
            zr = 0.0
            if prev_obs is not None and agent_id in prev_obs:
                prev_obs_agent = prev_obs[agent_id]
                if isinstance(prev_obs_agent, dict):
                    T_zi = float(prev_obs_agent.get("PeriodicBracketTax-tax_paid", 0.0))
                    zr = float(prev_obs_agent.get("PeriodicBracketTax-lump_sum", 0.0))
            
            rows.append({
                "month": m + 1,
                "agent": agent_id_int,
                "pw_i": pw_i,
                "pc_i": pc_i,
                "vi": vi,
                "ci_hat": ci_hat,
                "T_zi": T_zi,
                "zr": zr,
                "P": P,
                "si": si,
                "r": r
            })
        
        if (m + 1) % 60 == 0:
            print(f"  已处理 {m+1}/{n_months} 个月...")
    
    df = pd.DataFrame(rows).sort_values(["agent", "month"]).reset_index(drop=True)
    print(f"\n✓ 提取完成: {len(df)} 行，{df['agent'].nunique()} 个agent")
    
    print("\n数据统计:")
    print(f"  pw_i: 均值={df['pw_i'].mean():.3f}, 标准差={df['pw_i'].std():.3f}")
    print(f"  pc_i: 均值={df['pc_i'].mean():.3f}, 标准差={df['pc_i'].std():.3f}")
    print(f"  pw_i 唯一值: {df['pw_i'].unique()}")
    
    return df

def perform_regression_mixed(df):
    """
    混合回归：
    - pw_i 用 Logit（二元变量）
    - pc_i 用 OLS（连续变量）
    """
    print("\n" + "=" * 70)
    print("混合回归分析")
    print("  pw_i: Logit 回归（二元变量）")
    print("  pc_i: OLS 回归（连续变量）")
    print("=" * 70)
    
    independent_vars = ["vi", "ci_hat", "T_zi", "zr", "P", "si", "r"]
    
    results = {
        "pw_i": {var: 0 for var in independent_vars},
        "pc_i": {var: 0 for var in independent_vars}
    }
    
    detailed_results = []
    n_agents = df["agent"].nunique()
    processed = 0
    
    for agent_id, agent_df in df.groupby("agent"):
        if len(agent_df) < 10:
            continue
        
        processed += 1
        if processed % 20 == 0:
            print(f"  已处理 {processed}/{n_agents} 个agent...")
        
        agent_result = {"agent_id": agent_id, "n_obs": len(agent_df)}
        
        # === pw_i: Logit 回归 ===
        y_work = agent_df['pw_i'].values
        X = agent_df[independent_vars].values
        
        # Z-score标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_normalized = (X - X_mean) / X_std
        X_normalized = sm.add_constant(X_normalized)
        
        try:
            # 检查 y 是否有变化
            if y_work.std() > 1e-10 and len(np.unique(y_work)) > 1:
                model_logit = sm.Logit(y_work, X_normalized).fit(disp=0, maxiter=100)
                
                for i, var in enumerate(independent_vars):
                    pval = float(model_logit.pvalues[i + 1])
                    coef = float(model_logit.params[i + 1])
                    
                    if pval <= 0.05:
                        results['pw_i'][var] += 1
                    
                    agent_result[f"pw_i_{var}_coef"] = coef
                    agent_result[f"pw_i_{var}_pval"] = pval
            else:
                for var in independent_vars:
                    agent_result[f"pw_i_{var}_coef"] = np.nan
                    agent_result[f"pw_i_{var}_pval"] = np.nan
        except Exception as e:
            for var in independent_vars:
                agent_result[f"pw_i_{var}_coef"] = np.nan
                agent_result[f"pw_i_{var}_pval"] = np.nan
        
        # === pc_i: OLS 回归 ===
        y_cons = agent_df['pc_i'].values
        
        try:
            if y_cons.std() > 1e-10:
                model_ols = sm.OLS(y_cons, X_normalized).fit()
                
                for i, var in enumerate(independent_vars):
                    pval = float(model_ols.pvalues[i + 1])
                    coef = float(model_ols.params[i + 1])
                    
                    if pval <= 0.05:
                        results['pc_i'][var] += 1
                    
                    agent_result[f"pc_i_{var}_coef"] = coef
                    agent_result[f"pc_i_{var}_pval"] = pval
            else:
                for var in independent_vars:
                    agent_result[f"pc_i_{var}_coef"] = np.nan
                    agent_result[f"pc_i_{var}_pval"] = np.nan
        except Exception as e:
            for var in independent_vars:
                agent_result[f"pc_i_{var}_coef"] = np.nan
                agent_result[f"pc_i_{var}_pval"] = np.nan
        
        detailed_results.append(agent_result)
    
    print(f"\n✓ 回归完成: {processed} 个agent")
    
    return results, pd.DataFrame(detailed_results)

def print_and_save_results(results, detailed_df):
    """打印和保存结果"""
    print("\n" + "=" * 70)
    print("回归结果汇总（混合回归）")
    print("=" * 70)
    
    print("\n方法说明:")
    print("  pw_i: Logit 回归（因为是 0/1 二元变量）")
    print("  pc_i: OLS 回归（因为是 0-1 连续变量）")
    print("  ⚠️  由于数据限制（只有离散动作），结果可能不如论文")
    
    print("\n[Table 1] 显著影响agent决策的变量数量 (p ≤ 0.05):")
    print("-" * 70)
    print(f"{'变量':<12} | {'pw_i (Logit)':<15} | {'pc_i (OLS)':<15}")
    print("-" * 70)
    
    for var in ["vi", "ci_hat", "T_zi", "zr", "P", "si", "r"]:
        pw_count = results["pw_i"][var]
        pc_count = results["pc_i"][var]
        print(f"{var:<12} | {pw_count:<15} | {pc_count:<15}")
    
    print("\n论文 Table 1 参考值:")
    print("  pw_i: vi=60, ci_hat=37, T(zi)=60, zr=65, P=58, si=56, r=31")
    print("  pc_i: vi=65, ci_hat=73, T(zi)=51, zr=52, P=62, si=100, r=49")
    
    csv_path = os.path.join(OUT, "regression_detailed_results_v4_mixed.csv")
    detailed_df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存详细结果: {csv_path}")
    
    summary_df = pd.DataFrame(results).T
    summary_csv = os.path.join(OUT, "table1_regression_summary_v4_mixed.csv")
    summary_df.to_csv(summary_csv)
    print(f"✓ 保存 Table 1: {summary_csv}")
    
    return summary_df

def main():
    print("\n" + "=" * 70)
    print("单体回归分析 - 混合回归版本")
    print("Table 1: Decision Rationality")
    print("=" * 70)
    
    # 加载数据
    dense = load_dense_log(DATA)
    
    # 提取面板数据
    df = extract_panel_data(dense, DATA)
    
    # 保存面板数据
    panel_csv = os.path.join(OUT, "panel_data_v4_mixed.csv")
    df.to_csv(panel_csv, index=False)
    print(f"\n✓ 保存面板数据: {panel_csv}")
    
    # 混合回归
    results, detailed_df = perform_regression_mixed(df)
    
    # 结果展示和保存
    summary_df = print_and_save_results(results, detailed_df)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print("\n⚠️  注意:")
    print("  由于缺少完整的 LLM 倾向值数据，")
    print("  我们使用了离散动作 (0/1) 作为近似。")
    print("  结果可能与论文有较大差距。")

if __name__ == "__main__":
    main()



