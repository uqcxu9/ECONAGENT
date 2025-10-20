# -*- coding: utf-8 -*-
"""
单体回归实验 - 从 dialog 提取倾向值版本
Table 1: Decision Rationality
p_w_i, p_c_i ~ v_i + c_hat_i + T(z_i) + z_r + P + s_i + r
"""

import os
import pickle as pkl
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

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

def load_dialogs(run_dir):
    """加载 dialog"""
    dialog_path = os.path.join(run_dir, "dialog_240.pkl")
    if not os.path.exists(dialog_path):
        dialog_path = os.path.join(run_dir, "dialog.pkl")
    
    print(f"✓ 加载 dialog: {dialog_path}")
    with open(dialog_path, "rb") as f:
        return pkl.load(f)

def parse_llm_propensity(dialog_msg):
    """从 dialog 消息中解析 LLM 倾向值"""
    if not isinstance(dialog_msg, dict):
        return None, None
    
    if dialog_msg.get('role') != 'assistant':
        return None, None
    
    content = dialog_msg.get('content', '')
    if not content:
        return None, None
    
    # 尝试直接解析 JSON
    try:
        data = json.loads(content)
        work = data.get('work', data.get('Work'))
        consumption = data.get('consumption', data.get('Consumption'))
        return float(work) if work is not None else None, \
               float(consumption) if consumption is not None else None
    except:
        pass
    
    # 尝试用正则提取
    work_match = re.search(r'"[Ww]ork"\s*:\s*([\d.]+)', content)
    cons_match = re.search(r'"[Cc]onsumption"\s*:\s*([\d.]+)', content)
    
    work = float(work_match.group(1)) if work_match else None
    consumption = float(cons_match.group(1)) if cons_match else None
    
    return work, consumption

def extract_propensities_from_dialogs(dialogs, n_agents, n_months):
    """从 dialogs 提取所有 agent 的倾向值"""
    print("\n" + "=" * 70)
    print("从 dialog 提取倾向值")
    print("=" * 70)
    
    # propensities[month][agent_id] = (work, consumption)
    propensities = {}
    
    for agent_id in range(n_agents):
        agent_dialog = list(dialogs[agent_id])
        
        # 找出所有 assistant 的回复
        assistant_msgs = [msg for msg in agent_dialog if msg.get('role') == 'assistant']
        
        for month, msg in enumerate(assistant_msgs[:n_months]):
            work, consumption = parse_llm_propensity(msg)
            
            if month not in propensities:
                propensities[month] = {}
            
            propensities[month][agent_id] = {
                'work': work if work is not None else 0.0,
                'consumption': consumption if consumption is not None else 0.0
            }
    
    print(f"✓ 提取完成: {n_months} 个月 × {n_agents} 个agent")
    
    # 统计缺失率
    total = n_months * n_agents
    missing_work = sum(1 for m in propensities.values() 
                      for a in m.values() if a['work'] == 0.0)
    missing_cons = sum(1 for m in propensities.values() 
                      for a in m.values() if a['consumption'] == 0.0)
    
    print(f"  work 缺失率: {missing_work/total*100:.2f}%")
    print(f"  consumption 缺失率: {missing_cons/total*100:.2f}%")
    
    return propensities

def extract_panel_data_with_dialog(dense, run_dir, propensities):
    """提取面板数据（使用 dialog 中的倾向值）"""
    print("\n" + "=" * 70)
    print("提取回归面板数据（使用 dialog 倾向值）")
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
    print("\n步骤3: 提取agent数据（使用 dialog 倾向值）...")
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
            
            state_pre = state_pre_all.get(agent_id, {}) or {}
            
            # === 从 dialog 获取倾向值 ===
            if m in propensities and agent_id_int in propensities[m]:
                pw_i = propensities[m][agent_id_int]['work']
                pc_i = propensities[m][agent_id_int]['consumption']
            else:
                # 兜底：使用离散动作
                action = month_actions.get(agent_id, {}) or {}
                pw_i = float(action.get("SimpleLabor", 0))
                pc_i = float(action.get("SimpleConsumption", 0)) / 50.0
            
            pw_i = np.clip(pw_i, 0.0, 1.0)
            pc_i = np.clip(pc_i, 0.0, 1.0)
            
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
    print(f"  pw_i 唯一值数量: {df['pw_i'].nunique()}")
    print(f"  pc_i 唯一值数量: {df['pc_i'].nunique()}")
    
    return df

def perform_regression_ols(df):
    """OLS回归"""
    print("\n" + "=" * 70)
    print("OLS 回归分析")
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
        
        for target_var in ["pw_i", "pc_i"]:
            y = agent_df[target_var].values
            X = agent_df[independent_vars].values
            
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8
            X_normalized = (X - X_mean) / X_std
            X_normalized = sm.add_constant(X_normalized)
            
            try:
                model = sm.OLS(y, X_normalized).fit()
                
                for i, var in enumerate(independent_vars):
                    pval = float(model.pvalues[i + 1])
                    coef = float(model.params[i + 1])
                    
                    if pval <= 0.05:
                        results[target_var][var] += 1
                    
                    agent_result[f"{target_var}_{var}_coef"] = coef
                    agent_result[f"{target_var}_{var}_pval"] = pval
            
            except Exception as e:
                for var in independent_vars:
                    agent_result[f"{target_var}_{var}_coef"] = np.nan
                    agent_result[f"{target_var}_{var}_pval"] = np.nan
        
        detailed_results.append(agent_result)
    
    print(f"\n✓ OLS回归完成: {processed} 个agent")
    
    return results, pd.DataFrame(detailed_results)

def print_and_save_results(results_ols, detailed_df):
    """打印和保存结果"""
    print("\n" + "=" * 70)
    print("回归结果汇总（使用 dialog 倾向值）")
    print("=" * 70)
    
    print("\n[Table 1] 显著影响agent决策的变量数量 (p ≤ 0.05, OLS):")
    print("-" * 70)
    print(f"{'变量':<12} | {'pw_i (工作)':<15} | {'pc_i (消费)':<15}")
    print("-" * 70)
    
    for var in ["vi", "ci_hat", "T_zi", "zr", "P", "si", "r"]:
        pw_count = results_ols["pw_i"][var]
        pc_count = results_ols["pc_i"][var]
        print(f"{var:<12} | {pw_count:<15} | {pc_count:<15}")
    
    print("\n论文 Table 1 参考值:")
    print("  pw_i: vi=60, ci_hat=37, T(zi)=60, zr=65, P=58, si=56, r=31")
    print("  pc_i: vi=65, ci_hat=73, T(zi)=51, zr=52, P=62, si=100, r=49")
    
    csv_path = os.path.join(OUT, "regression_detailed_results_v3_dialog.csv")
    detailed_df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存详细结果: {csv_path}")
    
    summary_df = pd.DataFrame(results_ols).T
    summary_csv = os.path.join(OUT, "table1_regression_summary_v3_dialog.csv")
    summary_df.to_csv(summary_csv)
    print(f"✓ 保存 Table 1: {summary_csv}")
    
    return summary_df

def main():
    print("\n" + "=" * 70)
    print("单体回归分析 - 使用 dialog 倾向值")
    print("Table 1: Decision Rationality")
    print("=" * 70)
    
    # 加载数据
    dense = load_dense_log(DATA)
    dialogs = load_dialogs(DATA)
    
    # 从 dialog 提取倾向值
    n_months = len(dense["actions"])
    n_agents = 100
    propensities = extract_propensities_from_dialogs(dialogs, n_agents, n_months)
    
    # 提取面板数据
    df = extract_panel_data_with_dialog(dense, DATA, propensities)
    
    # 保存面板数据
    panel_csv = os.path.join(OUT, "panel_data_v3_dialog.csv")
    df.to_csv(panel_csv, index=False)
    print(f"\n✓ 保存面板数据: {panel_csv}")
    
    # OLS回归
    results_ols, detailed_df = perform_regression_ols(df)
    
    # 结果展示和保存
    summary_df = print_and_save_results(results_ols, detailed_df)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()



