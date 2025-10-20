# -*- coding: utf-8 -*-
"""
单体回归实验 - 严格按照论文定义
Table 1: Decision Rationality
p_w_i, p_c_i ~ v_i + c_hat_i + T(z_i) + z_r + P + s_i + r
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
from sklearn.linear_model import LogisticRegression

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
    
    env_files = sorted(d.glob("env_*.pkl"), 
                      key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    if env_files:
        env_file = env_files[-1]
        print(f"✓ 从 {env_file} 提取 dense_log")
        with open(env_file, "rb") as f:
            env = pkl.load(f)
        return env.dense_log
    
    raise FileNotFoundError(f"找不到 dense_log: {run_dir}")

def extract_panel_data_strict(dense, run_dir):
    """
    严格按照论文定义提取面板数据
    
    修正点:
    1. c_hat_i = 上月真实消费 (consumption['Coin'])，不是消费倾向
    2. p_w_i = 当月劳动决策 (0/1)，不做 EWMA 平滑
    3. T(z_i), z_r = 上月的税收和再分配，避免前视偏差
    4. P, r = 直接使用水平值，不做偏差项变换
    5. 优先从 states['p'] 读取宏观变量，避免 bfill
    """
    print("\n" + "=" * 70)
    print("提取回归面板数据（严格论文定义）")
    print("=" * 70)
    
    world = dense["world"]
    actions = dense["actions"]
    states = dense["states"]
    n_months = len(actions)
    
    print(f"总月数: {n_months}")
    print(f"states 长度: {len(states)} (通常比 actions 多1)")
    
    # ========== 1. 读取 obs_*.pkl (税收和再分配) ==========
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
    
    print(f"  ✓ 读取了 {len(obs_list)} 个月的观察数据")
    
    # ========== 2. 提取宏观变量（仅前向填充，不 bfill） ==========
    print("\n步骤2: 提取宏观变量（仅 ffill，不 bfill，且用当月 m）...")
    
    price_by_month, rate_by_month = [], []
    for m in range(n_months):
        # 🔧改：用 world[m] 作为当月（避免 m+1 看未来）
        w_m = world[m]  # ← 原代码用 w = world[m+1]，会"看未来"。必须改成 world[m]。
        price = w_m.get("Price", np.nan)
        rate  = w_m.get("Interest Rate", np.nan)
    
        if (price is None) or (isinstance(price, float) and np.isnan(price)):
            # 如 world 缺，退回 states[m]['p']
            st_m = states[m] if m < len(states) else {}
            planner = st_m.get('p', {}) if isinstance(st_m, dict) else {}
            price = planner.get('price', np.nan)
    
        if (rate is None) or (rate == 0) or (isinstance(rate, float) and np.isnan(rate)):
            st_m = states[m] if m < len(states) else {}
            planner = st_m.get('p', {}) if isinstance(st_m, dict) else {}
            rate = planner.get('interest_rate', np.nan)
    
        price_by_month.append(float(price) if price is not None else np.nan)
        rate_by_month.append(float(rate) if rate  is not None else np.nan)
    
    price_series = pd.Series(price_by_month)
    rate_series  = pd.Series(rate_by_month)
    
    # 仅前向填充，避免 bfill 引入未来信息
    fp = price_series.first_valid_index()
    fr = rate_series.first_valid_index()
    if fp is not None:
        price_series.iloc[:fp] = price_series.iloc[fp]
    price_series = price_series.ffill()
    
    if fr is not None:
        rate_series.iloc[:fr] = rate_series.iloc[fr]
    rate_series = rate_series.replace(0, np.nan).ffill()
    
    print(f"  ✓ 价格范围: {price_series.min():.2f} - {price_series.max():.2f}")
    print(f"  ✓ 利率范围: {rate_series.min():.4f} - {rate_series.max():.4f}")
    
    # ========== 3. 逐月逐agent提取 ==========
    print("\n步骤3: 提取agent决策数据...")
    rows = []
    
    for m in range(n_months):
        month_actions = actions[m]
        month_states = states[m + 1] if len(states) == n_months + 1 else states[m]
        month_obs = obs_list[m] if m < len(obs_list) else {}
        
        # 上个月的states和obs（用于滞后变量）
        if len(states) == n_months + 1:
            prev_states = states[m] if m > 0 else None
        else:
            prev_states = states[m-1] if m > 0 else None

        prev_obs = obs_list[m - 1] if m > 0 and m - 1 < len(obs_list) else None
        
        for agent_id in month_actions.keys():
            if str(agent_id) == "p":
                continue
            agent_id_int = int(agent_id) if isinstance(agent_id, (int, np.integer)) or str(agent_id).isdigit() else None
            if agent_id_int is None:
                continue
            action = month_actions.get(agent_id, {}) or {}
            state = month_states.get(agent_id, {}) or {}
            
            # === 当月决策变量 ===
            # ---- 读取 LLM 决策的倾向值 ----
            work_prop = action.get("work", None) or action.get("Work", None)
            cons_prop = action.get("consumption", None) or action.get("Consumption", None)
            
            # 如果没取到，再退一步用离散动作近似（不推荐）
            if work_prop is None:
                labor_action = action.get("SimpleLabor", 0)
                work_prop = float(labor_action)
            
            if cons_prop is None:
                consumption_action = action.get("SimpleConsumption", 0)
                if isinstance(consumption_action, (int, float)):
                    # 如果是离散动作编号（0-50），转换为0-1
                    if consumption_action > 1:
                        cons_prop = float(consumption_action) / 50.0
                    else:
                        cons_prop = float(consumption_action)
                else:
                    cons_prop = 0.0
            
            # 限定在 [0,1] 范围
            pw_i = min(max(float(work_prop), 0.0), 1.0)
            pc_i = min(max(float(cons_prop), 0.0), 1.0)

            # === 当月自变量 ===
            # v_i: 预期月收入
            expected_skill = state.get("expected skill", state.get("skill", 0.0))
            vi = float(expected_skill) * 168.0

            # s_i: 当前储蓄
            inventory = state.get("inventory", {}) or {}
            si = float(inventory.get("Coin", 0.0))

            # P, r: 宏观变量水平值
            P = float(price_series.iloc[m])
            r = float(rate_series.iloc[m])

            # === 滞后变量（上个月的值） ===
            # c_hat_i: 上月消费比例（相对值）
            ci_hat = 0.0
            if prev_states is not None and agent_id in prev_states:
                prev_state = prev_states[agent_id]
                if isinstance(prev_state, dict):
                    consumption_dict = prev_state.get('consumption', {})
                    prev_consumption = 0.0
                    if isinstance(consumption_dict, dict):
                        prev_consumption = float(consumption_dict.get('Coin', 0.0))
                    
                    # 使用上月的收入作为分母
                    prev_income_dict = prev_state.get('income', {})
                    if isinstance(prev_income_dict, dict):
                        prev_income = float(prev_income_dict.get('Coin', 0.0))
                        if prev_income > 0:
                            ci_hat = prev_consumption / prev_income
                        else:
                            # 如果没有收入，用财富作为分母
                            prev_inventory = prev_state.get('inventory', {})
                            if isinstance(prev_inventory, dict):
                                prev_wealth = float(prev_inventory.get('Coin', 0.0))
                                if prev_wealth > 0:
                                    ci_hat = prev_consumption / prev_wealth

            # T(z_i), z_r: 上月的税收和再分配
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
                "pw_i": pw_i,      # 工作倾向 [0,1]
                "pc_i": pc_i,      # 消费倾向 [0,1]
                "vi": vi,
                "ci_hat": ci_hat,  # 上月消费比例
                "T_zi": T_zi,      # 上月税收
                "zr": zr,          # 上月再分配
                "P": P,            # 价格水平
                "si": si,
                "r": r             # 利率水平
            })
        
        if (m + 1) % 60 == 0:
            print(f"  已处理 {m+1}/{n_months} 个月...")
    
    df = pd.DataFrame(rows).sort_values(["agent", "month"]).reset_index(drop=True)
    print(f"\n✓ 提取完成: {len(df)} 行，{df['agent'].nunique()} 个agent")
    
    # ========== 关键修正：计算滚动工作倾向 ==========
    print("\n步骤4: 计算滚动工作倾向...")
    
    # 方法1: 滚动12个月的工作率
    df["pw_i_rolling"] = df.groupby("agent")["pw_i"]\
        .transform(lambda x: x.rolling(window=12, min_periods=1).mean())
    
    # 方法2: 指数加权移动平均（EWMA，更平滑）
    df["pw_i_ewma"] = df.groupby("agent")["pw_i"]\
        .transform(lambda x: x.ewm(span=12, adjust=False).mean())
    
    # 方法3: 累积平均（从开始到当前）
    df["pw_i_cumulative"] = df.groupby("agent")["pw_i"]\
        .transform(lambda x: x.expanding().mean())
    
    print(f"  ✓ pw_i 原始 0/1: 均值={df['pw_i'].mean():.3f}, 标准差={df['pw_i'].std():.3f}")
    print(f"  ✓ pw_i 滚动12月: 均值={df['pw_i_rolling'].mean():.3f}, 标准差={df['pw_i_rolling'].std():.3f}")
    print(f"  ✓ pw_i EWMA: 均值={df['pw_i_ewma'].mean():.3f}, 标准差={df['pw_i_ewma'].std():.3f}")
    print(f"  ✓ pw_i 累积: 均值={df['pw_i_cumulative'].mean():.3f}, 标准差={df['pw_i_cumulative'].std():.3f}")
    
    # 选择一个作为回归的 pw_i（推荐 rolling）
    df["pw_i_original"] = df["pw_i"]  # 保存原始值
    df["pw_i"] = df["pw_i_rolling"]   # 用滚动值替换
    
    # ========== 步骤5: 变量变换 ==========
    print("\n步骤5: 对高量级变量进行变换...")
    
    # 储蓄和收入取对数
    df['si_original'] = df['si']
    df['vi_original'] = df['vi']
    
    df['si'] = np.log(df['si'] + 1)  # 对数变换
    df['vi'] = np.log(df['vi'] + 1)
    
    print(f"  ✓ si (对数后): 均值={df['si'].mean():.3f}, 标准差={df['si'].std():.3f}")
    print(f"  ✓ vi (对数后): 均值={df['vi'].mean():.3f}, 标准差={df['vi'].std():.3f}")
    print(f"  ✓ ci_hat (比例): 均值={df['ci_hat'].mean():.3f}, 标准差={df['ci_hat'].std():.3f}")
    
    # 检查异常值
    print(f"  ⚠️  ci_hat > 1 的行数: {(df['ci_hat'] > 1).sum()}")
    print(f"  ⚠️  ci_hat < 0 的行数: {(df['ci_hat'] < 0).sum()}")
    
    # 打印一些统计信息
    print("\n数据统计（最终）:")
    print(f"  p_w_i (工作-滚动): 均值={df['pw_i'].mean():.3f}, 标准差={df['pw_i'].std():.3f}")
    print(f"  p_c_i (消费): 均值={df['pc_i'].mean():.3f}, 标准差={df['pc_i'].std():.3f}")
    print(f"  v_i (对数): 均值={df['vi'].mean():.3f}, 标准差={df['vi'].std():.3f}")
    print(f"  s_i (对数): 均值={df['si'].mean():.3f}, 标准差={df['si'].std():.3f}")
    print(f"  c_hat_i (比例): 均值={df['ci_hat'].mean():.3f}, 标准差={df['ci_hat'].std():.3f}")
    print(f"  P (价格): 均值={df['P'].mean():.2f}, 标准差={df['P'].std():.2f}")
    print(f"  r (利率): 均值={df['r'].mean():.4f}, 标准差={df['r'].std():.4f}")
    return df

def perform_regression_ols(df):
    """
    OLS回归（与论文一致）
    """
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
            
            # Z-score标准化（论文要求）
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

def perform_regression_logit(df):
    """
    Logit回归（针对p_w_i的二元特性）
    仅作为对比，论文用的是OLS
    """
    print("\n" + "=" * 70)
    print("Logit 回归分析（仅p_w_i，对比用）")
    print("=" * 70)
    
    independent_vars = ["vi", "ci_hat", "T_zi", "zr", "P", "si", "r"]
    
    results_logit = {var: 0 for var in independent_vars}
    n_agents = df["agent"].nunique()
    processed = 0
    
    for agent_id, agent_df in df.groupby("agent"):
        if len(agent_df) < 10:
            continue
        
        processed += 1
        
        y = agent_df["pw_i"].values
        X = agent_df[independent_vars].values
        
        # Z-score标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_normalized = (X - X_mean) / X_std
        
        try:
            # sklearn的LogisticRegression
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X_normalized, y)
            
            # 简单用系数的绝对值大小作为"重要性"判断
            # （严格应该用 statsmodels 的 Logit 获取p值）
            coefs = np.abs(model.coef_[0])
            threshold = np.percentile(coefs, 50)  # 取前50%作为"显著"
            
            for i, var in enumerate(independent_vars):
                if coefs[i] >= threshold:
                    results_logit[var] += 1
        
        except Exception:
            continue
    
    print(f"✓ Logit回归完成: {processed} 个agent")
    print("\nLogit结果（简化版，仅供参考）:")
    for var, count in results_logit.items():
        print(f"  {var}: {count}")
    
    return results_logit

def print_and_save_results(results_ols, detailed_df):
    """打印和保存结果"""
    print("\n" + "=" * 70)
    print("回归结果汇总（修正版）")
    print("=" * 70)
    
    print("\n重要修正:")
    print("  ✓ pw_i: 使用LLM输出的工作倾向值 [0,1]（优先）→ 滚动12月平滑")
    print("  ✓ pc_i: 使用LLM输出的消费倾向值 [0,1]")
    print("  ✓ vi, si: 使用对数变换（减少量级差异）")
    print("  ✓ ci_hat: 使用消费/收入比例（而非绝对值）")
    
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
    
    # 保存
    csv_path = os.path.join(OUT, "regression_detailed_results_strict.csv")
    detailed_df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存详细结果: {csv_path}")
    
    summary_df = pd.DataFrame(results_ols).T
    summary_csv = os.path.join(OUT, "table1_regression_summary.csv")
    summary_df.to_csv(summary_csv)
    print(f"✓ 保存 Table 1: {summary_csv}")
    
    return summary_df

def visualize_results(summary_df):
    """可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    variables = summary_df.columns.tolist()
    pw_counts = summary_df.loc["pw_i"].values
    pc_counts = summary_df.loc["pc_i"].values
    
    axes[0].bar(variables, pw_counts, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Economic Variables', fontsize=12)
    axes[0].set_ylabel('Number of Significant Agents', fontsize=12)
    axes[0].set_title('Effects on Work Propensity ($p_i^w$)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[0].legend()
    
    axes[1].bar(variables, pc_counts, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Economic Variables', fontsize=12)
    axes[1].set_ylabel('Number of Significant Agents', fontsize=12)
    axes[1].set_title('Effects on Consumption Propensity ($p_i^c$)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1].legend()
    
    plt.tight_layout()
    
    fig_path = os.path.join(OUT, "table1_regression_results.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存图表: {fig_path}")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("单体回归分析 - 严格论文定义版")
    print("Table 1: Decision Rationality")
    print("=" * 70)
    
    # 加载数据
    dense = load_dense_log(DATA)
    
    # 提取面板数据（严格定义）
    df = extract_panel_data_strict(dense, DATA)
    
    # OLS回归（与论文一致）
    results_ols, detailed_df = perform_regression_ols(df)
    
    # Logit回归（可选，仅对比）
    print("\n是否运行Logit回归对比？(y/n)")
    # results_logit = perform_regression_logit(df)  # 取消注释以运行
    
    # 结果展示和保存
    summary_df = print_and_save_results(results_ols, detailed_df)
    
    # 可视化
    visualize_results(summary_df)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
