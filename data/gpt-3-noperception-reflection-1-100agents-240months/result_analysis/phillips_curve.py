# -*- coding: utf-8 -*-
"""
绘制菲利普斯曲线 (Phillips Curve)
横轴: Unemployment Rate (失业率)
纵轴: Wage Inflation (工资通胀率)
"""

import os
import pickle as pkl
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent"
MODEL = "gpt-3-noperception-reflection-1-100agents-240months"
DATA = os.path.join(BASE, "data", MODEL)
OUT  = os.path.join(DATA, "result_analysis")
os.makedirs(OUT, exist_ok=True)

P_DENSE = os.path.join(DATA, "dense_log.pkl")

# ============================================================================
# DummyUnpickler: 用于加载包含 ai_economist 对象的 pickle 文件
# ============================================================================
class DummyModule:
    """模拟任意模块"""
    def __getattr__(self, name):
        return DummyModule()
    
    def __call__(self, *args, **kwargs):
        return DummyModule()

class DummyUnpickler(pkl.Unpickler):
    """自定义 Unpickler，绕过 ai_economist 模块依赖"""
    def find_class(self, module, name):
        if 'ai_economist' in module:
            # 返回一个虚拟类
            return type(name, (), {})
        return super().find_class(module, name)

def compute_unemployment_rate():
    """
    计算年度失业率
    返回: 年度失业率列表
    """
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    states = dense.get("states", [])
    
    # 从第1个月开始（跳过初始化）
    monthly_unemp_rate = []
    
    for m in range(1, len(states)):
        month_states = states[m] or {}
        unemployed = 0
        total = 0
        
        for agent_id, agent_state in month_states.items():
            if agent_id == 'p':
                continue
            
            total += 1
            
            if isinstance(agent_state, dict):
                job = agent_state.get('endogenous', {}).get('job', None)
                if job == 'Unemployment':
                    unemployed += 1
        
        rate = unemployed / total if total > 0 else 0
        monthly_unemp_rate.append(rate)
    
    # 计算年度平均
    yearly_unemp_rate = []
    for y in range(0, len(monthly_unemp_rate), 12):
        chunk = monthly_unemp_rate[y:y+12]
        if len(chunk) == 12:
            yearly_unemp_rate.append(sum(chunk) / 12)
    
    return yearly_unemp_rate

def compute_monthly_avg_wage(dense):
    """
    计算每月平均工资 - 使用实际收入
    """
    states = dense.get('states', [])
    monthly_avg_wage = []
    
    for m in range(len(states)):
        month_states = states[m] or {}
        total_wage = 0
        count = 0
        
        for agent_id, agent_state in month_states.items():
            if agent_id == 'p':
                continue
            
            if isinstance(agent_state, dict):
                # 方法1：使用实际收入（最准确）
                income = agent_state.get('income', {}).get('Coin', 0)
                
                # 方法2：如果income为0，用skill计算潜在工资
                if income == 0:
                    skill = agent_state.get('skill', 0)
                    labor = agent_state.get('endogenous', {}).get('Labor', 0)
                    if labor > 0 and skill > 0:
                        income = skill * labor
                
                if income > 0:
                    total_wage += income
                    count += 1
        
        avg_wage = total_wage / count if count > 0 else 0
        monthly_avg_wage.append(avg_wage)
    
    return monthly_avg_wage

def compute_wage_inflation():
    """
    计算年度工资通胀率
    Wage Inflation = (平均工资_year_n - 平均工资_year_n-1) / 平均工资_year_n-1
    """
    print("\n" + "=" * 70)
    print("计算工资通胀率")
    print("=" * 70)
    
    # 方法1: 从 env 文件读取工资历史
    env_file = os.path.join(DATA, "env_240.pkl")
    
    if os.path.exists(env_file):
        print(f"从 {env_file} 读取数据...")
        try:
            with open(env_file, "rb") as f:
                env = DummyUnpickler(f).load()
            
            print("✓ 成功加载 env 文件 (使用 DummyUnpickler)")
            
            # 检查是否有工资历史数据
            if hasattr(env, 'world'):
                world = env.world
                
                # 尝试找工资数据
                for attr in ['wage', 'wages', 'hourly_wage', 'wage_rate']:
                    if hasattr(world, attr):
                        wage_history = getattr(world, attr)
                        print(f"✓ 从env找到工资数据: {attr}, 长度={len(wage_history)}")
                        return compute_inflation_from_history(wage_history, "工资")
        
        except Exception as e:
            print(f"从 env 文件读取失败: {e}")
    
    # 方法2: 从 states 计算每月平均工资
    print("\n从 states 计算每月平均工资...")
    
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    monthly_avg_wage = compute_monthly_avg_wage(dense)
    
    if not any(monthly_avg_wage):
        print("✗ 无法计算工资，所有值都是0")
        return None
    
    print(f"✓ 计算完成: {len(monthly_avg_wage)} 个月")
    print(f"  前5个月平均工资: {[f'{w:.2f}' for w in monthly_avg_wage[:5]]}")
    
    return compute_inflation_from_history(monthly_avg_wage, "工资")

def compute_inflation_from_history(history, name="价格"):
    """
    从历史数据计算年度通胀率
    """
    # 计算年度平均
    yearly_avg = []
    for y in range(0, len(history), 12):
        chunk = history[y:y+12]
        if len(chunk) == 12:
            yearly_avg.append(sum(chunk) / 12)
    
    print(f"\n{name}年度平均值:")
    for i, avg in enumerate(yearly_avg[:5], 1):
        print(f"  第{i}年: {avg:.2f}")
    
    # 计算通胀率
    inflation_rates = []
    for i in range(1, len(yearly_avg)):
        if yearly_avg[i-1] > 0:
            inflation = (yearly_avg[i] - yearly_avg[i-1]) / yearly_avg[i-1]
            inflation_rates.append(inflation)
        else:
            inflation_rates.append(0)
    
    print(f"\n{name}年度通胀率（%）:")
    for i, rate in enumerate(inflation_rates[:5], 2):
        print(f"  第{i}年: {rate*100:.2f}%")
    
    return inflation_rates

def plot_phillips_curve(unemployment_rates, wage_inflation_rates):
    """
    绘制菲利普斯曲线
    """
    if not unemployment_rates or not wage_inflation_rates:
        print("✗ 缺少数据，无法绘制")
        return
    
    # 确保数据长度一致
    # unemployment_rates 可能有20个点（年1-20）
    # wage_inflation_rates 只有19个点（年2-20，因为第1年没有增长率）
    
    # 对齐数据：使用年2-20的失业率
    if len(unemployment_rates) > len(wage_inflation_rates):
        unemployment_rates = unemployment_rates[1:len(wage_inflation_rates)+1]
    elif len(wage_inflation_rates) > len(unemployment_rates):
        wage_inflation_rates = wage_inflation_rates[:len(unemployment_rates)]
    
    print("\n" + "=" * 70)
    print("菲利普斯曲线数据")
    print("=" * 70)
    print(f"数据点数: {len(unemployment_rates)}")
    print(f"失业率范围: {min(unemployment_rates)*100:.2f}% - {max(unemployment_rates)*100:.2f}%")
    print(f"工资通胀率范围: {min(wage_inflation_rates)*100:.2f}% - {max(wage_inflation_rates)*100:.2f}%")
    
    # 计算相关系数
    if len(unemployment_rates) > 2:
        correlation, p_value = stats.pearsonr(unemployment_rates, wage_inflation_rates)
        print(f"\n皮尔逊相关系数: {correlation:.3f}")
        print(f"p值: {p_value:.4f}")
        if p_value < 0.01:
            print("✓ 显著负相关 (p < 0.01)" if correlation < 0 else "✓ 显著正相关 (p < 0.01)")
        elif p_value < 0.05:
            print("✓ 显著负相关 (p < 0.05)" if correlation < 0 else "✓ 显著正相关 (p < 0.05)")
        else:
            print("✗ 相关性不显著")
    
    # 保存数据到CSV
    csv_path = os.path.join(OUT, "phillips_curve_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "unemployment_rate", "wage_inflation"])
        for i, (u, w_inf) in enumerate(zip(unemployment_rates, wage_inflation_rates), 2):
            w.writerow([i, u, w_inf])
    print(f"\n✓ 保存数据: {csv_path}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    scatter = ax.scatter(unemployment_rates, wage_inflation_rates, 
                        s=100, alpha=0.6, c=range(len(unemployment_rates)),
                        cmap='viridis', edgecolors='black', linewidth=1)
    
    # 添加年份标签（可选，如果点太多可以注释掉）
    for i, (u, w) in enumerate(zip(unemployment_rates, wage_inflation_rates), 2):
        if i <= 10 or i > 15:  # 只标注部分年份，避免拥挤
            ax.annotate(f'{i}', (u, w), fontsize=8, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
    
    # 拟合线性回归线
    if len(unemployment_rates) > 2:
        z = np.polyfit(unemployment_rates, wage_inflation_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(unemployment_rates), max(unemployment_rates), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
               label=f'拟合线: y = {z[0]:.2f}x + {z[1]:.3f}')
    
    # 设置坐标轴
    ax.set_xlabel('Unemployment Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wage Inflation', fontsize=14, fontweight='bold')
    ax.set_title('Phillips Curve', fontsize=16, fontweight='bold')
    
    # 动态设置坐标轴范围（根据实际数据）
    u_min, u_max = min(unemployment_rates), max(unemployment_rates)
    w_min, w_max = min(wage_inflation_rates), max(wage_inflation_rates)
    
    # 添加一些边距
    u_margin = (u_max - u_min) * 0.1
    w_margin = (w_max - w_min) * 0.1
    
    ax.set_xlim(u_min - u_margin, u_max + u_margin)
    ax.set_ylim(w_min - w_margin, w_max + w_margin)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0.05, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Year', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(OUT, "phillips_curve.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存图片: {fig_path}")
    
    plt.show()
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("菲利普斯曲线绘制程序")
    print("=" * 70)
    
    # 1. 计算失业率
    print("\n步骤1: 计算年度失业率...")
    unemployment_rates = compute_unemployment_rate()
    
    if not unemployment_rates:
        print("✗ 无法计算失业率")
        return
    
    print(f"✓ 完成，共{len(unemployment_rates)}年数据")
    print(f"  失业率范围: {min(unemployment_rates)*100:.2f}% - {max(unemployment_rates)*100:.2f}%")
    
    # 2. 计算工资通胀率
    print("\n步骤2: 计算年度工资通胀率...")
    wage_inflation_rates = compute_wage_inflation()
    
    if not wage_inflation_rates:
        print("✗ 无法计算工资通胀率")
        print("\n提示: 请检查以下文件是否存在:")
        print(f"  - {os.path.join(DATA, 'env_240.pkl')}")
        print("\n或者检查 states 中是否包含 skill 或 income 信息")
        return
    
    print(f"✓ 完成，共{len(wage_inflation_rates)}年数据")
    print(f"  工资通胀率范围: {min(wage_inflation_rates)*100:.2f}% - {max(wage_inflation_rates)*100:.2f}%")
    
    # 3. 绘制菲利普斯曲线
    print("\n步骤3: 绘制菲利普斯曲线...")
    plot_phillips_curve(unemployment_rates, wage_inflation_rates)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()