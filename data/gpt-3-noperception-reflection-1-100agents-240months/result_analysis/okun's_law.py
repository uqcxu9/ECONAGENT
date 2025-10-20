# -*- coding: utf-8 -*-
"""
绘制奥肯法则 (Okun's Law)
横轴: Unemployment Rate Growth (失业率增长)
纵轴: Real GDP Growth (实际GDP增长率)
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

def compute_unemployment_growth(unemployment_rates):
    """
    计算失业率增长
    Growth = u_t - u_{t-1} (不是增长率，是绝对变化)
    """
    growth = []
    for i in range(1, len(unemployment_rates)):
        change = unemployment_rates[i] - unemployment_rates[i-1]
        growth.append(change)
    
    return growth

def compute_real_gdp_growth():
    """
    计算实际GDP增长率
    Real GDP = S × P0 (用第1年的价格作为基准价格)
    Growth Rate = (Real_GDP_t - Real_GDP_{t-1}) / Real_GDP_{t-1}
    """
    print("\n" + "=" * 70)
    print("计算实际GDP增长率")
    print("=" * 70)
    
    # 从 env 文件读取
    env_file = os.path.join(DATA, "env_240.pkl")
    
    if not os.path.exists(env_file):
        print(f"✗ 文件不存在: {env_file}")
        # 尝试其他文件
        for ep in [234, 228, 216, 12]:
            alt_file = os.path.join(DATA, f"env_{ep}.pkl")
            if os.path.exists(alt_file):
                print(f"✓ 找到备用文件: {alt_file}")
                env_file = alt_file
                break
    
    if not os.path.exists(env_file):
        print("✗ 找不到任何 env 文件")
        return None
    
    print(f"加载: {env_file}")
    
    try:
        with open(env_file, "rb") as f:
            env = DummyUnpickler(f).load()
        
        print("✓ 成功加载 (使用 DummyUnpickler)")
        
        if not hasattr(env, 'world'):
            print("✗ env 没有 world 属性")
            return None
        
        world = env.world
        
        # 获取价格历史
        if not hasattr(world, 'price'):
            print("✗ world 没有 price 属性")
            return None
        
        prices = world.price
        print(f"✓ 价格历史: {len(prices)} 个月")
        
        # 获取供给历史
        supply = None
        for attr_name in ['supply', 'production', 'total_production', 'output']:
            if hasattr(world, attr_name):
                supply = getattr(world, attr_name)
                print(f"✓ 供给数据: {attr_name}, {len(supply)} 个月")
                break
        
        if supply is None:
            print("✗ 找不到供给数据，从 actions 计算...")
            supply = compute_supply_from_actions()
            if supply is None:
                return None
        
        # 确保长度一致
        min_len = min(len(prices), len(supply))
        prices = prices[:min_len]
        supply = supply[:min_len]
        
        print(f"✓ 数据长度: {min_len} 个月")
        
        # 基准价格 = 第1年的平均价格
        base_price = sum(prices[:12]) / 12
        print(f"✓ 基准价格 (第1年平均): {base_price:.2f}")
        
        # 计算月度实际GDP
        monthly_real_gdp = [s * base_price for s in supply]
        
        # 计算年度实际GDP
        yearly_real_gdp = []
        for y in range(0, len(monthly_real_gdp), 12):
            chunk = monthly_real_gdp[y:y+12]
            if len(chunk) == 12:
                yearly_real_gdp.append(sum(chunk))
        
        print(f"✓ 年度实际GDP: {len(yearly_real_gdp)} 年")
        print(f"  前3年: {[f'{gdp:,.2f}' for gdp in yearly_real_gdp[:3]]}")
        
        # 计算增长率
        real_gdp_growth = []
        for i in range(1, len(yearly_real_gdp)):
            if yearly_real_gdp[i-1] > 0:
                growth = (yearly_real_gdp[i] - yearly_real_gdp[i-1]) / yearly_real_gdp[i-1]
                real_gdp_growth.append(growth)
            else:
                real_gdp_growth.append(0)
        
        print(f"✓ 实际GDP增长率: {len(real_gdp_growth)} 个数据点")
        print(f"  前5年增长率: {[f'{g*100:.2f}%' for g in real_gdp_growth[:5]]}")
        
        return real_gdp_growth
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return None

def compute_supply_from_actions():
    """
    从 actions 计算每月的供给
    Supply = Σ(劳动力 × 168小时 × 生产率1)
    """
    print("  从 actions 计算供给...")
    
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    actions = dense.get("actions", [])
    if not actions:
        print("  ✗ 没有 actions 数据")
        return None
    
    monthly_supply = []
    
    for m in range(len(actions)):
        month_actions = actions[m]
        supply = 0
        
        for agent_id, action in month_actions.items():
            if agent_id == 'p':
                continue
            
            # 提取劳动决策
            if isinstance(action, dict):
                labor = action.get('SimpleLabor', 0)
            elif isinstance(action, (list, tuple)) and len(action) > 0:
                labor = action[0]
            else:
                labor = 0
            
            # 如果工作，贡献 168 小时
            if labor == 1:
                supply += 168
        
        monthly_supply.append(supply)
    
    print(f"  ✓ 计算完成: {len(monthly_supply)} 个月")
    print(f"    前3个月供给: {monthly_supply[:3]}")
    
    return monthly_supply

def plot_okuns_law(unemployment_growth, real_gdp_growth):
    """
    绘制奥肯法则
    """
    if not unemployment_growth or not real_gdp_growth:
        print("✗ 缺少数据，无法绘制")
        return
    
    # 确保数据长度一致
    min_len = min(len(unemployment_growth), len(real_gdp_growth))
    unemployment_growth = unemployment_growth[:min_len]
    real_gdp_growth = real_gdp_growth[:min_len]
    
    print("\n" + "=" * 70)
    print("奥肯法则数据")
    print("=" * 70)
    print(f"数据点数: {len(unemployment_growth)}")
    print(f"失业率增长范围: {min(unemployment_growth)*100:.2f}% - {max(unemployment_growth)*100:.2f}%")
    print(f"实际GDP增长率范围: {min(real_gdp_growth)*100:.2f}% - {max(real_gdp_growth)*100:.2f}%")
    
    # 计算相关系数
    if len(unemployment_growth) > 2:
        correlation, p_value = stats.pearsonr(unemployment_growth, real_gdp_growth)
        print(f"\n皮尔逊相关系数: {correlation:.3f}")
        print(f"p值: {p_value:.4f}")
        if p_value < 0.001:
            print("✓✓✓ 高度显著 (p < 0.001)")
        elif p_value < 0.01:
            print("✓✓ 非常显著 (p < 0.01)")
        elif p_value < 0.05:
            print("✓ 显著 (p < 0.05)")
        else:
            print("✗ 相关性不显著")
        
        if correlation < 0:
            print("✓ 负相关关系（符合奥肯法则）")
        else:
            print("✗ 正相关关系（不符合奥肯法则）")
    
    # 保存数据到CSV
    csv_path = os.path.join(OUT, "okuns_law_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "unemployment_rate_growth", "real_gdp_growth"])
        for i, (u_g, gdp_g) in enumerate(zip(unemployment_growth, real_gdp_growth), 2):
            w.writerow([i, u_g, gdp_g])
    print(f"\n✓ 保存数据: {csv_path}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    scatter = ax.scatter(unemployment_growth, real_gdp_growth,
                        s=100, alpha=0.6, c=range(len(unemployment_growth)),
                        cmap='viridis', edgecolors='black', linewidth=1)
    
    # 添加年份标签（选择性标注）
    for i, (u, g) in enumerate(zip(unemployment_growth, real_gdp_growth), 2):
        if i <= 10 or i > 15:  # 只标注部分年份
            ax.annotate(f'{i}', (u, g), fontsize=8, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
    
    # 拟合线性回归线
    if len(unemployment_growth) > 2:
        z = np.polyfit(unemployment_growth, real_gdp_growth, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(unemployment_growth), max(unemployment_growth), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
               label=f'拟合线: y = {z[0]:.2f}x + {z[1]:.3f}')
        
        # 计算奥肯系数（斜率的绝对值）
        okun_coefficient = abs(z[0])
        print(f"\n奥肯系数: {okun_coefficient:.2f}")
        print(f"含义: 失业率每上升1个百分点，实际GDP下降约{okun_coefficient:.2f}个百分点")
    
    # 设置坐标轴
    ax.set_xlabel('Unemployment Rate Growth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Real GDP Growth', fontsize=14, fontweight='bold')
    ax.set_title("Okun's Law", fontsize=16, fontweight='bold')
    
    # 设置坐标轴范围（根据实际数据动态调整）
    u_range = max(unemployment_growth) - min(unemployment_growth)
    g_range = max(real_gdp_growth) - min(real_gdp_growth)
    
    ax.set_xlim(min(unemployment_growth) - 0.1*u_range, 
                max(unemployment_growth) + 0.1*u_range)
    ax.set_ylim(min(real_gdp_growth) - 0.1*g_range,
                max(real_gdp_growth) + 0.1*g_range)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 添加图例
    ax.legend(loc='best', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Year', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(OUT, "okuns_law.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存图片: {fig_path}")
    
    plt.show()
    plt.close()

def compare_with_paper_format():
    """
    绘制与论文相同格式的图（如果需要特定的坐标轴范围）
    """
    # 读取数据
    csv_path = os.path.join(OUT, "okuns_law_data.csv")
    if not os.path.exists(csv_path):
        print("请先运行 main() 生成数据")
        return
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    unemployment_growth = df['unemployment_rate_growth'].values
    real_gdp_growth = df['real_gdp_growth'].values
    
    # 绘制与论文格式完全一致的图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 散点图（不带颜色梯度，与论文一致）
    ax.scatter(unemployment_growth, real_gdp_growth,
              s=80, alpha=0.6, color='steelblue', edgecolors='black', linewidth=1)
    
    # 拟合线
    if len(unemployment_growth) > 2:
        z = np.polyfit(unemployment_growth, real_gdp_growth, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(unemployment_growth), max(unemployment_growth), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # 设置标题和标签
    ax.set_xlabel('Unemployment Rate Growth', fontsize=12)
    ax.set_ylabel('Real GDP Growth', fontsize=12)
    ax.set_title("Okun's Law", fontsize=14)
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    # 零线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    fig_path = os.path.join(OUT, "okuns_law_paper_format.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存论文格式图: {fig_path}")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("奥肯法则绘制程序")
    print("=" * 70)
    
    # 步骤1: 计算年度失业率
    print("\n步骤1: 计算年度失业率...")
    unemployment_rates = compute_unemployment_rate()
    
    if not unemployment_rates:
        print("✗ 无法计算失业率")
        return
    
    print(f"✓ 完成，共{len(unemployment_rates)}年数据")
    
    # 步骤2: 计算失业率增长
    print("\n步骤2: 计算失业率增长...")
    unemployment_growth = compute_unemployment_growth(unemployment_rates)
    print(f"✓ 完成，共{len(unemployment_growth)}个数据点")
    print(f"  范围: {min(unemployment_growth)*100:.2f}% - {max(unemployment_growth)*100:.2f}%")
    
    # 步骤3: 计算实际GDP增长率
    print("\n步骤3: 计算实际GDP增长率...")
    real_gdp_growth = compute_real_gdp_growth()
    
    if not real_gdp_growth:
        print("✗ 无法计算实际GDP增长率")
        return
    
    # 步骤4: 绘制奥肯法则
    print("\n步骤4: 绘制奥肯法则...")
    plot_okuns_law(unemployment_growth, real_gdp_growth)
    
    # 步骤5: 生成论文格式的图（可选）
    print("\n步骤5: 生成论文格式图...")
    compare_with_paper_format()
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()