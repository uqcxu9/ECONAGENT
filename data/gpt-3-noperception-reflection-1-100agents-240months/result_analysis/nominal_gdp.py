# -*- coding: utf-8 -*-
"""
计算名义GDP (Nominal GDP)
Definition: Annual nominal GDP = Σ(S × P) over one year
其中 S = Supply (月产出), P = Price (商品价格)
"""

import os
import pickle as pkl
import csv
import matplotlib.pyplot as plt
import numpy as np

BASE = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent"
MODEL = "gpt-3-noperception-reflection-1-100agents-240months"
DATA = os.path.join(BASE, "data", MODEL)
OUT  = os.path.join(DATA, "result_analysis")
os.makedirs(OUT, exist_ok=True)

P_DENSE = os.path.join(DATA, "dense_log.pkl")

# ============================================================================
# DummyUnpickler: 用于加载没有 ai_economist 模块的 pickle 文件
# ============================================================================
class DummyModule:
    """模拟模块，允许 pickle 反序列化时找到缺失的模块"""
    def __getattr__(self, name):
        return DummyModule()

class DummyUnpickler(pkl.Unpickler):
    """自定义 Unpickler，处理缺失的 ai_economist 模块"""
    def find_class(self, module, name):
        # 如果是 ai_economist 模块，返回一个占位类
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

def compute_gdp_from_dense_log():
    """
    从 dense_log.pkl 中计算 GDP
    """
    print("=" * 70)
    print("从 dense_log.pkl 计算名义GDP")
    print("=" * 70)
    
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    print(f"\ndense_log 包含的键: {list(dense.keys())}")
    
    # 检查 world 对象
    if 'world' not in dense:
        print("错误: dense_log 中没有 'world' 数据")
        return None
    
    world_data = dense['world']
    
    # 打印 world 的结构
    print(f"\nworld 数据类型: {type(world_data)}")
    
    if isinstance(world_data, dict):
        print(f"world 包含的键: {list(world_data.keys())}")
    elif hasattr(world_data, '__dict__'):
        print(f"world 对象的属性: {list(vars(world_data).keys())[:20]}")
    
    return world_data

def extract_supply_and_price(dense):
    """
    从 dense_log 中提取每月的 Supply 和 Price
    """
    # 方案1: 从 world 对象直接读取（如果有保存）
    if 'world' in dense:
        world = dense['world']
        
        # 检查是否有价格历史
        if isinstance(world, dict):
            prices = world.get('price', [])
            supply = world.get('supply', [])
        else:
            prices = getattr(world, 'price', [])
            supply = getattr(world, 'supply', []) if hasattr(world, 'supply') else []
        
        if prices:
            print(f"\n✓ 找到价格历史: {len(prices)} 个数据点")
            print(f"  前5个价格: {prices[:5]}")
        
        if supply:
            print(f"✓ 找到供给历史: {len(supply)} 个数据点")
            print(f"  前5个供给: {supply[:5]}")
        
        # 如果有完整数据，直接计算
        if prices and supply and len(prices) == len(supply):
            monthly_gdp = [s * p for s, p in zip(supply, prices)]
            return monthly_gdp, prices, supply
    
    # 方案2: 从 actions 计算 Supply
    print("\n从 actions 计算 Supply...")
    actions = dense.get('actions', [])
    states = dense.get('states', [])
    
    if not actions:
        print("错误: 没有 actions 数据")
        return None, None, None
    
    monthly_supply = []
    monthly_price = []
    
    # 尝试从 states 或 actions 中提取价格
    # 价格信息通常在 planner 的观察中
    for m in range(len(actions)):
        # 计算当月总产出
        # Supply = Σ(agent工作的小时数 × 生产率)
        # 根据文档1，每个工作的agent贡献 168 小时 × 1（生产率）
        supply = 0
        
        month_actions = actions[m]
        for agent_id, action in month_actions.items():
            if agent_id == 'p':
                continue
            
            # 提取劳动决策
            if isinstance(action, dict):
                labor = action.get('SimpleLabor', 0)
            elif isinstance(action, (list, tuple)):
                labor = action[0]
            else:
                labor = 0
            
            # 如果工作，贡献 168 小时
            if labor == 1:
                supply += 168  # 168小时 × 生产率1
        
        monthly_supply.append(supply)
        
        # 提取价格（从 states 的 planner 观察中）
        if m < len(states):
            state = states[m]
            
            # 尝试从 planner 的观察中获取价格
            if 'p' in state and isinstance(state['p'], dict):
                price = state['p'].get('price', None)
                if price is None:
                    # 尝试其他可能的键名
                    for key in state['p'].keys():
                        if 'price' in key.lower():
                            price = state['p'][key]
                            break
                
                if price is not None:
                    monthly_price.append(price)
                else:
                    monthly_price.append(None)
            else:
                monthly_price.append(None)
    
    # 检查是否成功提取价格
    valid_prices = [p for p in monthly_price if p is not None]
    if not valid_prices:
        print("警告: 无法从 states 提取价格")
        return None, None, monthly_supply
    
    print(f"\n✓ 计算完成:")
    print(f"  月份数: {len(monthly_supply)}")
    print(f"  前5个月供给: {monthly_supply[:5]}")
    print(f"  前5个月价格: {monthly_price[:5]}")
    
    # 计算 GDP
    monthly_gdp = []
    for s, p in zip(monthly_supply, monthly_price):
        if p is not None:
            monthly_gdp.append(s * p)
        else:
            monthly_gdp.append(None)
    
    return monthly_gdp, monthly_price, monthly_supply

def compute_gdp_from_env_file():
    """
    从保存的 env 文件中读取 GDP 数据
    这是最可靠的方法！
    """
    print("\n" + "=" * 70)
    print("从 env pickle 文件读取数据")
    print("=" * 70)
    
    # 加载最终的 env 文件（包含完整历史）
    env_file = os.path.join(DATA, "env_240.pkl")
    
    if not os.path.exists(env_file):
        print(f"错误: 文件不存在 {env_file}")
        
        # 尝试其他文件
        for ep in [234, 228, 216, 12]:
            alt_file = os.path.join(DATA, f"env_{ep}.pkl")
            if os.path.exists(alt_file):
                print(f"找到备用文件: {alt_file}")
                env_file = alt_file
                break
    
    if not os.path.exists(env_file):
        print("错误: 找不到任何 env pickle 文件")
        return None, None, None, None
    
    print(f"\n加载: {env_file}")
    
    try:
        with open(env_file, "rb") as f:
            env = DummyUnpickler(f).load()
        
        print("✓ 加载成功 (使用 DummyUnpickler)")
        
        # 检查 env.world
        if not hasattr(env, 'world'):
            print("错误: env 没有 world 属性")
            return None, None, None, None
        
        world = env.world
        print(f"\n检查 world 对象的属性...")
        
        # 获取价格历史
        if hasattr(world, 'price'):
            prices = world.price
            print(f"✓ 价格历史: {len(prices)} 个数据点")
            print(f"  前5个: {prices[:5]}")
            print(f"  后5个: {prices[-5:]}")
        else:
            print("✗ 没有 price 属性")
            return None, None, None, None
        
        # 检查是否有预计算的 GDP
        if hasattr(world, 'nominal_gdp'):
            nominal_gdp_yearly = world.nominal_gdp
            print(f"\n✓ 找到预计算的年度名义GDP: {len(nominal_gdp_yearly)} 个年度数据")
            print(f"  前5年: {[f'{gdp:,.2f}' for gdp in nominal_gdp_yearly[:5]]}")
            
            # 这是年度数据，直接返回
            return None, prices, None, nominal_gdp_yearly
        
        # 如果没有年度GDP，尝试获取供给数据手动计算
        # 可能的属性名: supply, production, total_production, output
        supply = None
        for attr_name in ['supply', 'production', 'total_production', 'output', 'goods', 'total_products']:
            if hasattr(world, attr_name):
                attr_val = getattr(world, attr_name)
                # total_products 可能是单个数字而不是列表
                if isinstance(attr_val, (list, tuple)) or hasattr(attr_val, '__iter__'):
                    supply = list(attr_val)
                    print(f"✓ 找到供给数据: {attr_name}, {len(supply)} 个数据点")
                    print(f"  前5个: {supply[:5]}")
                    break
        
        if supply is None:
            print("\n✗ 没有找到供给历史数据")
            print("可用属性:", [a for a in dir(world) if not a.startswith('_')][:30])
            return None, None, None, None
        
        # 计算月度 GDP
        monthly_gdp = [s * p for s, p in zip(supply, prices)]
        
        print(f"\n✓ 成功计算月度GDP: {len(monthly_gdp)} 个月")
        print(f"  前5个月GDP: {[f'{gdp:,.2f}' for gdp in monthly_gdp[:5]]}")
        
        return monthly_gdp, prices, supply, None
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def calculate_yearly_gdp(monthly_gdp):
    """计算年度GDP"""
    if not monthly_gdp:
        return None, None
    
    # 过滤 None 值
    monthly_gdp = [gdp for gdp in monthly_gdp if gdp is not None]
    
    years = []
    yearly_gdp = []
    
    for y in range(0, len(monthly_gdp), 12):
        chunk = monthly_gdp[y:y+12]
        if len(chunk) == 12:
            years.append(y // 12 + 1)
            yearly_gdp.append(sum(chunk))
    
    return yearly_gdp, years

def calculate_gdp_growth(yearly_gdp):
    """计算名义GDP增长率"""
    if not yearly_gdp or len(yearly_gdp) < 2:
        return []
    
    growth_rates = []
    for i in range(1, len(yearly_gdp)):
        growth = (yearly_gdp[i] - yearly_gdp[i-1]) / yearly_gdp[i-1] * 100
        growth_rates.append(growth)
    
    return growth_rates

def save_and_plot_gdp(yearly_gdp, years):
    """保存和绘制GDP数据"""
    if not yearly_gdp or not years:
        print("没有GDP数据可保存")
        return
    
    # 计算增长率
    growth_rates = calculate_gdp_growth(yearly_gdp)
    growth_years = years[1:] if growth_rates else []
    
    # 保存GDP的CSV
    csv_path = os.path.join(OUT, "nominal_gdp_yearly.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "nominal_gdp", "growth_rate"])
        for i, (y, gdp) in enumerate(zip(years, yearly_gdp)):
            if i == 0:
                w.writerow([y, gdp, "N/A"])
            else:
                w.writerow([y, gdp, growth_rates[i-1]])
    print(f"\n✓ 保存GDP CSV: {csv_path}")
    
    # 保存增长率的CSV
    if growth_rates:
        growth_csv_path = os.path.join(OUT, "nominal_gdp_growth_yearly.csv")
        with open(growth_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["year", "growth_rate"])
            for y, rate in zip(growth_years, growth_rates):
                w.writerow([y, rate])
        print(f"✓ 保存增长率CSV: {growth_csv_path}")
    
    # 绘制双图：GDP水平 + 增长率
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图：GDP水平
    axes[0].plot(years, yearly_gdp, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Nominal GDP', fontsize=12)
    axes[0].set_title('Annual Nominal GDP (Σ S×P over one year)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 在图上标注最大值和最小值
    max_idx = np.argmax(yearly_gdp)
    min_idx = np.argmin(yearly_gdp)
    axes[0].annotate(f'Max: {yearly_gdp[max_idx]:,.0f}', 
                     xy=(years[max_idx], yearly_gdp[max_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='green',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # 下图：GDP增长率（折线图）
    if growth_rates:
        # 绘制折线
        axes[1].plot(growth_years, growth_rates, marker='o', linewidth=2, markersize=8, 
                     color='#E74C3C', label='GDP Growth Rate')
        
        # 添加0轴参考线
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 标注平均增长率
        avg_growth = np.mean(growth_rates)
        axes[1].axhline(y=avg_growth, color='#3498DB', linestyle='--', linewidth=1.5, 
                       label=f'Average: {avg_growth:.2f}%')
        
        # 用颜色填充区域来表示正负增长
        axes[1].fill_between(growth_years, 0, growth_rates, 
                             where=[g >= 0 for g in growth_rates],
                             color='green', alpha=0.2, interpolate=True, label='Positive Growth')
        axes[1].fill_between(growth_years, 0, growth_rates, 
                             where=[g < 0 for g in growth_rates],
                             color='red', alpha=0.2, interpolate=True, label='Negative Growth')
        
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Growth Rate (%)', fontsize=12)
        axes[1].set_title('Nominal GDP Growth Rate (Year-over-Year)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    fig_path = os.path.join(OUT, "nominal_gdp_yearly.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存图表: {fig_path}")
    plt.close()
    
    # 单独绘制增长率图（更大更清晰）
    if growth_rates:
        plt.figure(figsize=(14, 6))
        colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in growth_rates]
        bars = plt.bar(growth_years, growth_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for i, (year, rate) in enumerate(zip(growth_years, growth_rates)):
            plt.text(year, rate + (0.5 if rate >= 0 else -0.5), f'{rate:.1f}%', 
                    ha='center', va='bottom' if rate >= 0 else 'top', fontsize=8)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        avg_growth = np.mean(growth_rates)
        plt.axhline(y=avg_growth, color='#3498db', linestyle='--', linewidth=2, 
                   label=f'Average Growth: {avg_growth:.2f}%')
        
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12)
        plt.title('Nominal GDP Growth Rate (Year-over-Year)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        growth_fig_path = os.path.join(OUT, "nominal_gdp_growth_yearly.png")
        plt.savefig(growth_fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存增长率图表: {growth_fig_path}")
        plt.close()
    
    # 统计信息
    print("\n" + "=" * 70)
    print("年度名义GDP统计")
    print("=" * 70)
    print(f"年份数量: {len(yearly_gdp)}")
    print(f"平均GDP: {np.mean(yearly_gdp):,.2f}")
    print(f"最大GDP: {np.max(yearly_gdp):,.2f} (第{years[np.argmax(yearly_gdp)]}年)")
    print(f"最小GDP: {np.min(yearly_gdp):,.2f} (第{years[np.argmin(yearly_gdp)]}年)")
    
    if growth_rates:
        print(f"\n平均增长率: {np.mean(growth_rates):.2f}%")
        print(f"增长率范围: {np.min(growth_rates):.2f}% 到 {np.max(growth_rates):.2f}%")
        print(f"最高增长: {np.max(growth_rates):.2f}% (第{growth_years[np.argmax(growth_rates)]}年)")
        print(f"最低增长: {np.min(growth_rates):.2f}% (第{growth_years[np.argmin(growth_rates)]}年)")
        
        # 统计正负增长年份
        positive_years = sum(1 for g in growth_rates if g > 0)
        negative_years = sum(1 for g in growth_rates if g < 0)
        print(f"\n正增长年份: {positive_years}/{len(growth_rates)}")
        print(f"负增长年份: {negative_years}/{len(growth_rates)}")
    
    print("\n前10年的GDP和增长率:")
    print("年份 |      GDP       | 增长率")
    print("-" * 40)
    for i in range(min(10, len(yearly_gdp))):
        if i == 0:
            print(f" {years[i]:2d}  | {yearly_gdp[i]:>12,.2f} |   N/A")
        else:
            growth = growth_rates[i-1]
            print(f" {years[i]:2d}  | {yearly_gdp[i]:>12,.2f} | {growth:+6.2f}%")

def main():
    print("\n" + "=" * 70)
    print("名义GDP计算程序")
    print("=" * 70)
    
    # 优先级1: 从 env 文件读取（最可靠）
    result = compute_gdp_from_env_file()
    
    # 检查返回值
    if result is None or len(result) < 4:
        print("\n✗ 无法从 env 文件读取数据")
        return
    
    monthly_gdp, prices, supply, yearly_gdp_precomputed = result
    
    # 如果有预计算的年度GDP，直接使用
    if yearly_gdp_precomputed is not None:
        yearly_gdp = list(yearly_gdp_precomputed)
        years = list(range(1, len(yearly_gdp) + 1))
        
        print(f"\n✓ 使用预计算的年度GDP数据")
        save_and_plot_gdp(yearly_gdp, years)
        
    elif monthly_gdp is not None:
        # 如果只有月度数据，计算年度
        print(f"\n✓ 从月度数据计算年度GDP")
        yearly_gdp, years = calculate_yearly_gdp(monthly_gdp)
        
        if yearly_gdp:
            save_and_plot_gdp(yearly_gdp, years)
        else:
            print("\n✗ 无法计算年度GDP")
    else:
        print("\n✗ 无法获取GDP数据")
        
        # 打印调试信息
        print("\n请手动检查:")
        print("1. env_240.pkl 是否存在？")
        print("2. env.world.nominal_gdp 是否存在？")
        print("3. env.world.price 和供给数据是否存在？")
        return
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()