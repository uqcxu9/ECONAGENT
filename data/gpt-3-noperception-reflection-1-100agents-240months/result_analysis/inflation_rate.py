# # inflation_rate.py
# import pickle as pkl
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def compute_inflation_rate(dense_log_path, save_fig_path=None, save_result_path=None):
#     """
#     从 dense_log_*.pkl 提取月度价格（world[i]['Price']），计算年度通胀率并绘图。
#     规则：
#       - world[0]['Price'] = P0（初始）
#       - 第1年通胀: (P̄1 - P0) / P0
#       - 第n年(n>=2): (P̄n - P̄n-1) / P̄n-1
#     """
#     # 读取
#     with open(dense_log_path, 'rb') as f:
#         dense_log = pkl.load(f)

#     # 提取 P0 与月度价格（跳过初始）
#     world = dense_log['world']
#     p0 = float(world[0]['Price'])
#     monthly_prices = np.array([float(m['Price']) for m in world[1:]], dtype=float)  # 1..T 月
#     total_months = len(monthly_prices)  # 通常 240

#     if total_months < 12:
#         raise ValueError(f"有效月份不足 12（仅 {total_months}），无法计算年度均价与通胀率。")

#     # 对齐整年
#     remainder = total_months % 12
#     if remainder != 0:
#         print(f"⚠️ 有效月份 {total_months} 非 12 的整数倍，将截断最后 {remainder} 个月以对齐整年。")
#         monthly_prices = monthly_prices[:total_months - remainder]

#     # 年均价（每12个月一组）
#     total_years = len(monthly_prices) // 12
#     avg_prices = [monthly_prices[i*12:(i+1)*12].mean() for i in range(total_years)]

#     # 年度通胀率：第一年对 P0，后续对上年均价
#     inflation_rates = []
#     inflation_rates.append((avg_prices[0] - p0) / p0)
#     for n in range(1, total_years):
#         inflation_rates.append((avg_prices[n] - avg_prices[n-1]) / avg_prices[n-1])

#     # 画图
#     # === 绘图 ===
#     years = np.arange(1, total_years + 1)
#     plt.figure(figsize=(8, 5))
#     plt.plot(years, np.array(inflation_rates) * 100.0, marker='o', linestyle='-')
#     plt.title('Annual Inflation Rate')
#     plt.xlabel('Year')
#     plt.ylabel('Inflation Rate (%)')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.xticks(years)

# # 👇 固定纵坐标范围 -0.2 到 0.2（对应 -20% 到 20%）
#     plt.ylim(-0.2 * 100, 0.2 * 100)

#     plt.tight_layout()


#     if save_fig_path:
#         d = os.path.dirname(save_fig_path)
#         if d:
#             os.makedirs(d, exist_ok=True)
#         plt.savefig(save_fig_path, dpi=300)
#         print(f'✅ 图像已保存到 {save_fig_path}')

#     plt.show()

#     # 保存结果（可选）
#     if save_result_path:
#         d = os.path.dirname(save_result_path)
#         if d:
#             os.makedirs(d, exist_ok=True)
#         with open(save_result_path, 'wb') as f:
#             pkl.dump(
#                 {
#                     'P0': float(p0),
#                     'avg_prices': [float(x) for x in avg_prices],
#                     'inflation_rates': [float(x) for x in inflation_rates],  # 比例（非百分比）
#                 }, f
#             )
#         print(f'✅ 结果已保存到 {save_result_path}')

#     return avg_prices, inflation_rates
# if __name__ == '__main__':
#     dense_log_file = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\dense_log_240.pkl"
#     fig_file = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\result_analysis\inflation_rate_plot.png"
#     result_file = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\result_analysis\inflation_rate.pkl"

#     avg_prices, inflation_rates = compute_inflation_rate(dense_log_file, fig_file, result_file)
#     print("年度均价（前5）:", [f"{x:.3f}" for x in avg_prices[:5]])
#     print("年度通胀率%（前5）:", [f"{x*100:.2f}" for x in inflation_rates[:5]])
# unemployment_rate.py
# inflation_rate.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 让没有 ai_economist 也能加载 env_240.pkl
class DummyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("ai_economist"):
            return type(name, (), {})
        return super().find_class(module, name)

def load_env(path):
    with open(path, "rb") as f:
        return DummyUnpickler(f).load()

def compute_and_plot_inflation(env_pkl_path, save_fig_path=None, save_result_path=None):
    # 1) 读取 env_240.pkl
    env = load_env(env_pkl_path)

    # 2) 从 World 对象取价格序列（含初始 P0）
    prices = list(map(float, env.world.price))           # 长度通常 241 = P0 + 240 月
    if len(prices) < 13:
        raise ValueError(f"价格序列太短：{len(prices)}，至少得有 13（P0 + 12个月）")

    p0 = prices[0]                                       # 初始价格
    monthly_prices = np.array(prices[1:], dtype=float)   # 真实月份 1..T

    # 3) 对齐整年（12个月一组）
    remainder = len(monthly_prices) % 12
    if remainder != 0:
        print(f"⚠️ 月份数 {len(monthly_prices)} 不是 12 的整数倍，将截断最后 {remainder} 个月。")
        monthly_prices = monthly_prices[:-remainder]

    total_years = len(monthly_prices) // 12
    if total_years == 0:
        raise ValueError("没有完整的 12 个月，无法计算年度均价。")

    # 4) 计算年度均价 P̄_n
    avg_prices = [monthly_prices[i*12:(i+1)*12].mean() for i in range(total_years)]

    # 5) 年度通胀率：第1年相对 P0，其后相对上一年均价
    inflation_rates = []
    inflation_rates.append((avg_prices[0] - p0) / p0)
    for n in range(1, total_years):
        inflation_rates.append((avg_prices[n] - avg_prices[n-1]) / avg_prices[n-1])

    # 6) 画图（纵轴固定 -20%~20%）
    years = np.arange(1, total_years + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(years, np.array(inflation_rates) * 100.0, marker='o', linestyle='-')
    plt.title('Annual Inflation Rate')
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.xticks(years)
    plt.ylim(-0.2 * 100, 0.2 * 100)   # -20% ~ 20%
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_fig_path:
        d = os.path.dirname(save_fig_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_fig_path, dpi=300)
        print(f"✅ 图像已保存到 {save_fig_path}")

    plt.show()

    # 7) 保存结果（可选）
    if save_result_path:
        d = os.path.dirname(save_result_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(save_result_path, 'wb') as f:
            pickle.dump(
                {
                    'P0': float(p0),
                    'avg_prices': [float(x) for x in avg_prices],
                    'inflation_rates': [float(x) for x in inflation_rates],  # 比例（非百分比）
                },
                f
            )
        print(f"✅ 结果已保存到 {save_result_path}")

    return avg_prices, inflation_rates

if __name__ == "__main__":
    env_pkl = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\env_240.pkl"
    fig_out = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\result_analysis\inflation_rate_plot.png"
    result_out = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months\result_analysis\inflation_rate.pkl"

    avg_prices, inflation_rates = compute_and_plot_inflation(env_pkl, fig_out, result_out)
    print("年度均价（前5）:", [f"{x:.3f}" for x in avg_prices[:5]])
    print("年度通胀率%（前5）:", [f"{x*100:.2f}" for x in inflation_rates[:5]])
