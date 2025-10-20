"""
查看第一个月有多少agent选择工作
"""
import pickle
import numpy as np

# 方法1: 从 dense_log 中查看第一个月的就业情况
def check_from_dense_log(dense_log_path):
    print("=" * 60)
    print("方法1: 从 dense_log 读取第一个月就业情况")
    print("=" * 60)
    
    with open(dense_log_path, 'rb') as f:
        dense_log = pickle.load(f)
    
    # dense_log['actions'] 是一个列表，每个元素是一个字典
    # actions[0] 是第1个月的动作
    if len(dense_log['actions']) > 0:
        first_month_actions = dense_log['actions'][0]  # 第1个月（索引0）
        
        # 统计选择工作的agent数量
        # action[0] = 1 表示选择工作，= 0 表示不工作
        work_count = 0
        unemployed_count = 0
        
        for agent_id, action in first_month_actions.items():
            if agent_id == 'p':  # 跳过planner
                continue
            
            work_action = action[0]  # 第一个元素是工作决策 (0或1)
            if work_action == 1:
                work_count += 1
            else:
                unemployed_count += 1
        
        total_agents = work_count + unemployed_count
        employment_rate = (work_count / total_agents) * 100 if total_agents > 0 else 0
        unemployment_rate = (unemployed_count / total_agents) * 100 if total_agents > 0 else 0
        
        print(f"\n📊 第1个月统计:")
        print(f"   总agent数: {total_agents}")
        print(f"   ✅ 选择工作: {work_count} ({employment_rate:.1f}%)")
        print(f"   ❌ 选择不工作: {unemployed_count} ({unemployment_rate:.1f}%)")
        print()
        
        # 显示每个agent的决策
        print("详细决策 (agent_id: [工作决策, 消费决策]):")
        for agent_id, action in sorted(first_month_actions.items(), key=lambda x: int(x[0]) if x[0] != 'p' else -1):
            if agent_id == 'p':
                continue
            work_status = "✅ 工作" if action[0] == 1 else "❌ 不工作"
            consumption = action[1]
            print(f"   Agent {agent_id:>3}: {work_status}, 消费档位={consumption}")
        
        return work_count, unemployed_count, total_agents


# 方法2: 从 env 中查看第一个月的就业情况
def check_from_env(env_path):
    print("\n" + "=" * 60)
    print("方法2: 从 env_6.pkl 读取前6个月的就业情况")
    print("=" * 60)
    
    # 让没有 ai_economist 也能加载
    class DummyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("ai_economist"):
                return type(name, (), {})
            return super().find_class(module, name)
    
    with open(env_path, "rb") as f:
        env = DummyUnpickler(f).load()
    
    # env.dense_log['actions'] 包含所有月份的动作
    actions = env.dense_log['actions']
    
    print(f"\n总共有 {len(actions)} 个月的数据\n")
    
    # 查看前3个月
    for month_idx in range(min(3, len(actions))):
        month_actions = actions[month_idx]
        
        work_count = 0
        unemployed_count = 0
        
        for agent_id, action in month_actions.items():
            if agent_id == 'p':
                continue
            if action[0] == 1:
                work_count += 1
            else:
                unemployed_count += 1
        
        total = work_count + unemployed_count
        employment_rate = (work_count / total) * 100 if total > 0 else 0
        
        print(f"第 {month_idx + 1} 个月: ✅ {work_count} 人工作 ({employment_rate:.1f}%), ❌ {unemployed_count} 人不工作")


# 方法3: 从 actions 文件查看
def check_from_actions(actions_path):
    print("\n" + "=" * 60)
    print("方法3: 从 actions_6.pkl 读取第6个月的动作")
    print("=" * 60)
    
    with open(actions_path, 'rb') as f:
        actions = pickle.load(f)
    
    # actions 是一个字典，键是 agent_id
    work_count = 0
    unemployed_count = 0
    
    for agent_id, action in actions.items():
        if agent_id == 'p':
            continue
        if action[0] == 1:
            work_count += 1
        else:
            unemployed_count += 1
    
    total = work_count + unemployed_count
    employment_rate = (work_count / total) * 100 if total > 0 else 0
    
    print(f"\n📊 第6个月统计:")
    print(f"   总agent数: {total}")
    print(f"   ✅ 选择工作: {work_count} ({employment_rate:.1f}%)")
    print(f"   ❌ 选择不工作: {unemployed_count} ({employment_rate:.1f}%)")


if __name__ == "__main__":
    import os
    
    # 设置路径
    base_path = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent\data\gpt-3-noperception-reflection-1-100agents-240months"
    
    dense_log_path = os.path.join(base_path, "dense_log.pkl")
    env_path = os.path.join(base_path, "env_6.pkl")
    actions_path = os.path.join(base_path, "actions_6.pkl")
    
    # 运行分析
    if os.path.exists(dense_log_path):
        check_from_dense_log(dense_log_path)
    else:
        print(f"⚠️ 找不到文件: {dense_log_path}")
    
    if os.path.exists(env_path):
        check_from_env(env_path)
    else:
        print(f"⚠️ 找不到文件: {env_path}")
    
    if os.path.exists(actions_path):
        check_from_actions(actions_path)
    else:
        print(f"⚠️ 找不到文件: {actions_path}")

