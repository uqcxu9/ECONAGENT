import pickle as pkl
import sys
import pprint
# view_pkl.py 顶部加
import sys
sys.path.insert(0, r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\original_code_gpt4\ACL24-EconAgent")


def view_pkl(file_path):
    """查看pickle文件内容"""
    print(f"\n{'='*60}")
    print(f"文件: {file_path}")
    print(f"{'='*60}\n")
    
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    
    print(f"数据类型: {type(data)}\n")
    
    if isinstance(data, dict):
        print(f"字典大小: {len(data)} 个键")
        print(f"键列表: {list(data.keys())}\n")
        
        # 显示所有键值对（因为只有5个agent）
        print("所有键值对:")
        for i, (k, v) in enumerate(data.items()):
            print(f"\n[{i+1}] Agent ID: {k}")
            print(f"    值类型: {type(v)}")
            if isinstance(v, list) and len(v) == 2:
                work_decision = "工作" if v[0] == 1 else "不工作"
                consumption_pct = v[1] * 0.02 * 100  # 转换回百分比
                print(f"    工作决策: {v[0]} ({work_decision})")
                print(f"    消费动作: {v[1]} (消费{consumption_pct:.0f}%的总资产)")
            elif isinstance(v, (list, dict)):
                print(f"    值长度: {len(v)}")
                if len(str(v)) < 200:
                    print(f"    值内容: {v}")
                else:
                    print(f"    值内容(前200字符): {str(v)[:200]}...")
            else:
                print(f"    值: {v}")
    
    elif isinstance(data, list):
        print(f"列表长度: {len(data)}")
        if len(data) > 0:
            print(f"第一个元素类型: {type(data[0])}\n")
            print("前3个元素:")
            for i, item in enumerate(data[:3]):
                print(f"\n[{i+1}] {item}")
    
    else:
        print(f"数据内容:\n{data}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python view_pkl.py <pkl文件路径>")
        print("\n可用的pkl文件:")
        print("  - data\\gpt-3-noperception-reflection-1-5agents-3months\\actions_3.pkl")
        print("  - data\\gpt-3-noperception-reflection-1-5agents-3months\\obs_3.pkl")
        print("  - data\\gpt-3-noperception-reflection-1-5agents-3months\\env_3.pkl")
        print("  - data\\gpt-3-noperception-reflection-1-5agents-3months\\dialog_3.pkl")
        print("  - data\\gpt-3-noperception-reflection-1-5agents-3months\\dense_log_3.pkl")
    else:
        view_pkl(sys.argv[1])

