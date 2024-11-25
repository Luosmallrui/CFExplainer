from pathlib import Path
import torch

def analyze_contract_data(file_path):
    try:
        data = torch.load(file_path)
        print("\n=== 数据结构分析 ===")
        
        if 'SCcfg' in data:
            cfg_data = data['SCcfg']
            print("\nSCcfg 的 tuple 结构:")
            if isinstance(cfg_data.get('SCcfg'), tuple):
                tuple_data = cfg_data['SCcfg']
                print(f"Tuple 长度: {len(tuple_data)}")
                
                # 分析元组中的每个元素
                for i, item in enumerate(tuple_data):
                    print(f"\n元素 {i}:")
                    print(f"类型: {type(item)}")
                    
                    if isinstance(item, dict):
                        print("字典键:", item.keys())
                        # 显示第一个键值对
                        if item:
                            first_key = next(iter(item))
                            print(f"第一个键值对 ({first_key}):", item[first_key])
                    
                    elif isinstance(item, list):
                        print(f"列表长度: {len(item)}")
                        if item:
                            print("第一个元素:", item[0])
                    
                    elif isinstance(item, (int, float, str, bool)):
                        print("值:", item)
                    
                    else:
                        print("类型:", type(item))
                        try:
                            print("内容示例:", str(item)[:200])  # 只显示前200个字符
                        except:
                            print("无法显示内容")

        # 漏洞信息
        print("\n漏洞标签:")
        print("Vulnerability list:", data['vulnerability_list'])
        print("Label dictionary:", data['label'])

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        traceback.print_exc()

# 测试代码
def test_dataset_structure():
    data_dir = "/root/autodl-tmp/SCcfg/processed"
    file_list = list(Path(data_dir).glob("*.pt"))
    
    if not file_list:
        print("未找到数据文件！")
        return
    
    print(f"找到 {len(file_list)} 个数据文件")
    
    # 分析第一个有效文件
    for file_path in file_list[:2]:  # 分析前两个文件
        print(f"\n分析文件: {file_path}")
        analyze_contract_data(file_path)

# 运行测试
if __name__ == "__main__":
    import traceback
    test_dataset_structure()