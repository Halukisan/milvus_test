import pandas as pd
import re
from pathlib import Path

def parse_description(description):
    result = {}
    
    # 确保 description 是字符串
    if not isinstance(description, str) or not description.strip():
        print(f"非字符串或空的描述: {description}")
        return result

    # 正则表达式用于匹配 "1、名称: 内容" 和 "2、型号: 内容"
    name_pattern = re.compile(r"1、名称\s*：\s*([^（]*?)(?=\d{1}2|\\s|$)", re.IGNORECASE)
    model_pattern = re.compile(r"2、型号\s*：\s*([^（]*?)(?=\d{1}3|\\s|$)", re.IGNORECASE)

    # 匹配名称
    name_match = name_pattern.search(description)
    if name_match:
        result["名称"] = name_match.group(1).strip()
        print(f"匹配到的名称: {result['名称']}")

    # 匹配型号
    model_match = model_pattern.search(description)
    if model_match:
        result["型号"] = model_match.group(1).strip()
        print(f"匹配到的型号: {result['型号']}")

    return result

def process_excel_files(file_path_a, file_path_b, output_file_path):
    # 检查文件是否存在
    if not file_path_a.exists() or not file_path_b.exists():
        raise FileNotFoundError("提供的文件路径无效或文件不存在，请检查文件路径。")

    # 读取A表数据
    df_a = pd.read_excel(file_path_a, sheet_name="Sheet1", header=0)
    
    # 打印列名以确认
    print("A表列名:", df_a.columns.tolist())

    # 读取B表数据，假设所有子表都有相同的结构
    excel_file_b = pd.ExcelFile(file_path_b)
    df_b = pd.concat([pd.read_excel(excel_file_b, sheet_name=sheet) for sheet in excel_file_b.sheet_names], ignore_index=True)

    # 打印列名以确认
    print("B表列名:", df_b.columns.tolist())

    # 遍历A表并更新数据
    updated_data = []
    for index, row in df_a.iterrows():
        project_description = row.get('项目特征描述', '')
        
        # 如果 project_description 不是字符串或为空，则跳过该行
        if not isinstance(project_description, str) or not project_description.strip():
            updated_data.append(dict(row))
            continue

        parsed_data = parse_description(project_description)
        name = parsed_data.get("名称")
        spec1 = parsed_data.get("型号")

        if name and spec1:
            cleaned_name = re.sub(r"\（[^）]*\）", "", name).strip().lower()
            cleaned_spec1 = spec1.strip().lower()

            # 查找所有名称匹配的记录
            matching_records = df_b[(df_b['材料名称'].str.lower() == cleaned_name)]
            
            # 如果有匹配的记录，再根据型号进一步筛选
            if not matching_records.empty:
                exact_match = matching_records[matching_records['型号'].str.lower() == cleaned_spec1]

                if not exact_match.empty:
                    contract_quantity = exact_match.iloc[0]['合同数量']
                    row_data = dict(row)
                    row_data['采购合同1（合同编号）'] = contract_quantity
                    updated_data.append(row_data)
                    print(f"成功匹配: 名称={cleaned_name}, 型号={cleaned_spec1}, 合同数量={contract_quantity}")
                else:
                    print(f"未能找到与名称={cleaned_name} 和型号={cleaned_spec1} 完全匹配的记录")
                    updated_data.append(dict(row))
            else:
                print(f"未能找到与名称={cleaned_name} 匹配的记录")
                updated_data.append(dict(row))
        else:
            print(f"未能解析出有效的名称或型号: {project_description}")
            updated_data.append(dict(row))

    # 更新DataFrame并写回A表
    updated_df_a = pd.DataFrame(updated_data)
    updated_df_a.to_excel(output_file_path, index=False, sheet_name="Sheet1")
    print("开始回写A表————————————————————————————————————————————————————————————")

if __name__ == "__main__":
    file_path_b = Path("副本001表_更新.xlsx")
    file_path_a = Path("副本苏州a表_更新.xlsx")
    output_file_path = Path("数据2.xlsx")

    process_excel_files(file_path_a, file_path_b, output_file_path)