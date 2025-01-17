import pandas as pd
import re
from pathlib import Path
def extract_dn(text):
    """从文本中提取 DN 后面跟随的数字部分"""
    if not isinstance(text, str):
        return ""
    
    # 正则表达式用于匹配 DN 后面跟随的数字部分
    dn_pattern = re.compile(r"DN(\d+)")
    match = dn_pattern.search(text)
    return f"DN{match.group(1)}" if match else text.strip()
def extract_non_chinese(text):
    """从文本中提取非汉字字符"""
    if not isinstance(text, str):
        return ""
    
    # 正则表达式用于匹配非汉字字符
    non_chinese_pattern = re.compile(r"[^\u4e00-\u9fff]+")
    matches = non_chinese_pattern.findall(text)
    
    # 连接所有非汉字字符并去除多余空白
    cleaned_text = ''.join(matches).strip()
    
    # 如果结果为空，则返回原始文本
    return cleaned_text if cleaned_text else text.strip()
def parse_description(description):
    # description = description.replace('\n', ' ').replace('\r', ' ')
    result = {}

    # 确保 description 是字符串
    if not isinstance(description, str) or not description.strip():
        print(f"非字符串或空的描述: {description}")
        return result

    # 正则表达式用于匹配 "1、名称: 内容" 和 "2、型号: 内容"
    # name_pattern = re.compile(r"1、名称\s*：\s*([^()（）]*?)(?=([()（）]|2、|$))", re.IGNORECASE)
    # model_pattern = re.compile(r"2、型号\s*：\s*(.*?)(?=3、|$)", re.IGNORECASE)
    # name_pattern = re.compile(r"1、类型：\s*：\s*([^()（）]*?)(?=([()（）]|2、|$))", re.IGNORECASE)
    name_pattern = re.compile(r"1、名称\s*：\s*(.*?)(?=2、|$)", re.IGNORECASE)
    # model_pattern = re.compile(r"型号\s*：\s*(DN\d+)", re.IGNORECASE)
    # model_pattern = re.compile(r"2、型号\s*：\s*(\d+[*xX]\d+)(?= N=)", re.IGNORECASE)
    model_pattern = re.compile(r"3、规格、压力等级\s*：\s*(.*?)(?=4、|$)", re.IGNORECASE)
    # 匹配名称
    name_match = name_pattern.search(description)
    if name_match:
        result["名称"] = name_match.group(1).strip()
        # print(f"匹配到的名称: {result['名称']}")
    # result["名称"] = "热镀锌无缝钢管"


    # 匹配型号
    model_match = model_pattern.search(description)
    if model_match:
        raw_model_text = model_match.group(1).strip()
        cleaned_model = extract_dn(raw_model_text)
        result["规格"] = cleaned_model
        if result.get("名称") == "闸阀":
            print(f"清理后的规格: {cleaned_model}")
        # print(f"清理后的规格: {cleaned_model}")

    return result


def process_excel_files(file_path_a, file_path_b, output_file_path):
    # 检查文件是否存在
    if not file_path_a.exists() or not file_path_b.exists():
        raise FileNotFoundError("提供的文件路径无效或文件不存在，请检查文件路径。")

    # 读取A表数据
    df_a = pd.read_excel(file_path_a, sheet_name="Sheet1", header=0)
    
    # 打印列名以确认
    # print("A表列名:", df_a.columns.tolist())

    # 读取B表数据，假设所有子表都有相同的结构
    excel_file_b = pd.ExcelFile(file_path_b)
    df_b = pd.concat([pd.read_excel(excel_file_b, sheet_name=sheet) for sheet in excel_file_b.sheet_names], ignore_index=True)

    # 打印列名以确认
    # print("B表列名:", df_b.columns.tolist())

    # 创建一个映射，方便快速查找
    contract_quantity_map = {}
    for _, row in df_b.iterrows():
        material_name = str(row.get('材料名称', ''))
        model = str(row.get('型号', '')).strip()
        # print(f"材料名称:{material_name}   型号:{model}")
        contract_quantity = row.get('合同数量')
        
        # 只添加有效的记录到映射中
        if material_name == "闸阀" and model and pd.notna(contract_quantity):
            contract_quantity_map[(material_name, extract_dn(model))] = contract_quantity
            print(f"添加到映射: 名称={material_name}, 型号={extract_dn(model)}, 合同数量={contract_quantity}")

    # 遍历A表并更新数据
    updated_data = []
    for index, row in df_a.iterrows():
            project_description = row.get('项目特征描述', '')
            
            # 如果 project_description 不是字符串或为空，则跳过该行
            if not isinstance(project_description, str) or not project_description:
                updated_data.append(dict(row))
                continue

            parsed_data = parse_description(project_description)
            name = parsed_data.get("名称")
            spec1 = parsed_data.get("规格")
            if name == "闸阀":
                print(f"从原始数据中获取到的名称:{name}   规格:{spec1}")

            if name == "闸阀":
                # cleaned_name = re.sub(r"\（[^）]*\）", "", name).strip()
                cleaned_name = name
            
                cleaned_spec1 = spec1
                if not spec1 is None:
                    print(f"从正则表达式中获取到的名称:{cleaned_name}   型号:{cleaned_spec1}")

                # 使用映射查找合同数量
                contract_quantity = contract_quantity_map.get((cleaned_name, cleaned_spec1))

                if contract_quantity is not None:
                    row_data = dict(row)
                    row_data['物资采购数量'] = contract_quantity  # 根据您的需求调整列名
                    updated_data.append(row_data)
                    print(f"成功匹配: 名称={cleaned_name}, 型号={cleaned_spec1.strip()}, 合同数量={contract_quantity}")
                else:
                    # print(f"未能找到与名称={cleaned_name} 和型号={cleaned_spec1} 完全匹配的记录")
                    updated_data.append(dict(row))
            else:
                # print(f"未能解析出有效的名称或型号: {project_description}")
                updated_data.append(dict(row))
       

    # 更新DataFrame并写回A表
    updated_df_a = pd.DataFrame(updated_data)
    updated_df_a.to_excel(output_file_path, index=False, sheet_name="Sheet1")
    print("开始回写A表————————————————————————————————————————————————————————————")

if __name__ == "__main__":
    file_path_b = Path("副本001表_更新.xlsx")
    file_path_a = Path("副本副本苏州a表新.xlsx")
    output_file_path = Path("数据23.xlsx")

    process_excel_files(file_path_a, file_path_b, output_file_path)