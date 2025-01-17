import pandas as pd
import re
from pathlib import Path

class ARecord:
    def __init__(self, sequence_number=None, project_description=None, purchase_quantity_contract1=None):
        self.sequence_number = sequence_number
        self.project_description = project_description
        self.purchase_quantity_contract1 = purchase_quantity_contract1

class BRecord:
    def __init__(self, serial_number=None, material_name=None, model=None, contract_quantity=None):
        self.serial_number = serial_number
        self.material_name = material_name
        self.model = model
        self.contract_quantity = contract_quantity

def parse_description(description):
    result = {}
    
    name_pattern = re.compile(r"1、名称\s*：\s*([^（]*?)(?=\d{1}2|\\s|$)", re.IGNORECASE)
    model_pattern = re.compile(r"2、型号\s*：\s*([^（]*?)(?=\d{1}3|\\s|$)", re.IGNORECASE)

    name_match = name_pattern.search(description)
    if name_match:
        result["名称"] = name_match.group(1).strip()

    model_match = model_pattern.search(description)
    if model_match:
        result["型号"] = model_match.group(1).strip()

    return result

def process_excel_files(file_path_a, file_path_b, output_file_path):
    # 检查文件是否存在
    if not file_path_a.exists() or not file_path_b.exists():
        raise FileNotFoundError("提供的文件路径无效或文件不存在，请检查文件路径。")

    # 读取A表数据，不指定sheet_name以获取所有工作表名称
    excel_file_a = pd.ExcelFile(file_path_a)
    print(f"文件 {file_path_a} 包含的工作表有：{excel_file_a.sheet_names}")

    sheet_names_a = excel_file_a.sheet_names
    if "Sheet1" not in sheet_names_a:
        print(f"警告：文件 {file_path_a} 中未找到名为 'Sheet1' 的工作表。将使用第一个工作表。")
        sheet_name_a = sheet_names_a[0] if sheet_names_a else None
    else:
        sheet_name_a = "Sheet1"
    
    df_a = pd.read_excel(excel_file_a, sheet_name=sheet_name_a, header=0)
    all_records_a = [
        ARecord(
            sequence_number=row.get('序号', None),
            project_description=row.get('项目特征描述', None),
            purchase_quantity_contract1=row.get('采购合同1（合同编号）', None)
        ) for _, row in df_a.iterrows()
    ]

    # 读取B表数据
    df_b = pd.read_excel(file_path_b, sheet_name=None)  # 如果有多个子表，这里会读取所有子表
    all_records_b = []
    for sheet_name, sheet_df in df_b.items():
        all_records_b.extend([
            BRecord(
                serial_number=row.get('序号', None),
                material_name=row.get('材料名称', None),
                model=row.get('型号', None),
                contract_quantity=row.get('合同数量', None)
            ) for _, row in sheet_df.iterrows()
        ])

    # 数据匹配逻辑
    for record_a in all_records_a:
        if not record_a.sequence_number or not record_a.project_description:
            continue
        
        parsed_data = parse_description(record_a.project_description)
        name = parsed_data.get("名称")
        spec1 = parsed_data.get("型号")

        if not spec1 or not name:
            continue

        cleaned_name = re.sub(r"\（[^）]*\）", "", name)

        for record_b in all_records_b:
            if record_b.material_name == cleaned_name.strip() and record_b.model == spec1:
                record_a.purchase_quantity_contract1 = record_b.contract_quantity
                break

    # 更新DataFrame并写回A表
    updated_df_a = pd.DataFrame([vars(record) for record in all_records_a])
    updated_df_a.to_excel(output_file_path, index=False, sheet_name=sheet_name_a if sheet_name_a else "Sheet1")
    print("开始回写A表————————————————————————————————————————————————————————————")

if __name__ == "__main__":
    file_path_b = Path("副本001b表.xlsx")
    file_path_a = Path("工作簿1 - 副本.xlsx")
    output_file_path = Path("副本苏州a表_更新.xlsx")

    process_excel_files(file_path_a, file_path_b, output_file_path)