import re
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
    result = {}
    
    # 确保 description 是字符串
    if not isinstance(description, str) or not description.strip():
        print(f"非字符串或空的描述: {description}")
        return result
    # 定义正则表达式模式
    name_pattern = re.compile(r"1、名称\s*：\s*(.*?)(?=2、|$)", re.IGNORECASE)
    # model_pattern = re.compile(r"型号\s*：\s*(DN\d+)", re.IGNORECASE)
    # model_pattern = re.compile(r"2、型号\s*：\s*(\d+[*xX]\d+)(?= N=)", re.IGNORECASE)
    model_pattern = re.compile(r"3、规格、压力等级\s*：\s*(.*?)(?=4、|$)", re.IGNORECASE)

    name_match = name_pattern.search(description)
    if name_match:
        result["名称"] = name_match.group(1).strip()
        print(f"匹配到的名称: {result['名称']}")


    # 匹配型号
    model_match = model_pattern.search(description)
    if model_match:
        raw_model_text = model_match.group(1).strip()
        # 二次筛选，去除汉字
        cleaned_model = extract_non_chinese(raw_model_text)
        result["规格、压力等级"] = cleaned_model
        print(f"清理后的规格、压力等级: {cleaned_model}")
    return result


# 示例测试字符串
test_description = "1、名称：闸阀 2、材质：铜芯球墨铸铁外壳 3、规格、压力等级：DN65 4、连接形式：法兰连接 5、其他:详见设计图纸及招标文件"

# 测试解析函数
parsed_data = parse_description(test_description)
print(parsed_data)

# 800*500 N=120W，220V
# 800*500 N=120W，220V