from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from Cre_milvus.dataBuilder.tools.csvmake import process_csv
from Cre_milvus.dataBuilder.tools.mdmake import process_md
from Cre_milvus.dataBuilder.tools.pdfmake import process_pdf
from Cre_milvus.dataBuilder.tools.txtmake import process_txt

def data_process(data_location, model_name, api_key, url_split):
    """
    data_location: 用户上传的文件夹路径
    model_name: 嵌入模型名称
    api_key: api_key
    url_split: 是否对文本做url切分
    自动识别文件夹下的csv、md、pdf、txt文件并多线程处理
    """
    dataList = []

    # 扫描文件夹下所有支持的文件
    folder = Path(data_location)
    files = list(folder.glob("**/*"))
    tasks = []

    def get_type_and_path(file_path):
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return "csv", str(file_path)
        elif suffix == ".md":
            return "md", str(file_path)
        elif suffix == ".pdf":
            return "pdf", str(file_path)
        elif suffix == ".txt":
            return "txt", str(file_path)
        else:
            return None, None

    # 构建任务
    for file in files:
        file_type, file_path = get_type_and_path(file)
        if not file_type:
            continue
        if file_type == "csv":
            tasks.append(("csv", file_path))
        elif file_type == "md":
            tasks.append(("md", file_path))
        elif file_type == "pdf":
            tasks.append(("pdf", file_path))
        elif file_type == "txt":
            tasks.append(("txt", file_path))

    # 多线程处理
    def process_one(task):
        file_type, file_path = task
        if file_type == "csv":
            return process_csv(csv_path=file_path, model_name=model_name, api_key=api_key)
        elif file_type == "md":
            return process_md(md_dir_path=file_path, model_name=model_name, api_key=api_key, url_split=url_split)
        elif file_type == "pdf":
            return process_pdf(pdf_path=file_path, model_name=model_name, api_key=api_key, url_split=url_split)
        elif file_type == "txt":
            return process_txt(txt_path=file_path, model_name=model_name, api_key=api_key, url_split=url_split)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_one, tasks))
        for res in results:
            if isinstance(res, list):
                dataList.extend(res)
            elif res is not None:
                dataList.append(res)

    return dataList