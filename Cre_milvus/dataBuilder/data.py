from concurrent.futures import ThreadPoolExecutor
from Cre_milvus.dataBuilder.tools.csvmake import process_csv
from Cre_milvus.dataBuilder.tools.mdmake import process_md
from Cre_milvus.dataBuilder.tools.pdfmake import process_pdf
from Cre_milvus.dataBuilder.tools.txtmake import process_txt

"""
数据处理
"""

def data_process(data_location, data_type, model_name, api_key, url_split):
    """
    data_location: 数据文件所在的文件夹的路径。图片数据请把图片的地址放到csv文件中。
    data_type: 文件类型
    model_name: 模型名称
    api_key: api_key
    url_split: 数据中是否存在url，如果存在url_split=True,否则url_split=False，提供url切分和提取功能
    """
    dataList = []
    # dataList的数据格式有两种情况，当url_split=True时，数据格式为[{'id': id, 'content': content, 'embedding': embedding, 'urls': [url1, url2, ...]}, ...]，否则数据格式为[(id,content,embedding), ...]。

    def process_file():
        if data_type == 'csv':
            return process_csv(
                csv_path=data_location,
                model_name=model_name,
                api_key=api_key
            )
        elif data_type == 'md':
            return process_md(
                md_dir_path=data_location,
                model_name=model_name,
                api_key=api_key,
                url_split=url_split
            )
        elif data_type == 'pdf':
            return process_pdf(
                pdf_path=data_location,
                model_name=model_name,
                api_key=api_key,
                url_split=url_split
            )
        elif data_type == 'txt':
            return process_txt(
                txt_path=data_location,
                model_name=model_name,
                api_key=api_key,
                url_split=url_split
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(process_file)
        dataList.append(future.result())

    return dataList