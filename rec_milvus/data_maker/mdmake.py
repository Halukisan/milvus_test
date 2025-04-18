from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os


def clean_insert(content):
    md_page = content
    # print(f"每一个元素的类型：{type(pdf_page)}.", 
    #     f"该文档的描述性数据：{pdf_page.metadata}", 
    #     f"查看该文档的内容:\n{pdf_page.page_content}", 
    #     sep="\n------\n")

    import re
    # 开始数据清洗：可以看到上文中读取的pdf文件不仅将一句话按照原文的分行添加了换行符\n，也在原本两个符号中间插入了\n，我们可以使用正则表达式匹配并删除掉\n。
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    md_page = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), md_page)
    # print(pdf_page.page_content)
    # 进一步分析，发现数据中还有很多•和空格，我们的简单实用replace方法即可。
    md_page = md_page.replace('•', '')
    md_page = md_page.replace(' ', '')
    md_page = md_page.replace('*', '')
    md_page = md_page.replace('\n\n','\n')
    print("在这呢")
    print(f"md_page的内容为:{md_page}")
    
    # 分割数据
    ''' 
    * RecursiveCharacterTextSplitter 递归字符文本分割
    RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
        这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
    RecursiveCharacterTextSplitter需要关注的是4个参数：

    * separators - 分隔符字符串数组
    * chunk_size - 每个文档的字符数量限制
    * chunk_overlap - 两份文档重叠区域的长度
    * length_function - 长度计算函数
    '''
    #导入文本分割器
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # 知识库中单段文本长度
    CHUNK_SIZE = 500

    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 50
    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    split_docs = text_splitter.split_text(md_page[0:2000])

    # split_docs = text_splitter.split_documents(pdf_page.page_content[0:1000])
    # print(f"切分后的文件数量：{len(split_docs)}")
    # print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
    # 将分割后的文本放入数组中

    # 确保docs列表不为空，并且每个元素都是非空字符串
    docs = [doc for doc in split_docs if doc]

    # print(f"docs的内容为：{docs}")


    from zhipuai import ZhipuAI

    # 替换为您的实际 API 密钥
    api_key = ""

    # 初始化 ZhipuAI 客户端
    client = ZhipuAI(api_key=api_key)


    # 准备输入文本
    texts  = docs

    from pymilvus import Collection, db, connections
    import numpy as np
    
    conn = connections.connect(host="127.0.0.1", port=19530)
    db.using_database("sample_db")
    coll_name = 'word_vector'
    # 加载集合，不加载就不能插入数据
    collection = Collection(coll_name)
    collection.load()
    # mids, embedings, counts, descs = [], [], [], []
    embedings, descs = [], []

    # idx = 1
    # 循环处理每个文本
    for text in texts:
        # 调用嵌入创建 API
        response = client.embeddings.create(
            model="embedding-2",  # 替换为您想要使用的模型名称
            input=text  # 传递单个文本
        )
        print(f"原文：{text},response：{response}")
        
        # mids.append(idx)
        embedings.append(response.data[0].embedding)
        descs.append(text)
        # counts.append()
        # idx+=1
        # print("mids 的长度：", len(mids))
        print("embedings 的长度：", len(embedings))
        print("descs 的长度：", len(descs))
        # print("counts 的长度：", len(counts))


    collection = Collection(coll_name)
    # mr = collection.insert([mids,embedings,descs,counts])
    mr = collection.insert([embedings,descs])

    print(mr)
    # print(f"最后数据的id为{idx}")


# 指定要读取的 Markdown 文件路径
markdown_file_path = "../data_base/knowledge_db/video_Test_Data"
# # 检查文件是否存在
# if os.path.exists(markdown_file_path):
#     # 打开并读取文件内容
#     with open(markdown_file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     # print(content)
# else:
#     print(f"文件 {markdown_file_path} 不存在")

# 列出目录中的所有文件
files_in_directory = os.listdir(markdown_file_path)
fin_content = ""    
    # 过滤出所有的Markdown文件
markdown_files = [file for file in files_in_directory if file.endswith('.md')]
    # 读取每个Markdown文件的内容
for markdown_file in markdown_files:
    file_path = os.path.join(markdown_file_path, markdown_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # print(content)
        # fin_content+=content
        clean_insert(content)

# print(fin_content)
# clean_insert(fin_content)
        









