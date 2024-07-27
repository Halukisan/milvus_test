import os

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = '../data_base/knowledge_db/pumkin_book'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
# print(file_paths[:3])

from langchain.document_loaders.pdf import PyMuPDFLoader
# from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:

    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    # elif file_type == 'md':
    #     loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())

# 载入后的变量类型为langchain_core.documents.base.Document, 文档变量类型同样包含两个属性

# page_content 包含该文档的内容。
# meta_data 为文档相关的描述性数据。

text = texts[1]
# print(f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)



from pymilvus.model.hybrid import BGEM3EmbeddingFunction
 
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',  # 指定模型名称
    device='cpu',  # 指定使用的设备，例如'cpu'或'cuda:0'
    use_fp16=False  # 指定是否使用fp16。如果`device`是`cpu`，则设置为'False'。
)

docs = [
    "人工智能学科成立于1956年。",
    "艾伦·图灵是第一个在人工智能领域进行大量研究的人。",
    "图灵出生于伦敦迈达维尔，成长于英格兰南部。",
]
 
docs_embeddings = bge_m3_ef.encode_documents(docs)
 
# 打印嵌入
print("嵌入：", docs_embeddings)
# 打印密集嵌入的维度
print("密集文档维度：", bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape)
# 由于稀疏嵌入以2D csr_array格式存在，我们将它们转换为列表以便更容易处理。
print("稀疏文档维度：", bge_m3_ef.dim["sparse"], list(docs_embeddings["sparse"])[0].shape)