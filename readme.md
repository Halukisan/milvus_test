# 如何开始
milvus_create目录下存放的是关键代码，ready.py和zhipuai_embedding.py这俩没有什么用，可以直接忽视，start.py为主要代码，embedding.py为测试代码，在实际运行中无关紧要，类似于用于学习的代码。
在进行向量查询之前，你需要向Milvus中导入数据（如果以及导入了，可以忽略，直接启动start.py）,
## 导入数据
这部分代码在data_maker中，mdmake用于将md文档插入到milvus中
## 可能出现的问题
插入过程中报错
> pymilvus.exceptions.ParamError: <ParamError: (code=1, message=invalid input, length of string exceeds max length. length: 499, max length: 256)
这表示数据长度超出了构建向量数据库时规定的长度，你需要重新构建向量数据库，打开csdn_milvus目录，进入create_milvus2.py中，修改对应行的最大长度要求，然后运行，然后运行create_index.py

