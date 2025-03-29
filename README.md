### data_base/knowledge_db
该目录下存放着用于向量化存储的数据，包含md，word，pdf等文件。
### data_maker
该目录下是通用文件处理代码，可以一次性获取所有的md文件或者pdf文件并读取内容进行格式化。
### csdn_milvus
该目录下存放着最基础的向量数据库的构建和数据处理，如果不了解milvus，先看这个文件下的代码。
### milvus_create 
该目录下存放着结合构建milvus和rag的代码
### video_url_test
该目录下存放的代码解决了如下问题

1. 想用rag实现一个视频推送功能，用户提问，然后大模型给出文字回答和相关文字解释视频的播放地址，在灌库之前应该怎么处理，直接让播放地址和文字一起，播放地址可能就会被切断。就是切割的时候会把视频的url 切成两半，然后输出的url地址打不开。
2. milvus的结构设计，有利于后期的优化处理
3. 快速便捷的批量读取各种类型文件数据并且自动获取其中的url并结合文字描述，合理地存储到milvus中
4. 用户询问一些专业知识，可以根据milvus里面获取到数据并且返回给我相应的url，但是，我如果问你好，或者哇哦等普通词语，他也会随便给我返回url。解决方案：根据召回分数进行筛选数据：这种闲聊召回的数据分数都不高，设定一个min score，召回的top k中的每条数据至少超过这个min score，才能融合到prompt中
5. gradio编写的极为简单的页面

后期扩展内容

1. 优化页面展示（url和文字分开的更加优美）
2. 增加functionCalling功能，优化数据的获取与展示
3. 待定......


### milvus_plus

参考推荐系统fun-rec中的余弦相似度分类数据得到的如下思路：

* 基于k-mean聚合后得到中心标准向量
* 自动读取文档内容
* 责任链模式进行匹配
* 分块存储数据，提高查询效率

使用步骤：
    首先init.py文件中，计算中心标准向量，这里的文件名称需要规范，因为下面的代码是写死的，这个文件要单独执行，主要就是计算中心标准向量，其他的都不做。对于这个示例代码，我需要存放到文件夹中的md文件名称需要是java、mysql、redis、web。

    此时我们获得到了中心标准向量，然后就可以进行下面的操作了，在create.py文件中，可以进行如下操作。读取所有你需要进行向量化存储的文件（这里不用规范名称了，只要是md文件都行），进入责任链，然后存储milvus中。
    至此，分块存储完成（详细请看代码，有注释）
    
    start.py中包含启动函数，运行之前，请手动把中心标准向量复制到里面，（此处不宜设置为自动处理，手动更好）。
    对于用户问题，同样的流程，向量化后进责任链，确定数据所处分块的位置后，取前五条最相似的数据进行相似度匹配，大于预设的值的可以作为参考数据，写入prompt中。


优化：
    分块存储后，模型的回答可以做优化，比如加入CoT，提高模型回答准确度。
### text_search_pic
使用文字搜索图片，使用了toWhere，向量化使用的是openai的clip-vit-base-patch16



## 相关技术介绍

### Towhere

Towhee 是一个开源的 **多模态数据处理框架**，专注于高效生成非结构化数据（如文本、图像、音频、视频等）的向量表示（Embeddings），并支持构建端到端的 AI 流水线（Pipeline）。它旨在简化从原始数据到向量化表示再到实际应用（如搜索、推荐、问答系统）的开发流程，尤其适用于需要处理多模态数据的场景。

---

### **一、Towhee 的核心功能**
1. **多模态 Embedding 生成**  
   - 支持文本、图像、音频、视频等非结构化数据的向量化。
   - 内置丰富的预训练模型（如 BERT、CLIP、ViT、ResNet、Whisper 等），可直接调用。
   - 支持自定义模型集成，灵活适配业务需求。

2. **流水线（Pipeline）构建**  
   - 提供声明式 API，通过链式调用快速组合数据处理步骤（如数据加载、预处理、模型推理、后处理等）。
   - 示例：一个图像搜索流水线可以包含 `图像解码 → 特征提取 → 向量归一化 → 存储到向量数据库`。

3. **高性能与可扩展性**  
   - 支持批量处理（Batch Processing）和 GPU 加速。
   - 分布式计算能力，适合大规模数据处理。
   - 通过算子（Operator）机制，可灵活扩展新功能。

4. **与向量数据库无缝集成**  
   - 深度兼容 Milvus、Elasticsearch、FAISS 等向量数据库，简化数据存储与检索流程。

---

### **二、Towhee 的架构**
Towhee 的架构围绕 **Operator** 和 **Pipeline** 设计：
1. **Operator（算子）**  
   - 原子化数据处理单元，每个 Operator 完成单一任务（如 `image_decode`、`text_embedding`）。
   - 分为两类：
     - **Hub Operators**：官方预置的算子，开箱即用。
     - **Custom Operators**：用户自定义算子，支持 Python 编写。

2. **Pipeline（流水线）**  
   - 通过连接多个 Operator 构建端到端的数据处理流程。
   - 支持并行执行、条件分支等复杂逻辑。
   - 示例代码：
     ```python
     from towhee import pipe, ops

     image_search_pipeline = (
         pipe.input('image_path')
             .map('image_path', 'image', ops.image_decode())
             .map('image', 'embedding', ops.image_embedding(model_name='clip_vit_base_patch32'))
             .output('embedding')
     )
     ```

3. **执行引擎**  
   - 自动优化计算图（如算子融合、并行调度）。
   - 支持同步/异步执行模式。

---

### **三、应用场景**
1. **跨模态搜索**  
   - 如“以图搜图”、“以文搜图”、“语音搜索”等。
   - 示例：用 CLIP 模型将文本和图像映射到同一向量空间，实现跨模态检索。

2. **推荐系统**  
   - 生成用户和物品的 Embedding，计算相似度进行推荐。

3. **问答系统（QA）**  
   - 将问题和文档编码为向量，通过语义匹配找到最佳答案。

4. **内容理解与分类**  
   - 对视频、音频进行内容分析（如标签生成、情感分析）。

---

### **四、快速入门示例**
1. **安装 Towhee**
   ```bash
   pip install towhee
   ```

2. **生成文本 Embedding**
   ```python
   import towhee

   text_embedding = (
       towhee.text('Hello, Towhee!')
           .text_embedding.transformers(model_name='bert-base-uncased')
           .to_list()[0]
   )
   print(text_embedding.shape)  # 输出: (768,)
   ```

3. **构建图像搜索流水线**
   ```python
   from towhee import pipe, ops

   pipeline = (
       pipe.input('url')
           .map('url', 'image', ops.image_decode())
           .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
           .output('vec')
   )

   result = pipeline('https://example.com/image.jpg')
   ```

---

### **五、Towhee 的优势与特点**
- **多模态统一接口**：一套 API 处理文本、图像、视频等多种数据类型。
- **开箱即用**：预集成 100+ 模型和算子，覆盖主流算法（如 OpenAI CLIP、Sentence-BERT）。
- **灵活性**：支持自定义算子与流水线，适配私有模型和业务逻辑。
- **生产就绪**：支持微服务部署（如 Docker、Kubernetes），提供 RESTful API 封装。

---

### **六、与其他工具的对比**
| 工具                            | 定位               | 特点                    |
| ----------------------------- | ---------------- | --------------------- |
| **Towhee**                    | 多模态 Embedding 框架 | 端到端流水线、多模态支持          |
| **Hugging Face Transformers** | 单模态模型库           | 文本/图像模型丰富，但需自行构建流程    |
| **Milvus**                    | 向量数据库            | 专注向量存储与检索，与 Towhee 互补 |
| **LangChain**                 | 大语言模型应用开发框架      | 侧重文本任务链式调用，多模态能力弱     |

---

### **七、社区与资源**
- **GitHub**: [https://github.com/towhee-io/towhee](https://github.com/towhee-io/towhee)
- **文档**: [https://towhee.io](https://towhee.io)
- **案例教程**: 提供跨模态搜索、推荐系统等实战示例。

---

### **总结**
Towhee 通过简化多模态数据处理流程，降低了 AI 应用的开发门槛。无论是快速验证原型还是部署生产系统，它都能提供高效灵活的解决方案。如果你需要处理复杂的非结构化数据并生成高质量的 Embedding，Towhee 是一个值得尝试的工具。