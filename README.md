# 项目介绍

## rec_milvus
包含以下内容：
1. 数据集
2. text_search_pic文字搜索图片
3. data_maker该目录下是通用文件处理代码，
4. video_url_test url存储与切分
5. milvus_gpu 启动GPU索引
6. milvus_multi_recall多路召回：Milvus+Elasticsearch+function_calling+text_search_pic功能

两种聚类方法
1. milvus_plus：基于k-mean聚合
2. milvus_HDBSCAN：基于密度的聚类算法





## Need to know
### 构建数据集要求
问题：噪声、重复、低质数据会污染知识库，导致检索到无关内容。
解决方案：
1. 清洗数据：去除HTML标签、特殊符号、乱码等噪声。
2. 去重：合并相似内容，避免冗余数据干扰检索。
3. 标准化：统一文本格式（如日期、单位）、大小写、标点符号。
4. 质量筛选：优先保留权威来源、高可信度的内容。
   
数据与场景的匹配性问题：知识库与应用场景偏离会导致检索失效。

解决方案：
1. 场景过滤：仅保留与目标任务相关的数据（例如医疗场景需剔除无关行业内容）。
2. 动态更新：定期增量更新数据，避免时效性内容过期。
3. 冷启动优化：初期可引入人工标注的高质量种子数据。

安全与合规风险问题：随意导入数据可能泄露敏感信息或引入偏见。

解决方案：
1. 敏感信息过滤：使用NER识别并脱敏（如身份证号、电话号码）。
2. 偏见检测：通过公平性评估工具（如Fairness Indicators）筛查歧视性内容。
3. 权限控制：对知识库分级访问，限制敏感数据检索权限。

还有以下注意事项：

文本分块（Chunking）需策略化问题：随意分块可能导致语义不完整，影响向量表示。

解决方案：
1. 按语义切分：使用句子边界检测、段落分割或基于语义相似度的算法（如BERT句间相似度）。
2. 动态调整块大小：根据数据特性调整（例如技术文档适合较长的块，对话数据适合短块）。
3. 重叠分块：相邻块保留部分重叠文本，避免关键信息被切分到边缘。

向量化模型的适配性问题：直接使用通用模型可能无法捕捉领域语义。

解决方案：
1. 领域微调：在领域数据上微调模型（如BERT、RoBERTa）以提升向量表征能力。
2. 多模态支持：若包含图表、代码等，需选择支持多模态的模型（如CLIP、CodeBERT）。
3. 轻量化部署：权衡精度与效率，可选择蒸馏后的模型（如MiniLM）。

索引结构与检索效率问题：海量数据未经优化会导致检索延迟。

解决方案：
1. 分层索引：对高频数据使用HNSW，长尾数据用IVF-PQ（Faiss或Milvus）。
2. 元数据过滤：为数据添加标签（如时间、类别），加速粗筛过程。
3. 分布式部署：按数据热度分片，结合缓存机制（如Redis）提升响应速度。

补充说明：向量知识库数据集也要是问答对？

将数据整理成问答对（QA Pair）形式是一种优化策略，而非必要步骤。但这种方式在特定场景下能显著提升检索和生成的效果。

以下是其核心原因和适用场景的分析：

1. 为什么问答对形式能优化RAG？
  
   （1）精准对齐用户查询意图问题：用户输入通常是自然语言问题（如“如何重置密码？”），而知识库若存储的是纯文本段落（如技术文档），检索时可能因语义差异导致匹配失败。
   问答对的优势：直接以“问题-答案”形式存储知识，检索时相似度计算更聚焦于“问题与问题”的匹配（Question-Question Similarity），而非“问题与段落”的匹配。
   例如，若知识库中存有QA对 Q: 如何重置密码？ → A: 进入设置页面，点击“忘记密码”...，当用户提问“密码忘了怎么办？”时，即使表述不同，向量模型也能捕捉到语义相似性。
   
   （2）降低生成模型的负担问题：若检索到的是长文本段落，生成模型（如GPT）需要从段落中提取关键信息并重组答案，可能导致信息冗余或遗漏。
   问答对的优势：答案部分已是对问题的直接回应，生成模型只需“改写”或“补充”答案，而非从头生成，降低幻觉风险。
   例如，QA对中的答案已结构化（如步骤列表），生成结果更规范。
   
   （3）提升检索效率与召回率问题：传统分块检索可能因文本块过长或过短导致关键信息丢失（如答案分散在多个段落）。
   问答对的优势：每个QA对是自包含的语义单元，检索时直接返回完整答案，减少上下文碎片化问题。可针对高频问题设计专用QA对，提高热门问题的响应速度和准确性。

2. 哪些场景适合问答对形式？
   
   （1）任务型对话系统适用场景：客服机器人、技术支持、医疗咨询等垂直领域。原因：用户需求明确，答案通常简短且结构化（如操作步骤、诊断建议）。案例：用户问：“如何退订会员？” → 直接匹配QA对中的答案：“登录账号→进入订阅管理→点击取消”。
   
   （2）FAQ（常见问题解答）库适用场景：产品帮助文档、政策解读等。原因：FAQ天然适合QA形式，直接覆盖高频问题。案例：知识库存储 Q: 保修期多久？ → A: 本产品保修期为2年。
   
   （3）知识密集型生成任务适用场景：需要精确引用事实的场景（如法律咨询、学术问答）。原因：QA对中的答案可作为“事实锚点”，减少生成模型的自由发挥。案例：用户问：“《民法典》规定离婚冷静期多久？” → 返回QA对中的法条原文。
   
问答对构建的注意事项并非所有数据都适合QA形式避免强制转换：叙述性文本（如小说、新闻）或开放域知识（如百科条目）更适合以段落或实体为中心存储。强行拆分为QA可能导致信息割裂（例如将“量子力学发展史”拆解为多个不连贯的问答）。

### 如何选择合适的索引
**ANNS向量索引**
近邻检索算法，该算法不局限于返回相似度最高的的topk结果，而是只搜索目标的近邻，在可以接受的范围内用精度来换速度。

根据实现方法，ANNS向量索引可以分为四种类型：基于树、基于图、基于哈希和基于量化。

**浮点嵌入的索引**

对于n维浮点嵌入，其占用的存储空间为n*float的大小，而用于浮点嵌入的距离度量是欧氏距离L2和内积IP。

这些类型的索引包括FLAT,IVF_FLAT,IVF_PQ,IVF_SQ8,HNSW,HNSW_SQ,HNSW_PQ,HNSW_PRQ 和SCANN ，用于基于 CPU 的 ANN 搜索。
1. **FLAT**
   对于需要完美的精确度（100%召回率）并且数据集比较小，FLAT不压缩向量，可以将FLAT的结果作为其他索引的比较点。
   
   FLAT采用的是暴力搜索，不适合海量的数据。

   在搜索的时候，可以指定度量metric_type为L2或IP

2. **IVF_FLAT**
   高速查询、尽可能高的召回率

   IVF_FLAT是FLAT的改进版，它将数据集分成多个子集，每个子集称为一个“聚类”，然后对每个聚类进行暴力搜索。

   构建的时候需要指定nlist（聚类数量）默认128

   在搜索的时候，支持普通搜索nprobe(要查询的单位数)和范围搜索max_empty_result_buckets（最大空白桶数，默认为2）

3. **IVF_SQ8**
   极高速的查询，内存资源有限，可以接受召回率稍许下降

   IVF_PQ是IVF_FLAT的改进版，它使用PQ（Product Quantization）对每个聚类进行量化，从而减少存储空间和搜索时间。

   当你的电脑配置比较拉的时候，可以选择IVF_SQ8，它比IVF_PQ更节省内存，但精度会稍微低一些。这种索引可以通过执行标量量化SQ，将每个FLOAT（4字节）转换为UINT8（1字节）。这样可以减少四分之三的磁盘、CPU和GPU内存消耗。

   参数同上一条

4. **IVF_PQ**
   高速查询、内存资源有限，可以接受召回率稍许下降

   PQ（乘积量化），把原始高维的向量分解为低维的笛卡尔乘积，每个低维向量称为一个“子向量”，然后对每个子向量进行量化。乘积量化不需要计算目标向量与所有单元中心的距离，而是能够计算目标向量与每个低维空间聚类中心的距离。

   IVF_PQ先进行IVF索引聚类，然后再对向量的乘积进行量化，时间换内存。

5. **SCANN**
   极高速查询、要求尽可能高的召回率，要求内存资源大

   可扩展近邻，与上面的PQ相似，但是SCANN是专门为GPU优化的，在GPU上运行速度更快。

6. **HNSW**
   极高速查询、要求尽可能高的召回率，要求内存资源大

   分层导航小世界图，是一种基于图的索引算法，根据一定的规则构建多层导航结构，在这种结构中，上层较为稀疏，节点之间的距离较远；下层较为密集，节点的距离相对较近。为了提高性能，HNSW 将图中每层节点的最大度数限制为M 。此外，您还可以使用efConstruction （建立索引时）或ef （搜索目标时）来指定搜索范围。

7. **HNSW_SQ**
   稍慢于HNSW的速度，内存有限，可以接受召回率稍许下降

   SQ是一种根据浮点数据的大小将其离散化为一组有限数值的技术。
   
   例如：

      SQ6 表示量化为2的6次方=64个离散值。

   这种方法减少了内存占用，又保留了数据的基本结构，结合了SQ，HNSW_SQ可以在索引大小和精确度之间进行可控的权衡，同时保持较高的每秒查询次数（QPS）性能。与标准 HNSW 相比，它只会适度增加索引构建时间。
8. **HNSW_PQ**
   比SQ稍慢的速度，内存！非常的！有限，可以接受召回率明显下降

   PQ将向量分解为多个子向量，每个子向量根据kmeans算法找到最近的那个中心点，并以此中心点作为其近似子向量。与 PQ 相结合，HNSW_PQ 可以在索引大小和准确性之间进行可控的权衡，但在相同的压缩率下，它的 QPS 值和召回率都比 HNSW_SQ 低。与 HNSW_SQ 相比，它建立索引的时间更长。
9.  **HNSW_PRQ**
   跟PQ差不多的速度、内存资源非常有限，召回率稍微下降

   与PQ相似，也是将向量分为多个子向量，每个子向量将被编码为nbits 。完成 pq 量化后，它会计算向量与 pq 量化向量之间的残差，并对残差向量应用 pq 量化。总共将进行nrq 次完整的 pq 量化，因此长度为dim 的浮动向量将被编码为m ⋅ nbits ⋅ nrqbits。

   HNSW_PRQ 与乘积残差量化器（PRQ）相结合，在索引大小和精确度之间提供了更高的可控权衡。与 HNSW_PQ 相比，在相同的压缩率下，HNSW_PRQ 的 QPS 值和召回率几乎相当。，建立索引的时间可能会增加数倍。

**二进制嵌入索引**

1. **BIN_FLAT**
   于FLAT完全相同，但只能用于二进制嵌入。
2. **BIN_IVF_FLAT**
   与IVF_FLAT完全相同，但只能用于二进制嵌入。

**稀疏嵌入式索引**
稀疏嵌入式索引仅支持IP 和BM25 （用于全文检索）度量。

**GPU索引**
前提：极高吞吐量的场景，与使用 CPU 索引相比，使用 GPU 索引并不一定能减少延迟。

1. **GPU_CAGRA**
   GPU_CAGRA 是为 GPU 优化的基于图的索引。内存使用量约为原始向量数据的 1.8 倍。
  
   构建时有如下的参数：
   * intermediate_graph_degree 通过在剪枝前确定图的度数来影响召回率和构建时间，推荐32/64
   * graph_degree 通过设置剪枝后图形的度数来影响搜索性能和召回率。这两个度数之间的差值越大，构建时间就越长。其值必须小于intermediate_graph_degree 的值。
   * build_algo 选择剪枝前的图形生成算法。可能的值：IVF_PQ:提供更高的质量，但构建时间较慢。NN_DESCENT提供更快的生成速度，但召回率可能较低。
   * cache_dataset_on_device 决定是否在 GPU 内存中缓存原始数据集。可能的值“true”:缓存原始数据集，通过完善搜索结果提高召回率。“false”不缓存原始数据集，以节省 GPU 内存。
   * adapt_for_cpu 决定是否使用 GPU 建立索引和使用 CPU 进行搜索。将该参数设置为true 时，搜索请求中必须包含ef 参数。
   
   查询时有如下参数：
   * itopk_size	决定搜索过程中保留的中间结果的大小。较大的值可能会提高召回率，但会降低搜索性能。它至少应等于最终的 top-k（极限）值，通常是 2 的幂次（例如 16、32、64、128）。
   * search_width	指定搜索过程中进入 CAGRA 图的入口点数量。增加该值可以提高召回率，但可能会影响搜索性能（如 1、2、4、8、16、32）。
   * min_iterations /max_iterations	控制搜索迭代过程。默认设置为0 ，CAGRA 会根据itopk_size 和search_width 自动确定迭代次数。手动调整这些值有助于平衡性能和准确性。
   * team_size	指定用于在 GPU 上计算度量距离的 CUDA 线程数。常用值是 2 的幂次，最高可达 32（例如 2、4、8、16、32）。它对搜索性能影响不大。默认值为0 ，Milvus 会根据向量维度自动选择team_size 。
   * ef	指定查询时间/准确性的权衡。ef 值越高，搜索越准确，但速度越慢。如果在建立索引时将adapt_for_cpu 设置为true ，则必须使用此参数。范围：[top_k, int_max]

2. **GPU_IVF_FLAT**
   需要与原始数据大小相等的内存。

   和IVF_FLAT类似，也是将数据划分为nlist（群组单位数）聚类单元，然后比较目标输入向量和每个聚类中心的距离。随着目标输入向量数 (nq) 和要搜索的簇数 (nprobe要查询的单位数) 的增加，查询时间也会急剧增加。

3. **GPU_IVF_PQ**
   占用内存较少，具体取决于压缩参数设置。

   PQ时乘积量化，将原始高维的向量空间均匀分解为低维向量空间的笛卡尔乘积，然后对分解后的低维向量进行量化。乘积量化不需要计算目标向量与所有单元中心的距离，而是能够计算目标向量与每个低维空间聚类中心的距离，大大降低了算法的时间复杂度和空间复杂度。

   IVF_PQ先进行IVF聚类，然后再对向量的乘积进行量化。其索引文件比 IVF_SQ8 更小，但在搜索向量时也会造成精度损失。

4. **GPU_BRUTE_FORCE**
   需要与原始数据大小相等的内存。

   专为对召回率要求极高的情况定制，通过暴力搜索，包装召回率为1。它只需要度量类型 (metric_type) 和 top-k (limit) 作为索引构建和搜索参数。



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