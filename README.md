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
