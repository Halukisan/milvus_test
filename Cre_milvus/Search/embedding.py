from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingGenerator:
    def __init__(self, model_name="shibing624/text2vec-base-chinese", device=None):
        # 可选择
        # shibing624/text2vec-base-chinese
        # BAAI/bge-base-zh
        # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text):
        # 使用tokenizer对输入的text进行编码，返回的inputs是一个字典，包含input_ids和attention_mask
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # 将inputs中的tensor数据移动到指定的设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 在不计算梯度的情况下，获取模型的输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取最后一层的隐藏状态，并取第一个token的隐藏状态作为embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        # 将numpy数组转换为列表
        return embedding.tolist()