from transformers import AutoTokenizer, AutoModel
import torch

class KeywordExtractor:
    def __init__(self, model_name="bert-base-chinese", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_keywords(self, text, top_k=5):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            attn_mean = attn.mean(dim=0)[0, 1:]  # [CLS]对每个token的注意力
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
            top_indices = attn_mean.topk(top_k).indices.tolist()
            keywords = [tokens[i] for i in top_indices if tokens[i].isalnum()]
        return keywords