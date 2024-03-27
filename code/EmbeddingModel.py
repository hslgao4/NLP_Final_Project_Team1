from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

class SciBERTEmbeddingModel:
    def __init__(self, model_name: str = 'allenai/scibert_scivocab_uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings
