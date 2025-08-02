from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class RelevanceScorer:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):

        self.model = SentenceTransformer(model_name)

    def score(self, prompt: str, completion: str) -> float:

        embeddings = self.model.encode([prompt, completion], convert_to_tensor=True)
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def batch_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        prompts, completions = zip(*pairs)
        emb1 = self.model.encode(prompts, convert_to_tensor=True)
        emb2 = self.model.encode(completions, convert_to_tensor=True)
        return [score.item() for score in util.cos_sim(emb1, emb2).diagonal()]
