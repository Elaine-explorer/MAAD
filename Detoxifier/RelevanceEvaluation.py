from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class RelevanceScorer:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1___5"):
        self.model = SentenceTransformer(model_name)

    def score(self, prompt: str, completion: str) -> float:

        embeddings = self.model.encode([prompt, completion], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1])
        return score.item()

    def batch_score(self, pairs: List[Tuple[str, str]]) -> List[float]:

        prompts, completions = zip(*pairs)
        embeddings_prompt = self.model.encode(prompts, convert_to_tensor=True)
        embeddings_completion = self.model.encode(completions, convert_to_tensor=True)
        scores = util.cos_sim(embeddings_prompt, embeddings_completion)
        return [score.item() for score in scores.diag()]
if __name__ == "__main__":
    scorer = RelevanceScorer()
    prompt = "What causes earthquakes?"
    completion = "Earthquakes are caused by the sudden release of energy in the Earth's crust."
    print(f"Single relevance score: {scorer.score(prompt, completion):.4f}")
    pairs = [
        ("What is AI?", "AI stands for artificial intelligence."),
        ("Describe photosynthesis.", "Photosynthesis is the process by which plants make food."),
        ("What is the capital of France?", "The capital of France is Paris.")
    ]
    scores = scorer.batch_score(pairs)
    for i, score in enumerate(scores):
        print(f"Pair {i+1} relevance score: {score:.4f}")