import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Union

import requests
from tqdm import tqdm

from PPLEvaluation import PerplexityEvaluator
from DiversityEvaluation import DiversityEvaluator
from RelevanceEvaluation import RelevanceScorer

class ToxicityEvaluator:
    def __init__(self, api_key: str, request_per_second: int = 45):
        self.api_key = api_key
        self.api_url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.api_key}"
        self.sleep_time = 1 / request_per_second

    def _call_api(self, text: str) -> float:
        headers = {"Content-Type": "application/json"}
        time.sleep(self.sleep_time)
        for _ in range(3):
            try:
                data = {
                    "comment": {"text": text},
                    "languages": ["en"],
                    "requestedAttributes": {"TOXICITY": {}},
                    "doNotStore": True
                }
                response = requests.post(self.api_url, json=data, headers=headers)
                response.raise_for_status()
                return response.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                time.sleep(1.5)
        return 0.0

    def evaluate(self, texts: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(texts, str):
            return self._call_api(texts)
        elif isinstance(texts, list):
            return [self._call_api(t) for t in texts]
        else:
            raise ValueError("Input should be a string or a list of strings.")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate_outputs(args):
    print(f"Evaluating: {args.input_file}")
    data = load_json(args.input_file)

    ppl_eval = PerplexityEvaluator(args.ppl_model_path)
    diversity_eval = DiversityEvaluator()
    relevance_scorer = RelevanceScorer(model_name=args.sim_model_path)
    toxicity_evaluator = ToxicityEvaluator(api_key=args.toxicity_api_key)

    for item in tqdm(data):
        prompt = item["prompt"]
        generations = item.get("generations", [])

        if "ppl" not in item:
            ppls = []
            for g in generations:
                prefix = prompt if "maad" not in args.input_file else g[:len(prompt)]
                suffix = g if "maad" not in args.input_file else g[len(prompt):]
                ppl = ppl_eval.compute(prefix, suffix)
                if ppl < 1e4:
                    ppls.append(ppl)
            item["ppl"] = ppls

        if "diversity" not in item:
            gen_suffix = generations if "maad" not in args.input_file else [g[len(prompt):] for g in generations]
            dist1, dist2, dist3 = diversity_eval.compute([gen_suffix])
            item["diversity"] = [dist1, dist2, dist3]

        if "relevance" not in item:
            relevances = []
            for g in generations:
                prefix = prompt if "maad" not in args.input_file else g[:len(prompt)]
                suffix = g if "maad" not in args.input_file else g[len(prompt):]
                relevances.append(relevance_scorer.score(prefix, suffix))
            item["relevance"] = relevances

        if "toxicity_scores" not in item:
            item["toxicity_scores"] = toxicity_evaluator.evaluate(generations)

    save_json(data, args.output_file)
    print(f"Evaluation saved to: {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to model output JSON file")
    parser.add_argument("--output_file", help="Path to save evaluated JSON file")
    parser.add_argument("--ppl_model_path", required=True, help="Model path for PPL evaluation")
    parser.add_argument("--sim_model_path", required=True, help="SentenceTransformer model path for similarity")
    parser.add_argument("--toxicity_api_key", required=True, help="Perspective API key")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace("model_output", "evaluation_output")

    if os.path.exists(args.output_file):
        print(f"Output already exists: {args.output_file}")
        sys.exit(0)

    evaluate_outputs(args)
