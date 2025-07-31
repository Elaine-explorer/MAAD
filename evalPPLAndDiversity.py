import argparse
import json
import os
from collections import defaultdict, Counter
import statistics
import numpy as np
from datasets import tqdm

from PPLEvaluation import PerplexityEvaluator

from DiversityEvaluation import DiversityEvaluator

def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


datas = load_data("evaluation_output/top_toxic/gpt2_base.json")

model_id = "/home/Userlist/shangyuhu/premodel/Llama-2-7b-ms/modelscope/Llama-2-7b-ms"

ppl_eval = PerplexityEvaluator(model_id)

diversity_eval = DiversityEvaluator()
contents = []
for data in tqdm(datas):
    generations = data["generations"]
    if "ppl" not in data:
        ppls = []
        for generation in generations:
            ppl = ppl_eval.compute_full_text(generation)
            print("ppl", ppl)
            ppls.append(ppl)
        data["ppl"] = ppls

    if "diversity" not in data:
        print(diversity_eval.compute([generations]))
        dist1, dist2, dist3 = diversity_eval.compute([generations])
        print("diversity", [dist1, dist2, dist3])
        data["diversity"] = [dist1, dist2, dist3]
    contents.append(data)
with open("evaluation_output/top_toxic/gpt2_base.json", 'w', encoding='utf-8') as outfile:
    json.dump(contents, outfile, indent=4, ensure_ascii=False)