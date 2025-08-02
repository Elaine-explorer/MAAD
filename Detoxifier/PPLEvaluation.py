import os

import pandas as pd
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import os

T = TypeVar("T")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

class PerplexityEvaluator:
    def __init__(self, model_id, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def compute(self, prefix, continuation):
        # 编码前缀和完整输入
        prefix_ids = self.tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        full_ids = self.tokenizer.encode(prefix + continuation, return_tensors="pt").to(self.device)

        with torch.no_grad():
            prefix_loss = self.model(prefix_ids, labels=prefix_ids).loss * (prefix_ids.shape[1] - 1)
            full_loss = self.model(full_ids, labels=full_ids).loss * (full_ids.shape[1] - 1)

        gen_len = full_ids.shape[1] - prefix_ids.shape[1]
        loss = (full_loss - prefix_loss) / gen_len
        ppl = math.exp(loss.item())
        return ppl

    def compute_ppl(self, prefix, continuation):
        prefix_ids = self.tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        full_ids = self.tokenizer.encode(prefix + continuation, return_tensors="pt").to(self.device)

        prefix_loss = self.model(prefix_ids, labels=prefix_ids)[0] * (prefix_ids.shape[1] - 1)

        full_loss = self.model(full_ids, labels=full_ids)[0] * (full_ids.shape[1] - 1)

        loss = (full_loss - prefix_loss) / (
                full_ids.shape[1] - prefix_ids.shape[1]
        )
        ppl = math.exp(loss.item())
        if ppl < 1e4:
            return ppl
        return None


    def compute_full_text(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            loss = self.model(input_ids, labels=input_ids).loss
            ppl = math.exp(loss.item())
        if ppl < 1e4:
            return ppl
        return None


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":

    model_id = "meta-llama/Llama-2-7b"

    ppl_eval = PerplexityEvaluator(model_id)