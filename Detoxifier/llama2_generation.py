from pathlib import Path
from typing import Union, List
import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
)
from transformers import AutoModelForCausalLM, LlamaPreTrainedModel

MAX_LENGTH = int(10000)

def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

class LLama2Generation:
    def __init__(
        self,
        model: Union[str, Path, LlamaPreTrainedModel] = "llama2",
        tokenizer: str = "",
        seed: int = 42,
    ):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)
        # Set up model
        model_name_or_path = str(model)
        print(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model = model
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<LLama2Generator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_len: int = 20,
        sample: bool = True,
        k: int = 0,
        p: float = 1.0,
        temperature: float = 1.0,
        **model_kwargs,
    ) -> List[str]:
        print(max_len)
        print(k)
        print(p)
        print(temperature)
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(
            prompt, padding=True, return_tensors="pt"
        )

        input_ids = encodings_dict["input_ids"].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_len,
                do_sample=sample,
                temperature=temperature,
                top_p=p,
                top_k=k,
                output_hidden_states=True,
            )
        decoded_outputs = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in generated_ids[:, input_seq_len:]
        ]
        return decoded_outputs
