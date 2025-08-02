import json
import os
import random
import sys
import argparse
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from Detoxifier.gpt2_generation import GPT2Generation
from Detoxifier.llama2_generation import LLama2Generation
from Detoxifier.opt_generation import OPTGeneration
from Detoxifier.mistral_generation import MistralGeneration

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipped invalid JSON line.")
    return data


def get_generator(model_type: str, model: str, tokenizer: str, seed: int):
    if model_type == "gpt2":
        return GPT2Generation(model=model, tokenizer=tokenizer, seed=seed)
    elif model_type == "llama2":
        return LLama2Generation(model=model, tokenizer=tokenizer, seed=seed)
    elif model_type == "opt":
        return OPTGeneration(model=model, tokenizer=tokenizer, seed=seed)
    elif model_type == "mistral":
        return MistralGeneration(model=model, tokenizer=tokenizer, seed=seed)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def build_corrector_dataset(input_path: str,
                             output_path: str,
                             generator,
                             max_len: int = 20,
                             warmup_ratio: float = 0.2):
    data = load_jsonl(input_path)
    corrector_contents = []

    for item in tqdm(data, desc="Building corrector dataset"):
        ori_prompt = item["original"]
        edited_prompt = item["edited"]

        ori_response = generator.generate([ori_prompt], max_len)[0]
        edited_response = generator.generate([edited_prompt], max_len)[0]

        if random.uniform(0, 1) > warmup_ratio:
            output = f"{ori_prompt} | {ori_response}"
        else:
            output = f"{edited_prompt} | {edited_response}"

        entry = {
            "instruction": "Edit the following suffix content led by prefix to make it safe, respectful, and non-toxic while retaining the original semantics:",
            "input": f"prefix:{ori_prompt} | suffix:{ori_response}",
            "output": output
        }
        corrector_contents.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corrector_contents, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(corrector_contents)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build corrector training data using detoxified prompt pairs and model responses.")

    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input edit_output.jsonl file.")
    parser.add_argument("--output_path", type=str, default="./data/antidote.json",
                        help="Path to save the output corrector dataset.")
    parser.add_argument("--model_type", type=str, choices=["gpt2", "llama2", "opt", "mistral"], default="gpt2",
                        help="Type of language model to use for generation.")
    parser.add_argument("--model", type=str, default="openai-community/gpt2-xl",
                        help="HuggingFace model path or identifier.")
    parser.add_argument("--tokenizer", type=str, default="openai-community/gpt2-xl",
                        help="HuggingFace tokenizer path or identifier.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--max_len", type=int, default=20,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.2,
                        help="Ratio of examples using the edited version as output.")

    args = parser.parse_args()

    generator = get_generator(args.model_type, args.model, args.tokenizer, args.seed)

    build_corrector_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        generator=generator,
        max_len=args.max_len,
        warmup_ratio=args.warmup_ratio
    )


if __name__ == "__main__":
    main()
