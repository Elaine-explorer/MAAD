import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from gpt2_generation import GPT2Generation
from llama2_generation import LLama2Generation
from opt_generation import OPTGeneration
from mistral_generation import MistralGeneration
from Detoxifier import Detoxifier

# Supported models
ALLOWED_MODELS = [
    "gpt2_base", "gpt2_maad",
    "llama2_base", "llama2_maad",
    "opt_base", "opt_maad",
    "mistral_base", "mistral_maad"
]

BASEMODEL_PATHS = {
    "gpt2": "AI-ModelScope/gpt2-large",
    "llama2": "modelscope/Llama-2-7b-ms",
    "opt": "facebook/opt-6___7b",
    "mistral": "AI-ModelScope/Mistral-7B-v0___1"
}


def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../data/ToxicRandom.jsonl")
    parser.add_argument('--detoxifier_path', required=True, help="Base model path for detoxifier")
    parser.add_argument('--lora_path', required=True, help="LoRA weights path for detoxifier")
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--model_type', type=str, choices=ALLOWED_MODELS, default="llama2_maad")
    parser.add_argument('--output_dir', default="./model_output")
    parser.add_argument('--n', type=int, default=25, help="Number of generations per prompt")
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_generator(model_type: str, model_path: str, seed: int):
    model_class_map = {
        "gpt2": GPT2Generation,
        "llama2": LLama2Generation,
        "opt": OPTGeneration,
        "mistral": MistralGeneration
    }
    model_prefix = model_type.split("_")[0]
    generator_class = model_class_map[model_prefix]
    return generator_class(model=model_path, tokenizer=model_path, seed=seed)


def generate_batch(generator, prompts, batch_size, max_len, detoxifier=None):
    all_generations = []
    for batch in tqdm(batchify(prompts, batch_size)):
        outputs = generator.generate(batch, max_len)
        if detoxifier:
            outputs = detoxifier.detoxify(batch, outputs)
        all_generations.extend(outputs)
    return all_generations


def generate_main(args):
    model_prefix = args.model_type.split("_")[0]
    model_path = BASEMODEL_PATHS[model_prefix]

    generator = load_generator(args.model_type, model_path, args.seed)
    detoxifier = Detoxifier(args.detoxifier_path, args.lora_path, args.tau, args.temperature, args.max_len) if "maad" in args.model_type else None

    # Output path
    output_type = "top_toxic" if "top" in args.data_path else "normal_toxic"
    output_dir = Path(args.output_dir) / output_type
    ensure_dir(output_dir)

    output_file = output_dir / f"{args.model_type}V_public.json"
    if output_file.exists() and not args.overwrite:
        print(f"[!] Output exists: {output_file}. Use --overwrite to replace it.")
        sys.exit(0)

    # Load prompts
    dataset = pd.read_json(args.data_path, lines=True)
    prompts = pd.json_normalize(dataset["prompt"])["text"].tolist()
    repeat_prompts = np.repeat(prompts, args.n).tolist()

    # Generate
    generations = generate_batch(generator, repeat_prompts, args.batch_size, args.max_len, detoxifier=detoxifier)

    # Group generations
    results = []
    for i, prompt in enumerate(prompts):
        start = i * args.n
        end = (i + 1) * args.n
        results.append({
            "data_id": f"data_{i+1}",
            "prompt": prompt,
            "generations": generations[start:end]
        })

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved to {output_file}")


def main():
    args = parse_args()
    generate_main(args)


if __name__ == "__main__":
    main()
