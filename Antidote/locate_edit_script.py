import argparse
import pandas as pd
import torch
import json
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import openai
import httpx
import re
from RelevanceScorer import RelevanceScorer
from ToxicityEvaluator import ToxicityEvaluator

openai.api_key = "sk-xxxxxxxxxxxx"
openai.base_url = "https://oneapi.xty.app/v1"
openai.http_client = httpx.Client(base_url="https://oneapi.xty.app/v1", follow_redirects=True)

def edit_masked_text(masked: str, original: str, model="gpt-3.5-turbo-0613") -> str:
    prompt_mask = re.sub(r"(\[MASK\]\s*)+", "<MASK>", masked)
    instruction = f'Replace "<MASK>" in: "{prompt_mask}" with content that makes it semantically related to "{original}" but non-toxic.'
    try:
        output = openai.ChatCompletion.create(
            model=model,
            messages=[
                {'role': 'system', 'content': "You are a helpful assistant."},
                {'role': 'user', 'content': instruction}
            ],
            temperature=1,
            max_tokens=100,
        )
        return output.choices[0].message.content.strip()
    except Exception as e:
        print(f"Edit Error: {e}")
        return masked.replace("[MASK]", "[REDACTED]")

def gradient_attribution(input_ids, attention_mask, model):
    input_ids = input_ids.clone().detach().requires_grad_(True)
    outputs = model(input_ids, attention_mask=attention_mask)
    score = torch.sigmoid(outputs.logits).squeeze()
    loss = -torch.log(score + 1e-8)
    loss.backward()
    grads = input_ids.grad.abs().sum(dim=-1).squeeze().cpu().numpy()
    return grads

def get_top_k_non_overlapping_spans(grads, tokens, max_window_size, top_k, valid_range):
    candidates = []
    for i in valid_range:
        for w in range(1, max_window_size + 1):
            j = i + w
            if j > len(tokens):
                continue
            if any(idx not in valid_range for idx in range(i, j)):
                continue
            score = grads[i:j].sum()
            candidates.append((i, j, score))

    candidates.sort(key=lambda x: -x[2])
    selected_spans, occupied = [], set()
    for i, j, score in candidates:
        if all(k not in occupied for k in range(i, j)):
            selected_spans.append((i, j))
            occupied.update(range(i, j))
        if len(selected_spans) >= top_k:
            break
    return selected_spans

def mask_gradient_spans(text, model, tokenizer, scorer, evaluator, max_window_size=3, top_k=1, tox_threshold=0.2, sim_threshold=0.8):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    grads = gradient_attribution(input_ids, attention_mask, model)
    valid_range = (input_ids[0] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0].tolist()
    spans = get_top_k_non_overlapping_spans(grads, tokens, max_window_size, top_k, valid_range)

    masked_tokens = tokens[:]
    span_texts = []
    for i, j in spans:
        for idx in range(i, j):
            masked_tokens[idx] = '[MASK]'
        span_texts.append(" ".join(tokens[i:j]))

    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)

    edited = constrained_edit(
        masked_text, text,
        scorer, evaluator,
        tox_threshold=tox_threshold,
        sim_threshold=sim_threshold
    )

    if edited:
        return {
            "original": text,
            "masked": masked_text,
            "masked_spans": span_texts,
            "edited": edited
        }

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print("跳过无效行:", line.strip())
    return pd.DataFrame(rows)

def load_datas(file_path):
    data = load_jsonl(file_path)
    dataset = pd.DataFrame(data)
    texts = pd.json_normalize(dataset['prompt'])['text']
    return texts


def save_jsonl(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def constrained_edit(masked: str,
                     original: str,
                     scorer: RelevanceScorer,
                     evaluator: ToxicityEvaluator,
                     tox_threshold: float = 0.2,
                     sim_threshold: float = 0.8,
                     max_retry: int = 3) -> str:
    candidate = None
    for attempt in range(max_retry):
        candidate = edit_masked_text(masked, original)

        tox = evaluator.evaluate(candidate)
        sim = scorer.score(original, candidate)

        print(f"[{attempt+1}] Tox: {tox:.3f} | Sim: {sim:.3f}")
        if tox <= tox_threshold and sim >= sim_threshold:
            return candidate

    return candidate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to .txt or .jsonl file with one text per line")
    parser.add_argument('--output_file', type=str, required=True, help="Output .jsonl path")
    parser.add_argument('--model_path', type=str, default='model', help="Trained RoBERTa model path")
    parser.add_argument('--max_window_size', type=int, default=3, help="Max n-gram size")
    parser.add_argument('--top_k', type=int, default=1, help="How many spans to mask")
    parser.add_argument('--tox_threshold', type=float, default=0.2)
    parser.add_argument('--sim_threshold', type=float, default=0.8)
    args = parser.parse_args()

    scorer = RelevanceScorer()
    evaluator = ToxicityEvaluator(api_key="your-perspective-api-key")

    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    if args.input_file:
        texts = load_datas(args.input_file)
        results = []
        for text in tqdm(texts, desc="mask toxic segment"):
            result = mask_gradient_spans(text, model, tokenizer, scorer, evaluator, args.max_window_size, args.top_k, args.tox_threshold, args.sim_threshold)
            if result:
                results.append(result)
        save_jsonl(results, args.output_file)

        print(f"\nsave to: {args.output_file}, {len(results)}")

    else:
        print("no input_file")
        exit(0)

if __name__ == "__main__":
    main()
