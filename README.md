# MAAD: Model-Agnostic Adaptive  Detoxification Framework

This repository contains the source code, model checkpoints, and experimental settings for our AAAI 2026 submission: **"From Chaos to Cure: A Prefix Heuristics Guided Model-Agnostic Adaptive  Detoxification Framework"**.

# Overview
- 111
- xxx
- xxx
- xxx

# Folder Structure

# Setup
```
conda create -n destein python=3.10
conda activate destein
pip install -r requirements.txt
```

```
python trainSegmentLocate.py \
  --data_path ../data/jigsaw_train.csv \
  --output_dir ./roberta_model
```


```
python locate_edit_script.py \
  --input_file ../data/train.jsonl \
  --output_file results/edit_output.jsonl \
  --model_path ./roberta_model \
  --max_window_size 3 \
  --top_k 1 \
  --tox_threshold 0.2 \
  --sim_threshold 0.8
```

```
python constructAntidote.py \
  --input_path ./results/edit_output.jsonl \
  --output_path ../data/antidote.json \
  --model_type gpt2 \
  --model openai-community/gpt2-xl \
  --tokenizer openai-community/gpt2-xl \
  --seed 2025 \
  --max_len 20 \
  --warmup_ratio 0.2
```

```
python trainDetoxifier.py \
  --model_path LLM-Research/Llama-3.2-1B \
  --finetune_path ../data/antidote.json \
  --save_path ./models/lora_finetuned \
  --max_length 512 \
  --batch_size 8 \
  --accum_steps 2 \
  --epochs 10 \
  --save_steps 200 \
  --learning_rate 5e-5
```


```
python generate_main.py \
  --data_path ../data/ToxicRandom.jsonl \
  --model_type llama2_maad \
  --detoxifier_path LLM-Research/Llama-3.2-1B \
  --lora_path ./models/lora_finetuned \
  --output_dir ./outputs \
  --n 25 \
  --batch_size 25 \
  --max_len 20 \
  --tau 0.1\
  --temperature 1.0
  --overwrite
```

```
python evaluate_generations.py \
  --input_file ./outputs/llama2_maad.json \
  --ppl_model_path modelscope/Llama-2-7b-ms \
  --sim_model_path BAAI/bge-large-en-v1___5 \
  --toxicity_api_key your_google_perspective_api_key

```

