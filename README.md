# From Chaos to Cure: A Prefix Heuristics Guided Model-Agnostic Adaptive Detoxification Framework

# Folder Structure
\begin{verbatim}
MAAD-Detoxifier/
├── Antidote/
│   ├── constructAntidote.py
│   ├── locate_edit_script.py
│   ├── RelevanceScorer.py
│   ├── ToxicityEvaluator.py
│   └── trainSegmentLocate.py
│
├── data/
│   ├── ToxicRandom.jsonl
│   ├── ToxicTop.jsonl
│   └── train.jsonl
│
├── Detoxifier/
│   ├── Detoxifier.py
│   ├── DiversityEvaluation.py
│   ├── evaluate.py
│   ├── generate_main.py
│   ├── gpt2_generation.py
│   ├── llama2_generation.py
│   ├── mistral_generation.py
│   ├── opt_generation.py
│   ├── PPLevaluation.py
│   ├── RelevanceEvaluation.py
│   └── trainDetoxifier.py
│
└── requirements.txt
\end{verbatim}

# Reproduce the Environment
To reproduce our results, first set up a clean Python environment using conda. We recommend Python 3.10 for compatibility with all dependencies. After activating the environment, install the required packages from requirements.txt:
```
conda create -n maad python=3.10
conda activate maad
pip install -r requirements.txt
```
This ensures consistency with our experiments and helps avoid version conflicts.



# Train Toxic Segment Locate
We fine-tune a RoBERTa model to detect toxic segments within user-generated text. This model identifies key toxic phrases that will later be edited in the antidote construction process. You can train the model using:
```
python trainSegmentLocate.py \
  --data_path ../data/jigsaw_train.csv \
  --output_dir ./roberta_model
```
The trained model will be saved in ./roberta_model.



# Toxic Segment Edit
Using the trained toxicity locator, we extract and modify high-toxicity segments. This script selects toxic windows based on a threshold and replaces them using similarity and toxicity criteria:

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
The output is an edited JSONL file that forms the basis for antidote synthesis.



# Antidote Dataset Synthesis
We synthesize "antidote" responses using a generative model (e.g., GPT-2). This script takes edited toxic inputs and generates detoxified targets, creating a dataset for training the detoxifier:

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
The resulting file antidote.json contains input–output pairs for fine-tuning.


# Detoxifier Train Process
We train a lightweight detoxifier on the antidote dataset using LoRA-based fine-tuning for parameter efficiency. The model learns to rewrite toxic inputs into less harmful alternatives:
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
Trained checkpoints are stored under ./models/lora_finetuned.


# Detoxifier Inference Process
We evaluate the detoxifier by generating rewrites for a toxic dataset. The script performs batched generation and saves outputs in JSON format:
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
Each toxic input is rewritten 25 times for robustness analysis.



# Evaluate
We quantitatively assess generation quality across perplexity, semantic similarity, and toxicity. This step uses external models and APIs (e.g., Google Perspective API):
```
python evaluate_generations.py \
  --input_file ./outputs/llama2_maad.json \
  --ppl_model_path modelscope/Llama-2-7b-ms \
  --sim_model_path BAAI/bge-large-en-v1___5 \
  --toxicity_api_key your_google_perspective_api_key

```
Metrics provide insight into fluency, preservation of meaning, and toxicity reduction.


