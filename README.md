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
