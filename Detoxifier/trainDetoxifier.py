import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen model with LoRA on detoxification data.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--finetune_path", type=str, required=True, help="Path to the fine-tuning dataset (.json)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save LoRA fine-tuned model")
    parser.add_argument("--max_length", type=int, default=500, help="Max token length per sample")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps interval to save model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def preprocess(example, tokenizer, max_length):
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def main():
    args = parse_args()
    df = pd.read_json(args.finetune_path)
    dataset = Dataset.from_pandas(df).shuffle(seed=2024)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda x: preprocess(x, tokenizer, args.max_length),
                          remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
