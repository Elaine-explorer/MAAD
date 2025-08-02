import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df['comment_text'].tolist()
        self.labels = df['target'].clip(0, 1).astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    labels = labels.squeeze()
    return {
        "mse": np.mean((preds - labels) ** 2),
        "mae": np.mean(np.abs(preds - labels)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/train.csv', help='Path to CSV file')
    parser.add_argument('--output_dir', type=str, default='./roberta_model', help='Where to save model')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)[["comment_text", "target"]].dropna()
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

    train_dataset = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_dataset = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
