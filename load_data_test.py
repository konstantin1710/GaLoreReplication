import torch
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train")
dataset = dataset.take(100)

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def tokenize_function(batch):
    encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
    }

tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text", "timestamp", "url"])
tokenized_dataset = tokenized_dataset.with_format("torch")