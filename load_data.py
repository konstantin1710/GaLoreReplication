import torch
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("allenai/c4", "realnewslike", split="train[:0.1%]") # TODO adjust split

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def tokenize_function(batch):
    encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids": torch.tensor(encoding["input_ids"]).clone().detach().to(torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"]).clone().detach().to(torch.long),
    }

dataset = dataset.map(tokenize_function, remove_columns=["text", "timestamp", "url"])
dataset.set_format("torch")