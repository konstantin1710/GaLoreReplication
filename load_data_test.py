import torch
from datasets import load_dataset

def load_data(mode, tokenizer):
    if mode == "pretraining":
        dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train")
        dataset = dataset.take(100)
    elif mode == "finetuning":
        dataset = load_dataset("glue", "sst2", streaming=True)
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")

    def tokenize_function(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        return {
            "input_ids": torch.tensor(encoding["input_ids"]).clone().detach().to(torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"]).clone().detach().to(torch.long),
        }

    dataset = dataset.map(tokenize_function, remove_columns=["text", "timestamp", "url"])
    dataset = dataset.with_format("torch")

    return dataset