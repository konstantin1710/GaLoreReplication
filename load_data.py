import torch
from datasets import load_dataset

def load_data(mode, tokenizer):
    if mode == "pretraining":
        return load_data_pretrain(tokenizer)
    elif mode == "finetuning":
        return load_data_finetune(tokenizer)
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")

def load_data_pretrain(tokenizer):
    dataset = load_dataset("allenai/c4", "realnewslike", split="train[:0.1%]") # TODO adjust split

    def tokenize_function_pretrain(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512) #TODO adjust max_length
        return {
            "input_ids": torch.tensor(encoding["input_ids"]).clone().detach().to(torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"]).clone().detach().to(torch.long),
        }

    dataset = dataset.map(tokenize_function_pretrain, remove_columns=["text", "timestamp", "url"])
    dataset.set_format("torch")

    return dataset

def load_data_finetune(tokenizer):
    dataset = load_dataset("glue", "sst2")

    def tokenize_function_finetune(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=512) #TODO adjust max_length
    
    dataset = dataset.map(tokenize_function_finetune)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return dataset
