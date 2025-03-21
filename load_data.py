import torch
from datasets import load_dataset

def load_data(args, tokenizer):
    if args.mode == "pretraining":
        return load_data_pretrain(args, tokenizer)
    elif args.mode == "finetuning":
        return load_data_finetune(args, tokenizer)
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")

def load_data_pretrain(args, tokenizer):
    dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train")
    dataset = dataset.take(args.num_training_tokens)

    def tokenize_function_pretrain(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
        return {
            "input_ids": torch.tensor(encoding["input_ids"]).clone().detach().to(torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"]).clone().detach().to(torch.long),
        }

    dataset = dataset.map(tokenize_function_pretrain, remove_columns=["text", "timestamp", "url"])
    dataset.with_format("torch")

    return dataset

def load_data_finetune(args, tokenizer):
    dataset = load_dataset("glue", "sst2")

    def tokenize_function_finetune(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=args.max_length)
    
    dataset = dataset.map(tokenize_function_finetune)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return dataset
