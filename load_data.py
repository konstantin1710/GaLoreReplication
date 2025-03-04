from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train") # TODO adjust split
dataset = dataset.take(1000)

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function)