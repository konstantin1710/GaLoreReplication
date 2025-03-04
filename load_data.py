from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("allenai/c4", "realnewslike", split="train[:0.1%]") # TODO adjust split

# load LLAMA tokenizer TODO variable model name
tokenizer = AutoTokenizer.from_pretrained("huggingface/llama-60m")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])