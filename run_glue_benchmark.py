import torch
import argparse
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO just a draft - not complete, not tested

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
parser.add_argument("--task", type=str, choices=["sst2", "mnli", "qqp"], default="sst2", help="GLUE task for evaluation")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model_type", type=str, choices=["roberta", "gpt2"], required=True)
args = parser.parse_args()

print(f"Loading model from {args.model_path}...")
if args.model_type == "roberta":
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
elif args.model_type == "gpt2":
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

dataset = load_dataset("glue", args.task)
metric = load_metric("glue", args.task)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset["validation"].map(preprocess_function, batched=True)
dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size)

all_preds = []
all_labels = []

print("Starting evaluation...")
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = batch["label"]

        if args.model_type == "roberta":
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        elif args.model_type == "gpt2":
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits[:, -1, :], dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

result = metric.compute(predictions=all_preds, references=all_labels)
print(f"Benchmark Results for {args.task}: {result}")
