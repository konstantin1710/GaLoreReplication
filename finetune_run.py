from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
from logger import logger_init, log_memory_usage, log_max_memory
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from args import args
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

accelerator = Accelerator()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {device}")
print(f"Using optimizer: {args.optimizer}")

logger_init()

# Lade das RoBERTa-Modell für Fine-Tuning
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Lade GLUE/SST2-Dataset oder anderes Benchmark-Set
dataset = load_dataset("glue", "sst2")
def tokenize_function(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = DataLoader(dataset["train"], batch_size=8, shuffle=True)

# Wähle den Optimizer
if args.optimizer == "galore":
    optimizer = GaLoreAdamW8bit(model.parameters(), lr=2e-4, weight_decay=0)
elif args.optimizer == "lora":
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

# Prepare for training
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

log_memory_usage("Before Fine-Tuning")

num_epochs = 3
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} (Fine-Tuning), Loss: {avg_loss:.4f}")

    log_memory_usage(f"After Epoch {epoch+1}")

# Speichere das fein-getunte Modell
model.save_pretrained("roberta_finetuned")
tokenizer.save_pretrained("roberta_finetuned")

log_max_memory()
