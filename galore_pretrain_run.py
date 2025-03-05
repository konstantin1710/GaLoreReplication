from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
from load_data import tokenizer, tokenized_dataset
from logger import logger_init, log_memory_usage, log_max_memory
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader

accelerator = Accelerator()

# TODO commandozeile übergibt "lora" oder "galore" als argument
# Parameter variabel/loop wie in Bericht beschrieben
# llama-60m verändern / json überarbeiten
# finetuning von roberta mit beiden optimizern
# benchmarking mit glue / anderen benchmarks
# warnings in lora(?) fixen

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

logger_init()

model_config = AutoConfig.from_pretrained("llama_60m.json")
model = AutoModelForCausalLM.from_config(model_config)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if model.config.pad_token_id is None or model.config.pad_token_id == -1:
    model.config.pad_token_id = tokenizer.pad_token_id

model.generation_config.pad_token_id = model.config.pad_token_id

# model = LlamaForCausalLM(model_config)

# param_groups = [{'params': non_galore_params}, 
#                 {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}]
# optimizer = GaLoreAdamW(param_groups, lr=0.01)

optimizer = GaLoreAdamW8bit(model.parameters(), lr=2e-4, weight_decay=0) #TODO adjust learning rate

train_dataloader = DataLoader(tokenized_dataset, batch_size=8) #TODO adjust batch size

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

log_memory_usage("Before Training")

num_epochs = 2 #TODO adjust number of epochs
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    batch_cnt = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backpropagation with `accelerate`
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        batch_cnt += 1

        if batch_cnt % 20 == 0: log_memory_usage(f"After Batch {batch_cnt}")
    
    avg_loss = total_loss / max(1, batch_cnt)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    log_memory_usage(f"After Epoch {epoch+1}")

model.save_pretrained("llama60m_galore")
tokenizer.save_pretrained("llama60m_galore")

log_max_memory()