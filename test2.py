from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
from load_data_test import tokenizer, tokenized_dataset
from logger import logger_init, log_memory_usage, log_max_memory
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from args import args
from peft import LoraConfig, get_peft_model

accelerator = Accelerator()
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {device}")
print(f"Using optimizer: {args.optimizer}")
logger_init()


def get_model(mode):
    """ Erstellt das Modell für Pretraining oder Fine-Tuning """
    if mode == "pretraining":
        model_config = AutoConfig.from_pretrained("config/llama_60m.json")
        model = AutoModelForCausalLM.from_config(model_config)
    elif mode == "finetuning":
        model_config = AutoConfig.from_pretrained("config/roberta_base.json")
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")
    
    # Pad Token setzen
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None or model.config.pad_token_id == -1:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    
    return model


def get_optimizer(model, optimizer_type):
    """ Erstellt den passenden Optimizer (GaLore oder LoRA) """
    if optimizer_type == "galore":
        return GaLoreAdamW8bit(model.parameters(), lr=2e-4, weight_decay=0)
    elif optimizer_type == "lora":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return torch.optim.AdamW(model.parameters(), lr=2e-4), model
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def train(model, optimizer, dataloader, num_epochs=3):
    """ Vereinheitlichtes Training für Pretraining und Fine-Tuning """
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    log_memory_usage("Before Training")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_cnt = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            batch_cnt += 1
            if batch_cnt % 20 == 0:
                log_memory_usage(f"After Batch {batch_cnt}")
        
        avg_loss = total_loss / max(1, batch_cnt)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        log_memory_usage(f"After Epoch {epoch+1}")
    
    return model


# --- Pipeline ausführen ---
model = get_model(args.mode)  # "pretraining" oder "finetuning"
optimizer, model = get_optimizer(model, args.optimizer)
dataloader = DataLoader(tokenized_dataset, batch_size=8)  # TODO: Batch Size anpassen

trained_model = train(model, optimizer, dataloader, num_epochs=3)

# Speichern des Modells
model_path = "llama60m_galore" if args.mode == "pretraining" else "roberta_finetuned"
trained_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

log_max_memory()
