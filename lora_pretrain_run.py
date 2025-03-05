from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from load_data_test import tokenizer, tokenized_dataset
from logger import logger_init, log_memory_usage, log_max_memory

accelerator = Accelerator()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

logger_init()

model_config = AutoConfig.from_pretrained("llama_60m.json")
base_model = AutoModelForCausalLM.from_config(model_config)

lora_config = LoraConfig( #TODO adjust
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if lora_model.config.pad_token_id is None or lora_model.config.pad_token_id == -1:
    lora_model.config.pad_token_id = tokenizer.pad_token_id

lora_model.generation_config.pad_token_id = lora_model.config.pad_token_id

lora_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-4) #TODO adjust learning rate

train_dataloader = DataLoader(tokenized_dataset, batch_size=8) #TODO shuffle?

lora_model, lora_optimizer, train_dataloader = accelerator.prepare(lora_model, lora_optimizer, train_dataloader)

log_memory_usage("Before Training")

num_epochs = 3
lora_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    batch_cnt = 0
    for batch in train_dataloader:
        lora_optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        accelerator.backward(loss)
        lora_optimizer.step()

        total_loss += loss.item()
        batch_cnt += 1

        if batch_cnt % 20 == 0: log_memory_usage(f"After Batch {batch_cnt}")

    avg_loss = total_loss / max(1, batch_cnt)
    print(f"Epoch {epoch+1} (LoRA), Loss: {avg_loss:.4f}")

    log_memory_usage(f"After LoRA Epoch {epoch+1}")

lora_model.save_pretrained("llama60m_lora")
tokenizer.save_pretrained("llama60m_lora")

log_max_memory()
