from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from galore_torch import GaLoreAdamW
from accelerate import Accelerator
from load_data import tokenizer, tokenized_dataset
from logger import log_memory_usage, log_cpu_memory

accelerator = Accelerator()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model_config = AutoConfig.from_pretrained("llama_60m.json")
base_model = AutoModelForCausalLM.from_config(model_config)

lora_config = LoraConfig( #TODO adjust LoRA config
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)


lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

lora_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-4)

train_dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

lora_model, lora_optimizer, train_dataloader = accelerator.prepare(lora_model, lora_optimizer, train_dataloader)

log_memory_usage("Before LoRA Training")
log_cpu_memory("Before LoRA Training")

num_epochs = 3
lora_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        lora_optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        accelerator.backward(loss)
        lora_optimizer.step()

        total_loss += loss.item()

        log_memory_usage("After LoRA Batch")
        log_cpu_memory("After LoRA Batch")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} (LoRA), Loss: {avg_loss:.4f}")

    log_memory_usage(f"After LoRA Epoch {epoch+1}")
    log_cpu_memory(f"After LoRA Epoch {epoch+1}")

lora_model.save_pretrained("llama60m_lora")
tokenizer.save_pretrained("llama60m_lora")

print(f"Max Allocated (LoRA): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
