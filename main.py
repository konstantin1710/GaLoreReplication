from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
from logger import logger_init, log_memory_usage, log_max_memory
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from args import args
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

# TODO benchmarking mit glue / anderen benchmarks
# hyperparameter an config binden (lr abhängig von modell, pretraining etc.)
# Parameter variabel

def get_model(mode, model_name="roberta"):
    """ Creates model for Pretraining or Fine-Tuning """
    if mode == "pretraining":
        model_config = AutoConfig.from_pretrained("config/llama_60m.json") #TODO adjust
        model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16) #TODO torch_dtype for all models?
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None or model.config.pad_token_id == -1:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id

    elif mode == "finetuning":
        if model_name == "roberta":
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2, torch_dtype=torch.bfloat16) #TODO torch_dtype for all models?
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif model_name == "gpt2":
            model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side = "left")

            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError("Invalid model name. Choose 'roberta' or 'gpt2'")
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")
    
    return model, tokenizer

# param_groups = [{'params': non_galore_params}, 
#                 {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}]
# optimizer = GaLoreAdamW(param_groups, lr=0.01)

def get_optimizer(mode, model, optimizer_type):
    """ creates optimizer (GaLore or LoRa) """
    if optimizer_type == "baseline":
        return AdamW(model.parameters(), lr=5e-5, weight_decay=0.01), model #TODO adjust
    elif optimizer_type == "galore":
        return GaLoreAdamW(model.parameters(), lr=2e-4, weight_decay=0), model #TODO adjust lr
    elif optimizer_type == "galore8bit":
        return GaLoreAdamW8bit(model.parameters(), lr=2e-4, weight_decay=0), model #TODO adjust lr
    elif optimizer_type == "lora":
        if mode == "finetuning":
            target_modules = ["query", "value"]
        else:
            target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig( #TODO adjust
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return torch.optim.AdamW(model.parameters(), lr=4e-4), model #TODO adjust lr
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def train(device, accelerator, scheduler, model, optimizer, dataloader, num_epochs=30):
    """ training model """
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    log_memory_usage("Before Training")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_cnt = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            if args.mode == "pretraining":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            elif args.mode == "finetuning":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")

            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            batch_cnt += 1
            if batch_cnt % 20 == 0:
                log_memory_usage(f"After Batch {batch_cnt}")
        
        avg_loss = total_loss / max(1, batch_cnt)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        log_memory_usage(f"After Epoch {epoch+1}")
    
    return model

if __name__ == "__main__":
    if (args.test):
        print("Test mode")
        # activates streaming for datasets (only for pretraining)
        if args.mode == "pretraining":
            from load_data_test import load_data
        elif args.mode == "finetuning":
            from load_data import load_data
        else:
            raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")
        accelerator = Accelerator()
    else:
        from load_data import load_data
        # bf16 only useful for A100 GPUs
        accelerator = Accelerator(mixed_precision="bf16")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print(f"Using optimizer: {args.optimizer}")
    logger_init(args.optimizer)

    model, tokenizer = get_model(args.mode, args.model_name)

    dataset = load_data(args.mode, tokenizer)

    optimizer, model = get_optimizer(args.mode, model, args.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) #TODO adjust T_max

    if args.mode == "pretraining":
        dataloader = DataLoader(dataset, batch_size=8) # TODO adjust batch size
    elif args.mode == "finetuning":
        dataloader = DataLoader(dataset["train"], batch_size=16) # TODO adjust batch size, shuffle?
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")

    trained_model = train(device, accelerator, scheduler, model, optimizer, dataloader, num_epochs=30) #TODO adjust num_epochs

    model_path = f"llama60m_{args.optimizer}_pretrained" if args.mode == "pretraining" else f"{args.model_name}_{args.optimizer}_finetuned" #TODO adjust llama60m
    trained_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    log_max_memory()