from load_data import load_data
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
from logger import init_csv, log_to_csv
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from args import args
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
import json
import datetime

def get_model(args):
    """ Creates model for Pretraining or Fine-Tuning """
    if args.mode == "pretraining":
        model_config = AutoConfig.from_pretrained(f"config/{args.model}.json")
        if args.dtype == "bf16":
            model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_config(model_config)
        
        # in the galore project they say: 
        # "it doesn't matter which tokenizer we use, because we train from scratch
        # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice"
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None or model.config.pad_token_id == -1:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id

    elif args.mode == "finetuning":
        if args.model == "roberta":
            if args.dtype == "bf16":
                model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2, torch_dtype=torch.bfloat16)
            else:
                model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif args.model == "gpt2":
            model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side = "left")

            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError("Invalid model name. Choose 'roberta' or 'gpt2'")
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")
    
    return model, tokenizer

def load_lora_config(args):
    """Loads LoRa configuration from file"""
    with open(args.lora_config, "r") as f:
        lora_params = json.load(f)

    target_modules = lora_params["target_modules_finetuning"] if args.mode == "finetuning" else lora_params["target_modules_pretraining"]

    return LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        target_modules=target_modules
    )

def load_galore_config(args):
    """Loads GaLore configuration from file"""
    with open(args.galore_config, "r") as f:
        return json.load(f)

def get_optimizer(args, model):
    """Creates optimizer (GaLore, LoRa, or baseline AdamW)"""
    if args.optimizer == "baseline":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay), model
    elif args.optimizer in ["galore", "galore8bit"]:
        galore_config = load_galore_config(args)
        param_groups = [
            {"params": model.parameters(), **galore_config}
        ]
        optimizer_class = GaLoreAdamW if args.optimizer == "galore" else GaLoreAdamW8bit
        return optimizer_class(param_groups, lr=args.lr, weight_decay=args.weight_decay), model
    elif args.optimizer in ["lora", "lora+galore8bit"]:
        lora_config = load_lora_config(args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        if args.optimizer == "lora":
            return AdamW(model.parameters(), lr=args.lr), model
        else:
            galore_config = load_galore_config()
            param_groups = [
                {"params": model.parameters(), **galore_config}
            ]
            return GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay), model
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

def train(device, accelerator, scheduler, model, optimizer, dataloader, num_epochs):
    """ training model """
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_cnt = 0
        for batch in dataloader:
            optimizer.zero_grad()
            start_time = datetime.datetime.now()
            
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
            
            compute_time = (datetime.datetime.now() - start_time).total_seconds()
            log_to_csv(epoch + 1, batch_cnt + 1, compute_time, loss.item())

            total_loss += loss.item()
            batch_cnt += 1
        
        avg_loss = total_loss / max(1, batch_cnt)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    if (args.test == "true"):
        print("Test mode")
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print(f"Using optimizer: {args.optimizer}")
    init_csv()

    model, tokenizer = get_model(args)

    dataset = load_data(args, tokenizer)

    optimizer, model = get_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)

    shuffle = True if args.shuffle == "true" else False
    if args.mode == "pretraining":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    elif args.mode == "finetuning":
        dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=shuffle)
    else:
        raise ValueError("Invalid mode. Choose 'pretraining' or 'finetuning'")

    trained_model = train(device, accelerator, scheduler, model, optimizer, dataloader, num_epochs=args.num_epochs)

    file_name = f"{args.model}_{args.optimizer}_pretrained" if args.mode == "pretraining" else f"{args.model}_{args.optimizer}_finetuned"
    model_path = f"models/{file_name}"
    trained_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)