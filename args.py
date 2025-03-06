import argparse

parser = argparse.ArgumentParser(description="Run training")
parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True, help="Training mode to use")
parser.add_argument("--optimizer", type=str, choices=["lora", "galore", "galore8bit", "baseline"], required=True, help="Optimizer type to use")
parser.add_argument("--model", type=str, choices=["llama_60m", "llama_1b", "llama_7b", "roberta", "gpt2"], required=True, help="Model to use")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--max_length", type=int, default=512, help="Max length of input tokens")
parser.add_argument("--train_split", type=float, default=10, help="Percentage of training data to use")
parser.add_argument("--shuffle", type=str, choices=["true", "false"], default="true", help="Shuffle data (doesn't work in streaming mode)")
parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="fp16", help="Data type to use") # TODO for now just bf16 working
parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate for optimizer")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
parser.add_argument("--tmax", type=int, default=30, help="Tmax for scheduler")
parser.add_argument("--lora_config", type=str, default="config/lora_config.json", help="Path to LoRa config file")
parser.add_argument("--test", type=str, choices=["true", "false"], default="false", help="Test mode")
args = parser.parse_args()
