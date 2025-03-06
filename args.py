import argparse

parser = argparse.ArgumentParser(description="Run training")
parser.add_argument("--optimizer", type=str, choices=["lora", "galore", "galore8bit", "baseline"], required=True, help="Optimizer type to use")
parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True, help="Training mode to use")
parser.add_argument("--model_name", type=str, choices=["roberta", "gpt2"], default="roberta", help="Model zu use for fine-tuning")
parser.add_argument("--test", type=bool, default=False, help="Test mode")
args = parser.parse_args()
