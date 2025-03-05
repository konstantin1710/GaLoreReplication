import argparse

parser = argparse.ArgumentParser(description="Run training")
parser.add_argument("--optimizer", type=str, choices=["lora", "galore", "galore8bit", "baseline"], required=True, help="Optimizer type to use")
parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True, help="Training mode to use")
parser.add_argument("--test", type=bool, default=True, help="Test mode") # TODO: adjust default value
args = parser.parse_args()
