import torch
import psutil
import datetime


def logger_init():
    with open("output.txt", "a") as f:
        f.write(f"\nLogging started at {datetime.datetime.now()}\n")

def log_memory_usage(stage=""):
    with open("output.txt", "a") as f:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            f.write(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB\n")
        else:
            mem = psutil.virtual_memory()
            f.write(f"[{stage}] CPU Memory Used: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB\n")

def log_max_memory():
    with open("output.txt", "a") as f:
        if torch.cuda.is_available():
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            max_reserved = torch.cuda.max_memory_reserved() / 1e9
            f.write(f"Max GPU Memory - Allocated: {max_allocated:.2f} GB, Reserved: {max_reserved:.2f} GB\n")
        else:
            mem = psutil.virtual_memory()
            f.write(f"Max CPU Memory Used: {mem.total / 1e9:.2f} GB\n")
