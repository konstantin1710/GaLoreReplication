import torch
import psutil

def log_memory_usage(stage=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def log_cpu_memory(stage=""):
    mem = psutil.virtual_memory()
    print(f"[{stage}] CPU Memory Used: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB")
