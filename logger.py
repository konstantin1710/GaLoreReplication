import torch
import psutil
import csv
import math

CSV_FILE = "output.csv"

def init_csv():
    """Initialize CSV file with headers."""
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "training_step", "compute_time", "peak_memory_usage_history_GB", 
                         "peak_memory_usage_allocated_GB", "peak_memory_usage_reserved_GB", 
                         "loss", "perplexity"])

def measure_memory():
    """Measure memory usage from CUDA or CPU."""
    if torch.cuda.is_available():
        history = torch.cuda.memory._record_memory_history()
        peak_history = max([entry["allocated_bytes.all.current"] for entry in history]) / 1e9 if history else 0
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        max_reserved = torch.cuda.max_memory_reserved() / 1e9
    else:
        mem = psutil.virtual_memory()
        peak_history = mem.used / 1e9
        max_allocated = mem.used / 1e9
        max_reserved = mem.total / 1e9  # Total system memory

    return peak_history, max_allocated, max_reserved

def log_to_csv(epoch, step, compute_time, loss):
    """Log training metrics to CSV file."""
    peak_history, max_allocated, max_reserved = measure_memory()
    perplexity = math.exp(loss) if loss < 100 else float("inf")  # Avoid overflow

    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, step, compute_time, peak_history, max_allocated, max_reserved, loss, perplexity])
