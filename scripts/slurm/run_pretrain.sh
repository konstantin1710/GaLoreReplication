#!/bin/bash

#SBATCH --job-name=galore_pretrain   # Name des Jobs
#SBATCH --nodes=1                    # 1 Knoten nutzen
#SBATCH --ntasks=1                    # 1 Aufgabe
#SBATCH --mem=32G                     # 32 GB RAM zuweisen
#SBATCH --cpus-per-task=4             # 4 CPU-Kerne pro Task
#SBATCH --gres=gpu:1                  # 1 GPU anfordern
#SBATCH --time=00:05:00               # Maximale Laufzeit von 5 Minuten

# Set TMPDIR to a different directory
export TMPDIR=/home/apzgb/tmp

# Docker-Image von Docker Hub ausführen
srun \
    --container-image=nvidia/cuda:12.3.0-base-ubuntu20.04 \
    --container-name=ml-container \
    --container-mounts=/home/apzgb/Dokumente/GaLoreReplication:/workspace \
    bash -i -c "cd /workspace && chmod +x ./scripts/shell/pretrain.sh"