#!/bin/bash

#SBATCH --job-name=galore_pretrain   # Name des Jobs
#SBATCH --nodes=1                    # 1 Knoten nutzen
#SBATCH --ntasks=1                    # 1 Aufgabe
#SBATCH --mem=32G                     # 32 GB RAM zuweisen
#SBATCH --cpus-per-task=4             # 4 CPU-Kerne pro Task
#SBATCH --gres=gpu:1                  # 1 GPU anfordern
#SBATCH --time=01:00:00               # Maximale Laufzeit von 1 Stunde

# Docker-Image von Docker Hub ausführen
srun \
    --container-image=docker://mcr.informatik.uni-halle.de#apcne/galore-replication \
    --container-name=ml-container \
    --container-mounts=/home/apzgb/Dokumente/GaLoreReplication:/workspace \
    bash -i -c "cd /workspace && chmod +x ./scripts/shell/pretrain.sh && ./scripts/shell/pretrain.sh"