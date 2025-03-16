#!/bin/bash

#SBATCH --job-name=galore_pretrain
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:10:00 

# Ensure the log directory exists
mkdir -p logs

# Build the Docker image
docker build -t galore_image -f /home/apzgb/Dokumente/GaLoreReplication/Dockerfile /home/apzgb/Dokumente/GaLoreReplication

# Run the Docker container with the appropriate resources
srun --container-image=galore_image --container-name=ml-container \
     --container-mounts /home/apzgb/Dokumente/GaLoreReplication:/workspace \
     --gres=gpu:1 --cpus-per-task=4 --mem=32G \
     --pty bash -i -c "cd /workspace && python train.py --optimizer=galore"