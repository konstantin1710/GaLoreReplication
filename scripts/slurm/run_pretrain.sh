#!/bin/bash
#SBATCH --job-name=galore_pretrain
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8        #TODO adjust
#SBATCH --mem=128G                #TODO adjust
#SBATCH --partition=gpu          # GPU-Partition
#SBATCH --output=logs/%j.out             # Log-File
#SBATCH --error=logs/%j.err              # Error-Log-File
#SBATCH --nodelist=workg01               # Bestimmter Knoten

srun --container-image=docker://pytorch/pytorch --container-name=ml-container \
     --container-mounts ~/Dokumente/GaLoreReplication:/workspace \
     --gres=gpu:1 --cpus-per-gpu=16 --mem-per-cpu=4G \
     --pty bash -i -c "cd /workspace && python train.py --optimizer=galore" #TODO adjust
