#!/bin/bash
#SBATCH --job-name=llama60m_galore
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8        #TODO adjust
#SBATCH --mem=64G                #TODO adjust
#SBATCH --time=24:00:00          #TODO adjust
#SBATCH --partition=gpu          # GPU-Partition

module load cuda/11.8

accelerate launch --multi_gpu main.py
