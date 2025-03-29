#!/bin/bash

python3 main.py \
    --mode pretraining \
    --optimizer galore \
    --model llama_1b \
    --batch_size 512 \
    --num_epochs 1 \
    --num_training_tokens 1000000 \
    --max_length 512 \
    --shuffle false \
    --dtype bf16 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --tmax 30 \
    --test false
