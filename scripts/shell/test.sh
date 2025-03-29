#!/bin/bash

python3 main.py \
    --mode pretraining \
    --optimizer galore \
    --model llama_60m \
    --batch_size 2 \
    --num_epochs 3 \
    --num_training_tokens 100 \
    --max_length 512 \
    --shuffle false \
    --dtype bf16 \
    --lr 4e-4 \
    --weight_decay 0.01 \
    --tmax 30 \
    --test true
