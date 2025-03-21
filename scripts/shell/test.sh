#!/bin/bash

python main.py \
    --mode pretraining \
    --optimizer galore \
    --model llama_1b \
    --batch_size 8 \
    --num_epochs 30 \
    --num_training_tokens 1000000 \
    --max_length 512 \
    --shuffle true \
    --dtype bf16 \
    --lr 4e-4 \
    --weight_decay 0.01 \
    --tmax 30 \
    --test false
