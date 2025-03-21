@echo off
python main.py ^
    --mode pretraining ^
    --optimizer galore ^
    --model llama_60m ^
    --batch_size 8 ^
    --num_epochs 30 ^
    --max_length 512 ^
    --num_training_tokens 1000000 ^
    --shuffle false ^
    --dtype bf16 ^
    --lr 4e-4 ^
    --weight_decay 0.01 ^
    --tmax 30 ^
    --test true