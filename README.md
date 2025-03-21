# Training Script Documentation

## Overview
This script is designed to facilitate model training with various configurations. Users can specify multiple parameters, including the training mode, optimizer, model type, and other training settings.

## Usage
```bash
python train.py --mode <mode> --optimizer <optimizer> --model <model> [other options]
```

## Input Parameters

### Required Parameters

| Parameter    | Type   | Choices                                  | Description                        |
|-------------|--------|------------------------------------------|------------------------------------|
| `--mode`    | string | `pretraining`, `finetuning`              | Specifies the training mode. |
| `--optimizer` | string | `lora`, `galore`, `galore8bit`, `lora+galore8bit`, `baseline` | Selects the optimizer type. |
| `--model`   | string | `llama_60m`, `llama_1b`, `llama_7b`, `roberta`, `gpt2` | Defines the model to train. |

### Optional Parameters

| Parameter        | Type  | Default  | Choices | Description |
|-----------------|------|----------|---------|-------------|
| `--batch_size`  | int  | `16`      | N/A     | Number of samples per batch. |
| `--num_epochs`  | int  | `30`      | N/A     | Number of training epochs. |
| `--max_length`  | int  | `512`     | N/A     | Maximum token length per input. |
| `--num_training_tokens` | int | `1e9`     | N/A     | Number of training tokens (only for pretraining). |
| `--shuffle`     | string | `true`   | `true`, `false` | Whether to shuffle training data (not applicable in streaming mode). |
| `--dtype`       | string | `fp16`   | `bf16`, `fp16` | Data type for training (currently only `bf16` is working). |
| `--lr`          | float | `4e-4`    | N/A     | Learning rate for optimizer. |
| `--weight_decay` | float | `0.01`   | N/A     | Weight decay for optimizer. |
| `--tmax`        | int  | `30`      | N/A     | Tmax for scheduler. |
| `--lora_config` | string | `config/lora_config.json` | N/A | Path to the LoRa configuration file. |
| `--galore_config` | string | `config/galore_config.json` | N/A | Path to the GaLore configuration file. |
| `--test`        | string | `false`  | `true`, `false` | Whether to enable test mode. Takes only 1000 tokens of dataset for pretraining and accelerator without bf16 (useful only for A100 GPUs). |

## Example Command

```bash
python train.py --mode pretraining --optimizer lora --model llama_1b --batch_size 32 --num_epochs 20 --shuffle false --lr 3e-4
```

This command runs the script in pretraining mode using the LoRa optimizer on the `llama_1b` model with a batch size of 32, 20 epochs, no data shuffling, and a learning rate of `3e-4`.

