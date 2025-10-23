#!/bin/bash
# Script to inspect custom OCR dataset with Molmo
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export MOLMO_DATA_DIR=/data/isaackang/data/molmo
export STR_DATA_DIR=~/data/STR/english_case-sensitive

# Default: process all samples in custom dataset with optimized prompt
# Options:
#   --num_examples N            : Number of samples to evaluate (default: -1 for all)
#   --batch_size N              : Batch size for inference (default: 1)
#   --max_new_tokens N          : Max new tokens to generate (default: 10)
#   --save_dir PATH             : Directory to save results (default: ./ocr_results)
#   --split SPLIT               : Dataset split (train/validation/test, default: train)
#   --prompt "text"             : Custom prompt for OCR evaluation
#   --fsdp                      : Use FSDP for loading (helps with OOM)
#   --pbar                      : Show progress bar
#   --case-sensitive            : Use case-sensitive matching (default: False)
#   --ignore-punctuation        : Ignore punctuation in matching (default: True)
#   --ignore-spaces             : Ignore spaces in matching (default: True)
#   --checkpoint MODEL          : Different model

/data/isaackang/anaconda3/envs/molmo/bin/torchrun --nproc_per_node=1 --master_port 29511 eval_custom_ocr.py \
    Molmo-7B-D-0924 \
  --num_examples 10 \
  --batch_size 1 \
  --max_new_tokens 50 \
  --save_dir ./ocr_results \
  --split train \
  --prompt "What is the main word in the image? Output only the text." \
  --case-sensitive false \
  --ignore-punctuation true \
  --ignore-spaces true \
  "$@"
