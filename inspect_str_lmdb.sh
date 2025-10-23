#!/bin/bash
# Script to inspect STR LMDB benchmarks with Molmo
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export MOLMO_DATA_DIR=/data/isaackang/data/molmo
export STR_DATA_DIR=~/data/STR/english_case-sensitive

# Default: process all samples per dataset with optimized prompt
# Options:
#   --max_examples N           : Number of samples per dataset (default: -1 for all)
#   --device_batch_size N      : Batch size for inference (default: 4)
#   --seq_len N                : Max sequence length (default: 1536)
#   --max_new_tokens N         : Max new tokens to generate (default: 50)
#   --save_dir PATH            : Directory to save results (default: ./str_results)
#   --eval_name NAME           : Evaluation name prefix (default: str_eval)
#   CUTE80 SVT SVTP...         : Specific datasets to evaluate (space-separated)
#   --prompt "text"            : Custom prompt for OCR evaluation
#   --fsdp                     : Use FSDP for loading (helps with OOM)
#   --pbar                     : Show progress bar
#   --overwrite                : Overwrite existing results
#   --save_images              : Save input images to results directory
#   --checkpoint MODEL         : Different model

/data/isaackang/anaconda3/envs/molmo/bin/torchrun --nproc_per_node=1 --master_port 29510 eval_str_ocr.py \
    Molmo-7B-D-0924 \
    CUTE80 SVT SVTP IC13_857 IC15_1811 IIIT5k_3000 \
  --max_examples -1 \
  --device_batch_size 4 \
  --seq_len 1536 \
  --max_new_tokens 50 \
  --save_dir ./str_results \
  --eval_name str_eval \
  --overwrite \
  # --save_images \
  "$@"