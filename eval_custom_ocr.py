#!/usr/bin/env python3
"""
Custom OCR Evaluation Script for Molmo

This script evaluates a Molmo model on a custom OCR dataset and generates
a detailed inspection report in the format you specified.

Usage:
    torchrun --nproc_per_node=1 eval_custom_ocr.py <checkpoint_path> [options]

Example:
    torchrun --nproc_per_node=1 eval_custom_ocr.py allenai/Molmo-7B-D-0924 --split train --num_examples 10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from olmo import ModelConfig, Molmo
from olmo.data import build_mm_preprocessor
from olmo.config import EvalConfig, DatasetEvaluatorConfig, EvaluatorConfig
from olmo.util import prepare_cli_environment
from olmo.data.academic_datasets import Custom
from scripts.mm_eval import ModelEvaluator
from launch_scripts.utils import get_evaluation

# Default prompt for OCR
DEFAULT_PROMPT = "What is the main word in the image? Output only the text."


def create_inspection_report(predictions_file, labels_file, output_file, model_name, prompt, num_examples, case_sensitive=True, ignore_punctuation=False, ignore_spaces=True):
    """Create a detailed inspection report in the specified format"""
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)
    
    # Load ground truth
    with open(labels_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Create ground truth lookup
    gt_lookup = {item['image_filename']: item['text'] for item in ground_truth_data}
    
    # Process predictions
    results = []
    correct_count = 0
    
    for i, pred in enumerate(predictions_data):
        if i >= num_examples:
            break
            
        image_filename = pred['image_filename']
        model_answer = pred.get('prediction', '').strip()
        ground_truth = gt_lookup.get(image_filename, '')
        
        # Prepare strings for comparison
        model_compare = model_answer
        gt_compare = ground_truth
        
        if not case_sensitive:
            model_compare = model_compare.lower()
            gt_compare = gt_compare.lower()
        
        if ignore_punctuation:
            import string
            model_compare = model_compare.translate(str.maketrans('', '', string.punctuation))
            gt_compare = gt_compare.translate(str.maketrans('', '', string.punctuation))
        
        if ignore_spaces:
            model_compare = model_compare.replace(' ', '')
            gt_compare = gt_compare.replace(' ', '')
        
        # Check if correct
        is_correct = model_compare == gt_compare
        if is_correct:
            correct_count += 1
        
        results.append({
            'sample_num': i + 1,
            'image_filename': image_filename,
            'image_id': pred.get('image_id', i + 1),
            'model_answer': model_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct
        })
    
    # Calculate accuracy
    accuracy = (correct_count / len(results)) * 100 if results else 0
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("InternVL Custom OCR Dataset Inspection\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Case-sensitive: {case_sensitive}, Ignore punctuation: {ignore_punctuation}\n")
        f.write("=" * 100 + "\n\n")
        
        for result in results:
            f.write(f"Sample {result['sample_num']}/{len(results)}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Image:          {result['image_filename']}\n")
            f.write(f"Image ID:       {result['image_id']}\n")
            f.write(f"Prompt:         {prompt}\n")
            f.write(f"Model Answer:   {result['model_answer'] if result['model_answer'] else 'None'}\n")
            f.write(f"Ground Truth:   {result['ground_truth']}\n")
            f.write(f"Correct:        {'✓' if result['is_correct'] else '✗'}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("Inspection Complete!\n")
        f.write(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Inspection report saved to: {output_file}")
    print(f"✓ Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Molmo on custom OCR dataset")
    parser.add_argument("checkpoint", help="Path to model checkpoint or HuggingFace model ID")
    parser.add_argument("--split", default="train", help="Dataset split (train/validation/test)")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples to evaluate (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Custom OCR prompt")
    parser.add_argument("--save_dir", default="./ocr_results", help="Directory to save results")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument('--case-sensitive', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use case-sensitive matching (default: False)')
    parser.add_argument('--ignore-punctuation', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to ignore punctuation in matching (default: True)')
    parser.add_argument('--ignore-spaces', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to ignore spaces in matching (default: True)')
    parser.add_argument('--fsdp', action='store_true',
                        help='Load with FSDP, can be used to avoid OOMs')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set up multiprocessing and distributed training (like eval_str_ocr.py)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    logger.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    dist.init_process_group(backend="nccl")
    logger.info("Process group initialized")
    
    # Prepare environment
    prepare_cli_environment()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up evaluation using the same pattern as eval_downstream.py
    from olmo.torch_util import get_world_size
    from dataclasses import replace
    
    # Get evaluation configuration for custom dataset
    eval_config = get_evaluation(
        name="custom",
        seq_len=1536,  # Default sequence length
        batch_size=args.batch_size * get_world_size(),
        max_examples=args.num_examples,
    )
    
    # Override with custom settings
    eval_config = replace(
        eval_config,
        max_new_tokens=args.max_new_tokens,
        data=replace(eval_config.data, split=args.split, shuffle=False),  # Keep images in order!
        mm_evaluator=replace(
            eval_config.mm_evaluator,
            n_to_log=1000,  # Log all examples (use large number)
            num_wandb_examples=300,
            save_predictions="_default",
        ),
        save_dir=args.save_dir,
        skip_if_metrics_cached=False,  # Always overwrite
    )
    
    # Create the main evaluation configuration
    from olmo.config import FSDPConfig, FSDPWrapStrategy, FSDPPrecision
    
    cfg = EvalConfig(
        evaluations=[eval_config],
        load_path=args.checkpoint,
        seed=6198,
        device_inf_eval_batch_size=args.batch_size,
        pbar=args.pbar,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ) if args.fsdp else None,
    )

    # Run evaluation
    logger.info(f"Starting evaluation on custom OCR dataset...")
    logger.info(f"Model: {args.checkpoint}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max examples: {args.num_examples}")
    
    evaluator = ModelEvaluator(cfg)
    evaluator.run()
    
    # Find the predictions file
    predictions_dir = save_dir / f"predictions-custom-{args.split}"
    predictions_file = predictions_dir / "predictions.json"
    
    if not predictions_file.exists():
        logger.error(f"Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    # Find the labels file
    labels_file = Path("~/data/molmo/torch_datasets/example_custom_dataset/labels.json").expanduser()
    if not labels_file.exists():
        logger.error(f"Labels file not found: {labels_file}")
        sys.exit(1)
    
    # Create inspection report
    report_file = save_dir / "inspection_report.txt"
    create_inspection_report(
        predictions_file,
        labels_file,
        report_file,
        args.checkpoint,
        args.prompt,
        args.num_examples if args.num_examples > 0 else 1000,
        case_sensitive=args.case_sensitive,
        ignore_punctuation=args.ignore_punctuation,
        ignore_spaces=args.ignore_spaces
    )
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()