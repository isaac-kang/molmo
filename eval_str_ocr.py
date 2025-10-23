#!/usr/bin/env python3
"""
STR OCR Evaluation Script for Molmo

This script evaluates a Molmo model on STR (Scene Text Recognition) datasets
stored in LMDB format using the existing evaluation framework.

Usage:
    torchrun --nproc_per_node=1 eval_str_ocr.py <checkpoint_path> <dataset_name> [options]

Example:
    torchrun --nproc_per_node=1 eval_str_ocr.py allenai/Molmo-7B-D-0924 IIIT5k_3000
    torchrun --nproc_per_node=1 eval_str_ocr.py allenai/Molmo-7B-D-0924 str
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from dataclasses import replace
from typing import cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from olmo.config import EvalConfig, FSDPConfig, FSDPWrapStrategy, FSDPPrecision
from olmo.torch_util import get_world_size
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment, )
from scripts.mm_eval import ModelEvaluator
from launch_scripts.utils import get_evaluation
from create_str_inspection_report import create_dataset_report, create_summary_report

log = logging.getLogger(__name__)

# Available STR datasets
STR_DATASETS = [
    "CUTE80", "IC03_860", "IC03_867", "IC13_1015", "IC13_857", 
    "IC15_1811", "IC15_2077", "IIIT5k_3000", "SVT", "SVTP"
]


class STRModelEvaluator(ModelEvaluator):
    """Custom ModelEvaluator that generates individual text files and shows results in terminal"""
    
    def __init__(self, config):
        super().__init__(config)
        self.dataset_results = []  # Use list to preserve order
        self.model_name = "Molmo-7B-D-0924"
        self.prompt = "What is the main word in the image? Output only the text."
        self.case_sensitive = False
        self.ignore_punctuation = True
        self.ignore_spaces = True
        self.save_images = False  # Will be set from command line args
    
    def run(self):
        """Override run method to add custom post-processing"""
        # Run the original evaluation
        super().run()
        
        # Generate individual text files and show results
        if len(self.config.evaluations) > 0:
            self._generate_individual_reports()
            self._show_terminal_results()
    
    def _generate_individual_reports(self):
        """Generate individual text files for each dataset in evaluation order"""
        results_dir = Path("str_results")
        if not results_dir.exists():
            return
        
        # Images will be saved in individual dataset directories
        
        # Process datasets in the same order as evaluation
        # This matches the order in the bash script: CUTE80 SVT SVTP IC13_857 IC15_1811 IIIT5k_3000
        evaluation_order = ["CUTE80", "SVT", "SVTP", "IC13_857", "IC15_1811", "IIIT5k_3000"]
        
        all_results = []  # Store all results for summary
        
        for dataset_name in evaluation_order:
            # Find the prediction directory for this dataset
            pred_dir = None
            for item in results_dir.iterdir():
                if item.is_dir() and item.name == f"predictions-{dataset_name}-validation-str_eval":
                    pred_dir = item
                    break
            
            if pred_dir is None:
                print(f"Warning: No prediction directory found for {dataset_name}")
                continue
                
            predictions_file = pred_dir / "predictions.json"
            if not predictions_file.exists():
                print(f"Warning: {predictions_file} not found for {dataset_name}")
                continue
            
            try:
                correct, total = create_dataset_report(
                    predictions_file, dataset_name, self.model_name, self.prompt,
                    self.case_sensitive, self.ignore_punctuation, self.ignore_spaces,
                    save_images=self.save_images, images_dir=results_dir if self.save_images else None
                )
                self.dataset_results.append((dataset_name, correct, total))
                all_results.append({
                    'dataset': dataset_name,
                    'correct': correct,
                    'total': total,
                    'accuracy': correct/total*100 if total > 0 else 0
                })
                print(f"✓ {dataset_name}: {correct}/{total} = {correct/total*100:.2f}%")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
        
        # Create summary file
        self._create_summary_file(all_results, results_dir)
    
    def _create_summary_file(self, all_results, results_dir):
        """Create a summary text file with all results"""
        summary_file = results_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("STR OCR Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Prompt: {self.prompt}\n")
            f.write(f"Case Sensitive: {self.case_sensitive}\n")
            f.write(f"Ignore Punctuation: {self.ignore_punctuation}\n")
            f.write(f"Ignore Spaces: {self.ignore_spaces}\n")
            f.write(f"Save Images: {self.save_images}\n\n")
            
            f.write("Results by Dataset:\n")
            f.write("-" * 30 + "\n")
            
            total_correct = 0
            total_examples = 0
            
            for result in all_results:
                f.write(f"{result['dataset']:<15} : {result['correct']:4d}/{result['total']:4d} = {result['accuracy']:6.2f}%\n")
                total_correct += result['correct']
                total_examples += result['total']
            
            f.write("-" * 30 + "\n")
            overall_accuracy = total_correct / total_examples * 100 if total_examples > 0 else 0
            f.write(f"{'OVERALL':<15} : {total_correct:4d}/{total_examples:4d} = {overall_accuracy:6.2f}%\n")
            
            f.write(f"\nTotal Examples: {total_examples}\n")
            f.write(f"Total Correct: {total_correct}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
            
            if self.save_images:
                f.write(f"\nImages saved to: {results_dir}/{{dataset}}/ directories\n")
        
        print(f"✓ Summary saved to: {summary_file}")
    
    def _show_terminal_results(self):
        """Show results in terminal like InternVL format"""
        if not self.dataset_results:
            return
        
        print("\n" + "=" * 80)
        print("✓ All evaluations complete!")
        print("✓ Summary saved to: str_results/summary.txt")
        print("=" * 80)
        print("\nSummary:")
        
        total_correct = 0
        total_examples = 0
        
        for dataset_name, correct, total in self.dataset_results:
            accuracy = correct / total * 100 if total > 0 else 0
            print(f"{dataset_name:<15} : {correct:4d}/{total:4d} = {accuracy:6.2f}%")
            total_correct += correct
            total_examples += total
        
        if total_examples > 0:
            avg_accuracy = total_correct / total_examples * 100
            print(f"Average: {avg_accuracy:.2f}%")
        print()


def main():
    parser = argparse.ArgumentParser(prog="Evaluate a model on STR OCR tasks")
    parser.add_argument("checkpoint",
                        help="Checkpoint to evaluate, should contain a config file and unshared model file")
    parser.add_argument("tasks", nargs="+", help="STR tasks to evaluate")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--seed", default=6198, type=int)
    parser.add_argument("--seq_len", default=1536, type=int,
                        help="Max sequence length to use")
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save the evaluation results")
    parser.add_argument("--save_to_checkpoint_dir", action="store_true",
                        help="Save to the checkpoint directory")
    parser.add_argument("--eval_name",
                        help="Name to use as a prefix when saving results")
    parser.add_argument("--pbar", action="store_true",
                        help="Show a progress bar")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true",
                        help="Load with FSDP, can be used to avoid OOMs")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Override max new tokens, otherwise use task-specific default")
    parser.add_argument("--prompt", default="What is the main word in the image? Output only the text.",
                        help="Custom prompt for OCR evaluation")
    parser.add_argument("--save_images", action="store_true",
                        help="Save input images to results directory")
    args, other_args = parser.parse_known_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    add_cached_path_clients()
    prepare_cli_environment()
    
    # Set custom prompt for STR datasets
    os.environ['STR_CUSTOM_PROMPT'] = args.prompt

    tasks = []
    for task in args.tasks:
        if task == "str":
            # STR OCR evaluation tasks
            tasks += [
                "CUTE80",
                "IC13_857", 
                "IC15_1811",
                "IIIT5k_3000",
                "SVT",
                "SVTP"
            ]
        elif "," in task:
            # Split comma-separated tasks
            tasks += [t.strip() for t in task.split(",")]
        else:
            tasks.append(task)
    tasks = list({k: None for k in tasks})  # de-duplicate but keep order

    inf_evaluators = []
    for task in tasks:
        eval_config = get_evaluation(
            name=task, seq_len=args.seq_len,
            batch_size=args.device_batch_size*get_world_size(),
            max_examples=args.max_examples,
        )
        if args.max_new_tokens:
            eval_config = replace(eval_config, max_new_tokens=args.max_new_tokens)
        inf_evaluators.append(replace(
            eval_config,
            mm_evaluator=replace(
                eval_config.mm_evaluator,
                n_to_log=4,
                num_wandb_examples=300,
                save_predictions="_default",
            ),
            save_to_checkpoint_dir=args.save_to_checkpoint_dir,
            save_dir=args.save_dir,
            eval_name=args.eval_name,
            skip_if_metrics_cached=not args.overwrite,
        ))

    checkpoint_dir = Path(args.checkpoint)
    if not (checkpoint_dir / "model.pt").exists() and args.checkpoint != "debug":
        candidates = []
        for file in checkpoint_dir.iterdir():
            match = re.match("^step([0-9]+)-unsharded.*", file.name)
            if match:
                candidates.append((file, int(match.group(1))))
        if len(candidates) == 0:
            raise FileNotFoundError(f"{checkpoint_dir} is a directory but it did not "
                                    f"contain any unsharded checkpoints")
        checkpoint_dir = max(candidates, key=lambda x: x[1])[0].absolute().as_posix()
        logging.info(f"Selected {checkpoint_dir} as oldest checkpoint in {checkpoint_dir}")
    else:
        checkpoint_dir = args.checkpoint

    cfg = EvalConfig(
        evaluations=inf_evaluators,
        load_path=checkpoint_dir,
        seed=args.seed,
        device_inf_eval_batch_size=args.device_batch_size,
        pbar=args.pbar,
        console_log_interval=10,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ) if args.fsdp else None,
    )

    # Skip OmegaConf merging for this script since we handle all arguments directly
    evaluator = STRModelEvaluator(cfg)
    evaluator.save_images = args.save_images
    evaluator.run()


if __name__ == "__main__":
    main()