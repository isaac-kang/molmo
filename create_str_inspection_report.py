#!/usr/bin/env python3
"""
Create detailed STR inspection reports similar to InternVL format
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_ground_truth(dataset_name: str, case_sensitive: bool = True) -> Dict[str, str]:
    """Load ground truth from LMDB file"""
    import os
    import lmdb
    
    # Use STR_DATA_DIR environment variable if set, otherwise default to case-sensitive
    str_data_dir = os.environ.get('STR_DATA_DIR', '~/data/STR/english_case-sensitive')
    str_data_dir = Path(str_data_dir).expanduser()
    lmdb_path = str_data_dir / f"lmdb/evaluation/{dataset_name}"
    
    gt_data = {}
    env = lmdb.open(str(lmdb_path), readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode()
            if key_str.startswith('label-'):
                # Convert label-000000001 to image-000000001
                image_id = key_str.replace('label-', 'image-')
                text = value.decode()
                gt_data[image_id] = text
    env.close()
    return gt_data

def normalize_text(text: str, case_sensitive: bool = True, ignore_punctuation: bool = True, ignore_spaces: bool = True) -> str:
    """Normalize text for comparison"""
    normalized = text
    
    if not case_sensitive:
        normalized = normalized.lower()
    
    if ignore_punctuation:
        import string
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    
    if ignore_spaces:
        normalized = normalized.replace(' ', '')
    
    return normalized

def _save_dataset_images(dataset_name: str, predictions_data: List[Dict], images_dir: Path):
    """Save input images for a dataset"""
    import lmdb
    from PIL import Image
    import io
    
    # Set up paths
    lmdb_path = Path(f"~/data/STR/english_case-sensitive/lmdb/evaluation/{dataset_name}").expanduser()
    
    # Create dataset-specific image directory (same as dataset directory)
    dataset_images_dir = images_dir / dataset_name
    dataset_images_dir.mkdir(exist_ok=True)
    
    # Open LMDB
    env = lmdb.open(str(lmdb_path), readonly=True)
    
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            
            for i, pred in enumerate(predictions_data):
                # Get image ID
                image_id = pred.get('image_id', str(i + 1))
                if 'image-' in image_id:
                    image_id = image_id.split('-')[-1]
                
                # Look for the image in LMDB
                key = f"image-{image_id.zfill(9)}".encode()
                value = txn.get(key)
                
                if value:
                    # Save image
                    image = Image.open(io.BytesIO(value)).convert('RGB')
                    image_filename = f"{dataset_name}_{i:04d}_{image_id}.png"
                    image_path = dataset_images_dir / image_filename
                    image.save(image_path)
                else:
                    print(f"Warning: Image not found for {image_id} in {dataset_name}")
    
    finally:
        env.close()
    
    print(f"✓ Saved {len(predictions_data)} images to {dataset_images_dir}")

def create_dataset_report(predictions_file: Path, dataset_name: str, model_name: str, 
                        prompt: str, case_sensitive: bool = True, 
                        ignore_punctuation: bool = True, ignore_spaces: bool = True,
                        save_images: bool = False, images_dir: Path = None) -> Tuple[int, int]:
    """Create detailed inspection report for a single dataset"""
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)
    
    # Load ground truth
    gt_data = load_ground_truth(dataset_name, case_sensitive)
    
    # Create output directory
    output_dir = Path("str_results") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process predictions
    results = []
    correct_count = 0
    
    for i, pred in enumerate(predictions_data):
        image_filename = pred.get('image_filename', f'sample_{i:04d}.png')
        model_answer = pred.get('prediction', '').strip()
        full_prompt = pred.get('prompt', '')
        
        # Extract image ID from filename or metadata
        image_id = pred.get('image_id', str(i + 1))
        if 'image-' in image_id:
            # Extract number from image-000000001 format
            image_id = image_id.split('-')[-1]
        
        ground_truth = gt_data.get(f'image-{image_id.zfill(9)}', '')
        
        # Normalize for comparison
        model_compare = normalize_text(model_answer, case_sensitive, ignore_punctuation, ignore_spaces)
        gt_compare = normalize_text(ground_truth, case_sensitive, ignore_punctuation, ignore_spaces)
        
        # Check if correct
        is_correct = model_compare == gt_compare
        if is_correct:
            correct_count += 1
        
        results.append({
            'sample_num': i + 1,
            'image_filename': image_filename,
            'image_id': image_id,
            'model_answer': model_answer,
            'ground_truth': ground_truth,
            'full_prompt': full_prompt,
            'is_correct': is_correct
        })
    
    # Save images if requested
    if save_images and images_dir:
        _save_dataset_images(dataset_name, predictions_data, images_dir)
    
    # Generate detailed report
    report_file = output_dir / "predictions.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Molmo STR LMDB Dataset Inspection - {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Matching - Case-sensitive: {case_sensitive}, Ignore punct: {ignore_punctuation}, Ignore space: {ignore_spaces}\n")
        f.write("=" * 100 + "\n\n")
        
        for result in results:
            f.write(f"Sample {result['sample_num']}/{len(results)}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Image:          {result['image_filename']}\n")
            f.write(f"Image ID:       {result['image_id']}\n")
            f.write(f"Full Prompt:    {result['full_prompt']}\n")
            f.write(f"Model Answer:   {result['model_answer'] if result['model_answer'] else 'None'}\n")
            f.write(f"Ground Truth:   {result['ground_truth']}\n")
            f.write(f"Correct:        {'✓' if result['is_correct'] else '✗'}\n\n")
    
    print(f"✓ Created detailed report for {dataset_name}: {correct_count}/{len(results)} = {correct_count/len(results)*100:.2f}%")
    return correct_count, len(results)

def create_summary_report(results: Dict[str, Tuple[int, int]], model_name: str, prompt: str,
                         case_sensitive: bool = True, ignore_punctuation: bool = True, 
                         ignore_spaces: bool = True):
    """Create summary report"""
    
    summary_file = Path("str_results") / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Molmo STR Benchmarks Summary\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Matching - Case-sensitive: {case_sensitive}, Ignore punct: {ignore_punctuation}, Ignore space: {ignore_spaces}\n")
        f.write("=" * 100 + "\n\n")
        
        total_correct = 0
        total_examples = 0
        
        for dataset_name, (correct, total) in results.items():
            accuracy = correct / total * 100 if total > 0 else 0
            f.write(f"{dataset_name:<15} : {correct:4d}/{total:4d} = {accuracy:6.2f}%\n")
            total_correct += correct
            total_examples += total
        
        f.write("\n" + "=" * 100 + "\n")
        if total_examples > 0:
            avg_accuracy = total_correct / total_examples * 100
            f.write(f"Average Accuracy: {avg_accuracy:.2f}%\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Created summary report: {total_correct}/{total_examples} = {total_correct/total_examples*100:.2f}%")

def main():
    # Configuration
    model_name = "Molmo-7B-D-0924"
    prompt = "What is the main word in the image? Output only the text."
    case_sensitive = False
    ignore_punctuation = True
    ignore_spaces = True
    
    # Find prediction files
    results_dir = Path("str_results")
    if not results_dir.exists():
        print("Error: str_results directory not found. Run evaluation first.")
        return
    
    # Look for prediction files
    prediction_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("predictions-"):
            prediction_dirs.append(item)
    
    if not prediction_dirs:
        print("Error: No prediction directories found in str_results")
        return
    
    # Process each prediction directory
    all_results = {}
    
    for pred_dir in prediction_dirs:
        predictions_file = pred_dir / "predictions.json"
        if not predictions_file.exists():
            print(f"Warning: {predictions_file} not found, skipping")
            continue
        
        # Extract dataset name from directory name
        # Handle formats like: predictions-CUTE80-validation-str_eval
        dataset_name = pred_dir.name.replace("predictions-", "")
        if "-validation-" in dataset_name:
            dataset_name = dataset_name.split("-validation-")[0]
        elif "-evaluation" in dataset_name:
            dataset_name = dataset_name.replace("-evaluation", "")
        
        try:
            correct, total = create_dataset_report(
                predictions_file, dataset_name, model_name, prompt,
                case_sensitive, ignore_punctuation, ignore_spaces
            )
            all_results[dataset_name] = (correct, total)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Create summary report
    if all_results:
        create_summary_report(all_results, model_name, prompt, case_sensitive, ignore_punctuation, ignore_spaces)
        print(f"\n✓ Created inspection reports for {len(all_results)} datasets")
    else:
        print("No datasets processed successfully")

if __name__ == "__main__":
    main()
