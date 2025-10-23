# Molmo - Multimodal Language Model

This repository contains the Molmo multimodal language model implementation and evaluation scripts.

## Overview

Molmo is a multimodal language model that combines vision and language capabilities for various tasks including:
- OCR (Optical Character Recognition)
- Visual Question Answering (VQA)
- Image Captioning
- Pointing and Counting tasks

## Features

- **Native OLMo Integration**: Uses the native OLMo model architecture
- **HuggingFace Compatibility**: Supports HuggingFace transformers interface
- **STR Dataset Evaluation**: Built-in support for Scene Text Recognition datasets
- **Custom OCR Evaluation**: Flexible evaluation framework for custom OCR tasks
- **MoE Support**: Includes Mixture of Experts (MoE) model variants

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n molmo python=3.9
conda activate molmo

# Install dependencies
pip install -r requirements.txt
```

### Evaluation Scripts

#### STR Dataset Evaluation
```bash
# Evaluate on STR datasets (SVT, CUTE80, etc.)
./inspect_str_lmdb.sh
```

#### Custom OCR Evaluation
```bash
# Evaluate on custom OCR dataset
./inspect_custom_ocr.sh
```

## Scripts

- `eval_str_ocr.py`: Evaluate on STR datasets (SVT, CUTE80, IC13_857, IC15_1811, IIIT5k_3000, SVTP)
- `eval_custom_ocr.py`: Evaluate on custom OCR datasets
- `eval_custom_ocr_hf.py`: HuggingFace-based evaluation script
- `create_str_inspection_report.py`: Generate detailed evaluation reports

## Data Setup

The evaluation scripts expect STR datasets to be organized as:
```
~/data/STR/english_case-sensitive/
├── lmdb/
│   └── evaluation/
│       ├── SVT/
│       ├── CUTE80/
│       └── ...
└── images/
    └── evaluation/
        ├── SVT/
        ├── CUTE80/
        └── ...
```

Set the `STR_DATA_DIR` environment variable to point to your data directory:
```bash
export STR_DATA_DIR=~/data/STR/english_case-sensitive
```

## Model Variants

- **Molmo-7B-D-0924**: 7B parameter model
- **MolmoE-1B**: Mixture of Experts model with ~1B active parameters

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- LMDB
- Pillow
- NumPy
- Pandas

## License

[Add your license information here]

## Citation

[Add citation information if applicable]