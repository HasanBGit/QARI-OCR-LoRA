#!/usr/bin/env python3
"""
Training Script for Qari-OCR LoRA
Run LoRA/DoRA fine-tuning of Qari-OCR model.

Usage:
    python scripts/train.py
    python scripts/train.py --dataset ./ARABIC_OCR_FINAL --output ./checkpoints/qari_lora
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import train_model, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Train Qari-OCR LoRA model')
    parser.add_argument('--dataset', type=str, help='Training dataset path')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lora-r', type=int, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, help='LoRA alpha')
    parser.add_argument('--no-dora', action='store_true', help='Disable DoRA')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig()

    # Override with command line args
    if args.dataset:
        config.train_dataset_path = args.dataset
    if args.output:
        config.output_dir = args.output
        config.results_dir = str(Path(args.output).parent / "results" / "qari_lora")
        config.logs_dir = str(Path(args.output).parent / "logs" / "qari_lora")
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.lora_alpha:
        config.lora_alpha = args.lora_alpha
    if args.no_dora:
        config.use_dora = False
    if args.no_4bit:
        config.load_in_4bit = False

    print("=" * 60)
    print("QARI-OCR LORA TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Run training
    train_model(config)


if __name__ == "__main__":
    main()
