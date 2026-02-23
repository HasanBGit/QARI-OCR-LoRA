"""
Training Script for Qari-OCR LoRA
Fine-tune Qari-OCR model using LoRA/DoRA
"""

import gc
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from datasets import load_from_disk

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from .config import TrainingConfig


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_processor(config: TrainingConfig):
    """Load the base model and processor with quantization."""
    print(f"Loading model: {config.model_id}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True
    )

    # Configure quantization
    bnb_config = None
    if config.load_in_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        print("  4-bit quantization enabled")

    # Model kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    # Flash Attention
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print("  Using SDPA")

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_id,
        **model_kwargs
    )

    print(f"Model loaded! Parameters: {model.num_parameters() / 1e9:.2f}B")

    return model, processor


def apply_lora(model, config: TrainingConfig):
    """Apply LoRA adapters to the model."""
    print("Applying LoRA/DoRA adapters...")

    # Prepare for k-bit training
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=config.use_dora,
        use_rslora=config.use_rslora,
    )

    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  DoRA={config.use_dora}, RSLoRA={config.use_rslora}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    return model, lora_config


@dataclass
class Qwen2VLDataCollator:
    """Data collator for Qwen2-VL training."""
    processor: any
    max_length: int = 2048

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_texts = []
        batch_images = []

        for example in examples:
            messages = example['messages']
            formatted_messages = []
            example_images = []

            for msg in messages:
                role_map = {
                    "<|User|>": "user",
                    "<|Assistant|>": "assistant",
                    "user": "user",
                    "assistant": "assistant",
                }
                role = role_map.get(msg['role'], msg['role'].lower())

                content = []
                images = msg.get('images', [])
                if images:
                    for img in images:
                        if img is not None:
                            content.append({"type": "image", "image": img})
                            example_images.append(img)

                text_content = msg.get('content', '')
                if text_content:
                    text_content = text_content.replace("<image>", "").strip()
                    if text_content:
                        content.append({"type": "text", "text": text_content})

                if content:
                    formatted_messages.append({"role": role, "content": content})

            text = self.processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            batch_texts.append(text)
            batch_images.extend(example_images)

        batch = self.processor(
            text=batch_texts,
            images=batch_images if batch_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        batch["labels"] = batch["input_ids"].clone()
        return batch


class ProgressCallback(TrainerCallback):
    """Callback for progress tracking."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append({
                "step": state.global_step,
                "loss": logs["loss"],
                "timestamp": datetime.now().isoformat()
            })
            print(f"Step {state.global_step} - Loss: {logs['loss']:.4f}")

    def on_save(self, args, state, control, **kwargs):
        state_info = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "loss_history": self.loss_history[-100:],
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.results_dir / "training_state.json", "w") as f:
            json.dump(state_info, f, indent=2)


def train_model(config: Optional[TrainingConfig] = None):
    """Main training function."""
    if config is None:
        config = TrainingConfig()

    # Create directories
    for dir_path in [config.output_dir, config.results_dir, config.logs_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("\n" + "=" * 60)
    print("QARI-OCR LORA TRAINING")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps}")
    print("=" * 60)

    # Load model
    clear_memory()
    model, processor = load_model_and_processor(config)

    # Apply LoRA
    model, lora_config = apply_lora(model, config)
    lora_config.save_pretrained(config.output_dir)

    # Load data
    print(f"\nLoading dataset from {config.train_dataset_path}")
    dataset = load_from_disk(config.train_dataset_path)
    print(f"Total samples: {len(dataset)}")

    # Split dataset
    shuffled = dataset.shuffle(seed=config.seed)
    split_idx = int(len(shuffled) * 0.95)
    train_dataset = shuffled.select(range(split_idx))
    eval_dataset = shuffled.select(range(split_idx, len(shuffled)))

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Data collator
    data_collator = Qwen2VLDataCollator(
        processor=processor,
        max_length=config.max_seq_length,
    )

    # Compute warmup steps
    total_steps = (len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="none",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        seed=config.seed,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback(config.results_dir)],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = Path(config.output_dir) / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    # Save training info
    training_info = config.to_dict()
    training_info["completed_at"] = datetime.now().isoformat()
    training_info["train_samples"] = len(train_dataset)
    training_info["eval_samples"] = len(eval_dataset)

    with open(final_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"\nTraining complete! Model saved to: {final_path}")

    return model, processor


if __name__ == "__main__":
    config = TrainingConfig()
    train_model(config)
