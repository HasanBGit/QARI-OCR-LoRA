"""Training configuration for Qari-OCR LoRA"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for Qari-OCR LoRA training."""

    # Model
    model_id: str = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"
    model_tag: str = "qari_v0.3_lora"

    # LoRA/DoRA Configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_dora: bool = True
    use_rslora: bool = True
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence settings
    max_seq_length: int = 2048
    max_image_size: int = 1024

    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 5

    # Paths
    train_dataset_path: str = "./ARABIC_OCR_FINAL"
    output_dir: str = "./checkpoints/qari_lora"
    results_dir: str = "./results/qari_lora"
    logs_dir: str = "./logs/qari_lora"

    # Other
    seed: int = 42
    dataloader_num_workers: int = 4

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "model_id": self.model_id,
            "model_tag": self.model_tag,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
            "target_modules": self.target_modules,
            "load_in_4bit": self.load_in_4bit,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "max_image_size": self.max_image_size,
            "train_dataset_path": self.train_dataset_path,
            "output_dir": self.output_dir,
        }
