# Qari-OCR-LoRA: Additional Model for NakbaNLP 2026 Shared Task

<p align="center">
<img src="https://placehold.co/800x200/e0f2fe/0369a1?text=Qari-Arabic-OCR" alt="Qari Arabic OCR">
</p>

This repository contains an **additional experimental model** — a LoRA fine-tuned **Qari-OCR** — developed during the **[NakbaNLP 2026 Shared Task](https://acrps.ai/nakba-nlp-manu-understanding-2026)** on Arabic Manuscript Understanding (Subtask 2: Systems Track).

> **Note:** This is **not** our main submission. Our primary winning model is **[Ketaba-OCR](https://huggingface.co/HassanB4/Ketab-OCR-LoRA)** which achieved 1st place with CER 0.0819.

#### By: [Hassan Barmandah](https://scholar.google.com/citations?user=XXXXX), [Fatimah Emad Eldin](https://scholar.google.com/citations?user=CfX6eA8AAAAJ&hl=ar), [Khloud Al Jallad](https://scholar.google.com/citations?user=XXXXX), [Omar Nacer](https://scholar.google.com/citations?user=XXXXX)

[![Main Model](https://img.shields.io/badge/Main_Model-Ketab--OCR-green)](https://huggingface.co/HassanB4/Ketab-OCR-LoRA)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Page-F9D371)](https://huggingface.co/HassanB4/Qari-OCR-LoRA)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## Model Description

This is an **additional experimental model** that fine-tunes **Qari-OCR** using **Low-Rank Adaptation (LoRA)** with DoRA and RSLoRA for Arabic handwritten text recognition. The base model is NAMAA-Space's Qari-OCR v0.3, built on Qwen2-VL-2B architecture.

While this model achieves reasonable results (CER 0.2635 on blind test), our **main submission [Ketaba-OCR](https://huggingface.co/HassanB4/Ketab-OCR-LoRA)** significantly outperforms it with CER 0.0819.

The model transcribes cropped line images from Arabic manuscripts into machine-readable text, optimized for the **Omar Al-Saleh Memoir Collection** (1951-1965) written in Ruq'ah and Naskh script variants.

### Key Features

* **Parameter Efficiency**: LoRA fine-tuning with only ~37.6M trainable parameters (1.67% of total)
* **DoRA + RSLoRA**: Weight-Decomposed Low-Rank Adaptation with rank stabilization for improved training
* **Lightweight Base**: 2.2B parameter model (Qwen2-VL-2B) for faster inference
* **Experimental**: Additional model for comparison with our main HRT-based approach

---

## 🚀 How to Use

You can use the fine-tuned model directly with the `transformers` and `peft` libraries.

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image

# Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
 
# Load LoRA adapter
model = PeftModel.from_pretrained(model, "HassanB4/Qari-OCR-LoRA")
model.eval()

# Load processor
processor = AutoProcessor.from_pretrained(
    "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
    trust_remote_code=True
)

# Example inference
image = Image.open("manuscript_line.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Below is the image of one page of a document. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

transcription = processor.decode(output_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(transcription)
```

---

## ⚙️ Training Procedure

The system employs LoRA/DoRA fine-tuning of the Qari-OCR model.

### Training Data

The model was fine-tuned on the official NakbaNLP 2026 dataset from the Omar Al-Saleh Memoir Collection:

| Split | Samples | Description |
| :--- | :---: | :--- |
| Training | 15,163 | Line images with gold transcriptions (95%) |
| Validation | 799 | Line images for evaluation (5%) |
| Dev Test | 2,095 | Development test set |
| Blind Test | 2,671 | Held-out for official evaluation |

### Hyperparameters

| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Base Model** | NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct | **Architecture** | Qwen2-VL-2B |
| **Model Size** | ~2.21B parameters | **Trainable Params** | 37.6M (1.67%) |
| **LoRA Rank (r)** | 32 | **LoRA Alpha (α)** | 64 |
| **Target Modules** | q, k, v, o, gate, up, down | **LoRA Dropout** | 0.05 |
| **DoRA** | True | **RSLoRA** | True |
| **Learning Rate** | 2×10⁻⁴ | **Optimizer** | AdamW (fused) |
| **LR Scheduler** | Cosine | **Warmup Ratio** | 0.03 |
| **Batch Size** | 2 (per GPU) | **Gradient Accumulation** | 8 |
| **Effective Batch** | 16 | **Number of Epochs** | 3 |
| **Max Gradient Norm** | 1.0 | **Weight Decay** | 0.01 |
| **Max Sequence Length** | 2048 | **Precision** | bfloat16 |

### Frameworks

* PyTorch 2.5+
* Hugging Face Transformers ≥4.45.0
* PEFT ≥0.14.0
* bitsandbytes ≥0.43.0

---

## 📊 Evaluation Results

The model was evaluated on both development and blind test sets provided by the NakbaNLP 2026 organizers.

### Test Set Scores

| Dataset | CER | WER | Samples |
| :--- | :---: | :---: | :---: |
| Development Test | 0.5413 | 0.8873 | 2,095 |
| **Blind Test** | **0.2635** | **0.5521** | 2,671 |

### Comparison with Other Models

| Model | Blind CER | Blind WER | Notes |
| :--- | :---: | :---: | :--- |
| **Ketaba-OCR (Our Main Model)** | **0.0819** | **0.2588** | 1st Place Winner |
| Qari-OCR LoRA (This Model) | 0.2635 | 0.5521 | Additional experiment |
| Qari-OCR v0.3 (Zero-Shot) | 0.300 | 0.485 | Base model |
| Arabic OCR 4-bit v2 (Sherif) | 0.3234 | 0.6203 | — |
| Qwen2.5-VL-7B (Zero-Shot) | 0.6808 | 0.9198 | — |

---

## ⚠️ Limitations

* **Domain Specificity**: Optimized for 1950s Ruq'ah/Naskh manuscripts; requires adaptation for other periods/styles
* **Higher Error Rate**: CER of 0.26 is higher than the HRT-based approach (0.08), suggesting the smaller model capacity limits performance
* **Degraded Images**: Performance degrades on severely faded or damaged manuscript regions
* **No Ensemble**: Results are from a single model without ensemble techniques

---

## 🙏 Acknowledgements

We thank the **NakbaNLP 2026 organizers** for access to the Omar Al-Saleh Memoir Collection. We acknowledge **NAMAA-Space** for the Qari-OCR pretrained model, and the **Hugging Face** community for PEFT libraries.

### Related Links

* [NakbaNLP 2026 Workshop](https://sina.birzeit.edu/nakba-nlp/2026/)
* [AR-MS Shared Task Website](https://acrps.ai/nakba-nlp-manu-understanding-2026)
* [Base Model (Qari-OCR v0.3)](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct)

---

## 📜 Citation

If you use this work, please cite our main paper:

```bibtex
@inproceedings{barmandah2026ketaba,
    title={{Ketaba-OCR at NakbaNLP 2026 Shared Task: Efficient Adaptation of Vision-Language Models for Handwritten Text Recognition}},
    author={Barmandah, Hassan and Eldin, Fatimah Emad and Al Jallad, Khloud and Nacer, Omar},
    year={2026},
    booktitle={Proceedings of the 2nd International Workshop on Nakba Narratives as Language Resources (NakbaNLP 2026)},
    publisher={RASD}
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 License.
