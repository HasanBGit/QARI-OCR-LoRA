"""
Inference Module for Qari-OCR LoRA
Load model and transcribe Arabic manuscript images.
"""

import torch
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel


# Default OCR prompt
DEFAULT_PROMPT = "Below is the image of one page of a document. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."


def load_model(
    adapter_path: str = "HassanB4/Qari-OCR-LoRA",
    base_model: str = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
    load_in_4bit: bool = False,
    device_map: str = "auto",
):
    """
    Load the Qari-OCR model with LoRA adapter.

    Args:
        adapter_path: Path to LoRA adapter (local or HuggingFace Hub)
        base_model: Base model ID
        load_in_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model: {base_model}")

    # Model kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    # Configure quantization if requested
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print("  4-bit quantization enabled")

    # Flash Attention
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print("  Using SDPA")

    # Load base model
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        **model_kwargs
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )

    print("Model loaded successfully!")

    return model, processor


def transcribe_image(
    image: Union[str, Path, Image.Image],
    model,
    processor,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    do_sample: bool = False,
) -> str:
    """
    Transcribe a single image.

    Args:
        image: Image path or PIL Image
        model: Loaded model
        processor: Loaded processor
        prompt: OCR prompt
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling

    Returns:
        Transcribed text
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Create message format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    # Process inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

    # Decode (skip input tokens)
    input_len = inputs['input_ids'].shape[1]
    transcription = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

    return transcription


def transcribe_batch(
    images: List[Union[str, Path, Image.Image]],
    model,
    processor,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    show_progress: bool = True,
) -> List[str]:
    """
    Transcribe a batch of images.

    Args:
        images: List of image paths or PIL Images
        model: Loaded model
        processor: Loaded processor
        prompt: OCR prompt
        max_new_tokens: Maximum tokens to generate
        show_progress: Whether to show progress bar

    Returns:
        List of transcribed texts
    """
    from tqdm.auto import tqdm

    results = []
    iterator = tqdm(images, desc="Transcribing") if show_progress else images

    for image in iterator:
        try:
            text = transcribe_image(
                image=image,
                model=model,
                processor=processor,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            results.append(text)
        except Exception as e:
            print(f"Error processing image: {e}")
            results.append("")

        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Transcribe images with Qari-OCR LoRA')
    parser.add_argument('images', nargs='+', help='Image paths')
    parser.add_argument('--adapter', default='HassanB4/Qari-OCR-LoRA', help='LoRA adapter')
    parser.add_argument('--use-4bit', action='store_true', help='Use 4-bit quantization')

    args = parser.parse_args()

    model, processor = load_model(
        adapter_path=args.adapter,
        load_in_4bit=args.use_4bit,
    )

    for image_path in args.images:
        print(f"\n{'='*50}")
        print(f"Image: {image_path}")
        transcription = transcribe_image(image_path, model, processor)
        print(f"Transcription:\n{transcription}")
