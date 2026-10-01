"""
FastTune-Lab: Accelerated Fine-Tuning Pipeline powered by Unsloth.
Supports Llama 3, Mistral, Gemma, Qwen, and custom Alpaca/ChatML datasets.
"""

import os
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Import local utilities
from utils import load_config, format_alpaca_dataset, GPUMemoryTracker

def parse_args():
    parser = argparse.ArgumentParser(description="FastTune-Lab Fine-Tuning Script")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--model_name", type=str, default=None, help="Hugging Face model ID or path")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    return parser.parse_args()


def train(config_path: str, overrides: dict):
    print("=" * 60)
    print("🚀 Initializing FastTune-Lab Training Engine")
    print("=" * 60)

    # 1. Load Configurations
    cfg = load_config(config_path)
    if overrides.get("model_name"):
        cfg["model"]["name"] = overrides["model_name"]
    if overrides.get("max_steps"):
        cfg["training"]["max_steps"] = overrides["max_steps"]
    if overrides.get("batch_size"):
        cfg["training"]["per_device_train_batch_size"] = overrides["batch_size"]
    if overrides.get("output_dir"):
        cfg["training"]["output_dir"] = overrides["output_dir"]

    # 2. Check for Unsloth
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth is not installed. Please install it using: "
            "pip install 'unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git'"
        )

    print(f"\n📦 Loading Model: {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"].get("max_seq_length", 2048),
        dtype=cfg["model"].get("dtype", None),
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
    )

    # 3. Configure LoRA / QLoRA
    print("\n⚙️ Configuring Optimized LoRA Adapters...")
    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_cfg.get("random_state", 3407),
    )

    # 4. Load & Format Dataset
    dataset_name = cfg["dataset"]["name"]
    print(f"\n📚 Loading Dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split=cfg["dataset"].get("split", "train"))
    formatted_dataset = format_alpaca_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        max_samples=cfg["dataset"].get("max_samples", None),
    )
    print(f"Dataset formatted successfully ({len(formatted_dataset)} samples).")

    # 5. Training Arguments
    train_cfg = cfg["training"]
    output_dir = train_cfg.get("output_dir", "outputs/final_model")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        max_steps=train_cfg.get("max_steps", 60),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=train_cfg.get("logging_steps", 5),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        seed=train_cfg.get("seed", 3407),
        output_dir=output_dir,
        report_to="none",
    )

    # 6. SFTTrainer Execution
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"].get("max_seq_length", 2048),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("\n🔥 Starting Training with Unsloth Acceleration...")
    with GPUMemoryTracker() as tracker:
        trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("🎉 Training Completed Successfully!")
    print(f"⏱️ Total Time Elapsed: {tracker.elapsed_time_sec:.2f} seconds")
    print(f"💾 Peak VRAM Consumption: {tracker.peak_vram_gb:.2f} GB")
    print("=" * 60)

    # 7. Save Final Adapter
    print(f"\n💾 Saving fine-tuned LoRA model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Model weights and tokenizer saved.")


if __name__ == "__main__":
    args = parse_args()
    overrides = {
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
    }
    train(args.config, overrides)
