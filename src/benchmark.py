"""
FastTune-Lab: Empirical Performance & Memory Benchmark Suite.
Compares Unsloth vs Standard Hugging Face PEFT (BitsAndBytes 4-bit).
Outputs a visual comparison chart and JSON metrics report.
"""

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from utils import GPUMemoryTracker, format_alpaca_dataset, plot_benchmark_results

def parse_args():
    parser = argparse.ArgumentParser(description="FastTune-Lab Benchmark Suite")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--steps", type=int, default=30, help="Number of benchmark steps to run per method")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def benchmark_standard_hf(model_name: str, dataset, tokenizer, steps: int, batch_size: int, seq_length: int):
    """Runs standard Hugging Face PEFT 4-bit fine-tuning."""
    print("\n------------------------------------------------------------")
    print("▶️ [1/2] Benchmarking Standard Hugging Face PEFT (BitsAndBytes 4-bit)...")
    print("------------------------------------------------------------")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    hf_model = get_peft_model(hf_model, lora_config)

    training_args = TrainingArguments(
        output_dir="outputs/tmp_hf",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_steps=steps,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=hf_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=seq_length,
        args=training_args,
    )

    with GPUMemoryTracker() as tracker:
        trainer.train()

    del hf_model, trainer
    torch.cuda.empty_cache()

    tokens_processed = steps * batch_size * seq_length
    throughput = tokens_processed / max(tracker.elapsed_time_sec, 0.001)

    return {
        "vram_gb": tracker.peak_vram_gb,
        "time_sec": tracker.elapsed_time_sec,
        "throughput_tokens_per_sec": throughput,
    }


def benchmark_unsloth(model_name: str, dataset, steps: int, batch_size: int, seq_length: int):
    """Runs Unsloth FastLanguageModel fine-tuning."""
    print("\n------------------------------------------------------------")
    print("▶️ [2/2] Benchmarking Unsloth FastLanguageModel...")
    print("------------------------------------------------------------")

    from unsloth import FastLanguageModel

    unsloth_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_length,
        load_in_4bit=True,
    )

    unsloth_model = FastLanguageModel.get_peft_model(
        unsloth_model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_args = TrainingArguments(
        output_dir="outputs/tmp_unsloth",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_steps=steps,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=unsloth_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=seq_length,
        args=training_args,
    )

    with GPUMemoryTracker() as tracker:
        trainer.train()

    del unsloth_model, trainer
    torch.cuda.empty_cache()

    tokens_processed = steps * batch_size * seq_length
    throughput = tokens_processed / max(tracker.elapsed_time_sec, 0.001)

    return {
        "vram_gb": tracker.peak_vram_gb,
        "time_sec": tracker.elapsed_time_sec,
        "throughput_tokens_per_sec": throughput,
    }


def run_benchmark():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("🔬 FastTune-Lab: Running Controlled Optimization Benchmark")
    print(f"Model: {args.model_name} | Steps: {args.steps} | Batch Size: {args.batch_size}")
    print("=" * 60)

    # Prepare shared dataset & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw_data = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = format_alpaca_dataset(raw_data, tokenizer=tokenizer, max_samples=300)

    # 1. Run Standard HF PEFT
    hf_results = benchmark_standard_hf(
        args.model_name, dataset, tokenizer, args.steps, args.batch_size, args.seq_length
    )

    # 2. Run Unsloth
    unsloth_results = benchmark_unsloth(
        args.model_name, dataset, args.steps, args.batch_size, args.seq_length
    )

    # Calculations
    vram_reduction = ((hf_results["vram_gb"] - unsloth_results["vram_gb"]) / max(hf_results["vram_gb"], 0.001)) * 100
    speedup = unsloth_results["throughput_tokens_per_sec"] / max(hf_results["throughput_tokens_per_sec"], 0.001)

    results_summary = {
        "model": args.model_name,
        "steps": args.steps,
        "standard_hf": hf_results,
        "unsloth": unsloth_results,
        "vram_reduction_pct": round(vram_reduction, 2),
        "throughput_speedup_factor": round(speedup, 2),
    }

    # Save Results
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    chart_path = os.path.join(args.output_dir, "benchmark_comparison.png")
    plot_benchmark_results(
        hf_vram=hf_results["vram_gb"],
        unsloth_vram=unsloth_results["vram_gb"],
        hf_throughput=hf_results["throughput_tokens_per_sec"],
        unsloth_throughput=unsloth_results["throughput_tokens_per_sec"],
        output_path=chart_path,
    )

    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"HF Peak VRAM:       {hf_results['vram_gb']:.2f} GB")
    print(f"Unsloth Peak VRAM:  {unsloth_results['vram_gb']:.2f} GB  (🔽 {vram_reduction:.1f}% reduction)")
    print(f"HF Throughput:      {hf_results['throughput_tokens_per_sec']:.1f} tok/s")
    print(f"Unsloth Throughput: {unsloth_results['throughput_tokens_per_sec']:.1f} tok/s  (⚡ {speedup:.2f}x faster)")
    print(f"\n📁 Report saved to: {json_path}")
    print(f"🖼️ Chart saved to:  {chart_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
