"""
FastTune-Lab: Utility functions for dataset formatting, memory tracking,
and visual benchmark plotting.
"""

import os
import time
import yaml
import torch
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset

# Standard Alpaca Prompt Template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


class GPUMemoryTracker:
    """Context manager and tracker for monitoring peak GPU VRAM allocation."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.is_cuda = torch.cuda.is_available()
        self.start_time = 0.0
        self.end_time = 0.0
        self.peak_vram_gb = 0.0

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device_id)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.is_cuda:
            peak_bytes = torch.cuda.max_memory_allocated(self.device_id)
            self.peak_vram_gb = peak_bytes / (1024 ** 3)
        else:
            self.peak_vram_gb = 0.0

    @property
    def elapsed_time_sec(self) -> float:
        return self.end_time - self.start_time

    @staticmethod
    def get_current_vram_gb(device_id: int = 0) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        return 0.0


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_alpaca_dataset(dataset: Dataset, tokenizer, max_samples: Optional[int] = None) -> Dataset:
    """Formats Alpaca-style dataset using the standard template and tokenizer EOS."""
    eos_token = tokenizer.eos_token if hasattr(tokenizer, "eos_token") else "</s>"

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = ALPACA_PROMPT.format(instruction, input_text, output) + eos_token
            texts.append(text)
        return {"text": texts}

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    return dataset.map(formatting_prompts_func, batched=True)


def plot_benchmark_results(
    hf_vram: float,
    unsloth_vram: float,
    hf_throughput: float,
    unsloth_throughput: float,
    output_path: str = "outputs/benchmark_comparison.png",
):
    """Generates side-by-side comparison charts for VRAM and Throughput."""
    try:
        import matplotlib.pyplot as plt
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Colors
        c_hf = "#ef4444"
        c_unsloth = "#10b981"

        # VRAM Plot
        ax1.bar(["Standard HF PEFT", "Unsloth"], [hf_vram, unsloth_vram], color=[c_hf, c_unsloth], width=0.5)
        ax1.set_ylabel("Peak VRAM (GB)", fontsize=11, fontweight="bold")
        ax1.set_title("Peak VRAM Consumption (Lower is Better)", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        for i, v in enumerate([hf_vram, unsloth_vram]):
            ax1.text(i, v + 0.15, f"{v:.1f} GB", ha="center", fontweight="bold")

        # Throughput Plot
        ax2.bar(["Standard HF PEFT", "Unsloth"], [hf_throughput, unsloth_throughput], color=[c_hf, c_unsloth], width=0.5)
        ax2.set_ylabel("Throughput (tokens/sec)", fontsize=11, fontweight="bold")
        ax2.set_title("Training Speed (Higher is Better)", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)
        for i, v in enumerate([hf_throughput, unsloth_throughput]):
            ax2.text(i, v + 0.5, f"{v:.1f} tok/s", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"✅ Benchmark chart saved to {output_path}")
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
