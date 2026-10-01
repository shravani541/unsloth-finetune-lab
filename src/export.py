"""
FastTune-Lab: Model Export & Deployment Engine.
Exports fine-tuned LoRA checkpoints to:
1. 16-bit / 4-bit standalone Hugging Face weights
2. Quantized GGUF formats (q4_k_m, q8_0, f16) for llama.cpp / Ollama
3. Automatically generates Ollama Modelfile
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="FastTune-Lab Export Engine")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/exported_model", help="Target export path")
    parser.add_argument(
        "--format",
        type=str,
        choices=["merged_16bit", "merged_4bit", "gguf", "all"],
        default="gguf",
        help="Export format target",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "q5_k_m", "f16"],
        help="GGUF quantization level",
    )
    parser.add_argument("--ollama", action="store_true", help="Generate Ollama Modelfile")
    return parser.parse_args()


def export_model():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("📦 FastTune-Lab: Exporting Fine-Tuned Model")
    print(f"Checkpoint: {args.model_path}")
    print(f"Format: {args.format} | Quantization: {args.quantization}")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is required for model export.")

    print("\n⏳ Loading base model and adapter checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # 1. 16-bit Float Merged
    if args.format in ["merged_16bit", "all"]:
        merged_16b_path = os.path.join(args.output_dir, "merged_16bit")
        print(f"\n💾 Saving 16-bit merged model to: {merged_16b_path}")
        model.save_pretrained_merged(merged_16b_path, tokenizer, save_method="merged_16bit")
        print("✅ 16-bit merged model export complete.")

    # 2. 4-bit Merged
    if args.format in ["merged_4bit", "all"]:
        merged_4b_path = os.path.join(args.output_dir, "merged_4bit")
        print(f"\n💾 Saving 4-bit merged model to: {merged_4b_path}")
        model.save_pretrained_merged(merged_4b_path, tokenizer, save_method="merged_4bit")
        print("✅ 4-bit merged model export complete.")

    # 3. GGUF Export
    if args.format in ["gguf", "all"]:
        gguf_path = os.path.join(args.output_dir, "gguf")
        print(f"\n⚡ Exporting GGUF ({args.quantization}) to: {gguf_path}")
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method=args.quantization)
        print("✅ GGUF export complete.")

    # 4. Generate Ollama Modelfile
    if args.ollama:
        modelfile_path = os.path.join(args.output_dir, "Modelfile")
        gguf_filename = f"unsloth.{args.quantization.upper()}.gguf"
        modelfile_content = f"""# Ollama Modelfile for FastTune-Lab Fine-Tuned LLM
FROM ./gguf/{gguf_filename}

# Runtime Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

# System Prompt Template
TEMPLATE \"\"\"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{{{ .Prompt }}}}

### Response:
\"\"\"
"""
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        print("\n" + "-" * 60)
        print(f"🦙 Ollama Modelfile generated at: {modelfile_path}")
        print("To run locally with Ollama:")
        print(f"  cd {args.output_dir}")
        print("  ollama create my-custom-model -f Modelfile")
        print("  ollama run my-custom-model")
        print("-" * 60)

    print("\n🎉 Export finished successfully!")


if __name__ == "__main__":
    export_model()
